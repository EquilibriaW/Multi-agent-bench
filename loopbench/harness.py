"""
loopbench.harness

Deterministic multi-agent orchestration baseline:
- planner bootstrap
- coder implementation via DB-backed assign/claim loop
- review rounds with cherry-pick merge + public validation
- rework tasks via DB-backed parallel execution
- final patch generation
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

from .agents import RoleDriver, RoleRunResult
from .coordination import ClaimedTask, CoordinationDB
from .events import EventLogger
from .interfaces import MultiAgentHarness, ToolRouter
from .path_utils import safe_path_component
from .role_actions import (
    assignment_deviations,
    extract_role_action_plan,
    iter_assignment_paths,
    path_allowed_by_assignment,
    paths_from_unified_patch,
)
from .knowledge_surfaces import KnowledgeSurfaces
from .knowledge_tool_context import build_knowledge_tool_context
from .review_diff_context import (
    build_candidate_merge_commits,
    build_review_diff_tool_context,
    extract_inline_diffs,
)
from .review_logic import (
    PublicValidationRecord,
    PublicValidationState,
    PublicValidationSummary,
    ReviewDecision,
    evaluate_public_validation,
    extract_review_decision,
    is_noop_public_validation,
    inspected_review_commits_by_role,
    nominated_commits_without_review_inspection,
    public_validation_state,
    public_validate_policy,
    review_round_has_accepted_work,
    verification_command_stats,
)
from .run_artifacts import ReviewLedger, ReviewLedgerEntry, RunArtifacts
from .schema import Budget, TaskPack, ToolCall
from .shell import ensure_success, run_command
from .team_protocol import TeamProtocol
from .time_utils import now_ms


@dataclass
class HarnessState:
    merged_commits: Dict[str, Set[str]] = field(default_factory=dict)
    review_round: int = 0
    merge_conflicts: int = 0
    public_validation_attempts: int = 0
    public_validation_noop_rounds: int = 0
    role_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class WorkerPhaseOutcome:
    role: str
    claimed: int = 0
    completed: int = 0
    failed: int = 0
    warnings: List[str] = field(default_factory=list)
    role_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DeterministicHarness(MultiAgentHarness):
    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        role_paths: Dict[str, Path],
        role_drivers: Dict[str, RoleDriver],
        base_commit: str,
        max_review_rounds: int,
        event_logger: EventLogger,
        reflection_enabled: bool = False,
    ):
        self.run_id = run_id
        self.run_dir = run_dir
        self.role_paths = role_paths
        self.role_drivers = role_drivers
        self.base_commit = base_commit
        self.max_review_rounds = max_review_rounds
        self.event_logger = event_logger
        self.reflection_enabled = reflection_enabled
        self._tools: ToolRouter | None = None

        self.planner = self._pick_planner(role_paths)
        self.coders = [r for r in role_paths if r != self.planner]

        self.artifacts = RunArtifacts(self.run_dir)
        self.team = TeamProtocol(
            db=CoordinationDB(self.run_dir / "coordination" / "coordination.db"),
            planner=self.planner,
            coders=self.coders,
        )
        self.ledger = ReviewLedger(self.run_dir / "review_ledger.json")
        self.knowledge: Optional[KnowledgeSurfaces] = None
        if self.reflection_enabled:
            self.knowledge = KnowledgeSurfaces(self.run_dir / "knowledge")

    def run(self, task: TaskPack, budget: Budget, tools) -> Dict[str, Any]:
        state = HarnessState(merged_commits={coder: set() for coder in self.coders})
        start_monotonic = time.monotonic()
        self._run_start_epoch = time.time()
        self._budget = budget
        self._tools = tools

        self.artifacts.append_status(f"run {self.run_id} started")
        self._init_coordination()

        self._bootstrap_phase(task, state)
        self._implementation_phase(task, state)

        public_pass = self._review_phase(task, state, tools)

        final_patch_path = self._finalize_phase(task, state)
        coordination_summary = self.team.summary()
        workflow_summary_path = self.artifacts.write_workflow_summary(
            task_id=task.task_id,
            planner_role=self.planner,
            coder_roles=list(self.coders),
            coordination_db_path=str(self.team.db_path),
            plan_path=str(self.run_dir / "plans" / "plan.md"),
            subtasks_path=str(self.run_dir / "plans" / "subtasks.yaml"),
            review_rounds_executed=state.review_round,
            public_validate_policy=public_validate_policy(getattr(task.substrate, "public_validate_policy", None)),
            public_validation_attempts=state.public_validation_attempts,
            public_validation_noop_rounds=state.public_validation_noop_rounds,
            merge_conflicts=state.merge_conflicts,
            merged_commits=state.merged_commits,
            role_outputs=state.role_outputs,
            coordination_summary=coordination_summary,
            public_pass=public_pass,
            final_patch_path=str(final_patch_path),
        )

        elapsed = time.monotonic() - start_monotonic
        metrics = {
            "review_iterations": state.review_round,
            "merge_conflicts": state.merge_conflicts,
            "public_validation_attempts": state.public_validation_attempts,
            "public_validation_noop_rounds": state.public_validation_noop_rounds,
            "wall_clock_sec": elapsed,
        }
        for key, value in coordination_summary.items():
            metrics[f"coordination_{key}"] = value

        self.artifacts.append_status(f"run {self.run_id} completed; public_pass={public_pass}")

        return {
            "final_patch_path": str(final_patch_path),
            "workflow_summary_path": str(workflow_summary_path),
            "public_pass": public_pass,
            "metrics": metrics,
            "role_outputs": state.role_outputs,
        }

    def _init_coordination(self) -> None:
        self.team.initialize()
        self.artifacts.append_status(f"coordination initialized: sqlite at {self.team.db_path}")

    def _bootstrap_phase(self, task: TaskPack, state: HarnessState) -> None:
        self.event_logger.begin_span("bootstrap", name="phase.bootstrap", tags=["phase", "bootstrap"])
        context = {
            "task_id": task.task_id,
            "task_kind": task.kind,
            "phase": "bootstrap",
            "role": self.planner,
            "coders": self.coders,
            "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
        }
        result = self._invoke_role(self.planner, "bootstrap", context, raise_on_failure=False, span_id="bootstrap")
        state.role_outputs[f"{self.planner}:bootstrap"] = result.output
        self.artifacts.write_role_summary(
            role=self.planner,
            phase="bootstrap",
            result=result,
        )
        if not result.ok:
            self.artifacts.append_open_question(
                "planner bootstrap failed; falling back to default plan/subtasks for continued loop execution"
            )

        plan_path = self.run_dir / "plans" / "plan.md"
        subtasks_path = self.run_dir / "plans" / "subtasks.yaml"
        default_plan = (
            f"# Plan for {task.task_id}\n\n"
            "1. Inspect task requirements and affected modules.\n"
            "2. Implement core functional changes in coder_a.\n"
            "3. Implement supporting tests, env, and validation updates in coder_b.\n"
            "4. Review and merge coder commits in planner_reviewer.\n"
            "5. Run public validation and iterate until green.\n"
        )

        plan_md = result.output.get("plan_markdown")
        if isinstance(plan_md, str) and plan_md.strip():
            plan_path.write_text(plan_md, encoding="utf-8")
        else:
            plan_path.write_text(default_plan, encoding="utf-8")

        subtasks = result.output.get("subtasks")
        if not isinstance(subtasks, list) or not subtasks:
            coder_a = self.coders[0] if self.coders else "coder_a"
            coder_b = self.coders[1] if len(self.coders) > 1 else coder_a
            subtasks = [
                {
                    "id": "S1",
                    "role": coder_a,
                    "title": "Core implementation",
                    "paths": ["."],
                    "acceptance": "Core behavior implemented according to task prompt.",
                },
                {
                    "id": "S2",
                    "role": coder_b,
                    "title": "Validation and integration",
                    "paths": ["tests", "Dockerfile", "."],
                    "acceptance": "Validation or environment adjustments aligned with implementation.",
                },
            ]

        subtasks = self._normalize_subtask_roles(subtasks)
        subtasks = self._collapse_overlapping_paths(subtasks)
        subtasks_path.write_text(yaml.safe_dump({"subtasks": subtasks}, sort_keys=False), encoding="utf-8")

        if self.knowledge is not None:
            self.knowledge.seed_from_bootstrap(
                plan_md=plan_path.read_text(encoding="utf-8"),
                subtasks=subtasks,
            )

        seeded_count = self.team.seed_implementation(
            task_id=task.task_id,
            plan_path=plan_path,
            subtasks_path=subtasks_path,
            subtasks=subtasks,
        )
        self.artifacts.append_status(
            f"bootstrap finished: plan + subtasks written, seeded implementation tasks={seeded_count}"
        )
        self.event_logger.end_span("bootstrap")

    def _normalize_subtask_roles(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize subtask roles to valid coder names.

        If the planner outputs generic roles like "coder" instead of "coder_a"/"coder_b",
        round-robin assign them to actual coder names.
        """
        valid_roles = set(self.coders)
        needs_fix = any(str(s.get("role", "")).strip() not in valid_roles for s in subtasks)
        if not needs_fix:
            return subtasks

        self.artifacts.append_open_question(
            "bootstrap subtasks had invalid role names; normalizing to actual coder names"
        )
        rr_idx = 0
        for subtask in subtasks:
            role = str(subtask.get("role", "")).strip()
            if role not in valid_roles:
                subtask["role"] = self.coders[rr_idx % len(self.coders)]
                rr_idx += 1
        return subtasks

    def _collapse_overlapping_paths(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """If multiple coders are assigned overlapping *specific* file paths, reassign all to one coder.

        Two coders editing the same file guarantees cherry-pick merge conflicts.
        Wildcard paths like "." or "tests" are ignored for overlap detection —
        they're planning shortcuts, not real file-level conflicts.
        """
        if len(self.coders) < 2:
            return subtasks
        # Broad/vague paths that planners use as shorthand — not real overlap signals.
        _VAGUE_PATHS = {".", "tests", "test", "src", "lib", "app", "config"}

        paths_by_role: Dict[str, Set[str]] = {}
        for s in subtasks:
            role = str(s.get("role", ""))
            paths = set(str(p) for p in s.get("paths", []))
            paths_by_role.setdefault(role, set()).update(paths)

        roles = [r for r in paths_by_role if r in set(self.coders)]
        if len(roles) < 2:
            return subtasks

        # Only check overlap on specific paths (not vague top-level dirs).
        specific_by_role = {
            r: {p for p in ps if p not in _VAGUE_PATHS}
            for r, ps in paths_by_role.items()
            if r in set(self.coders)
        }
        has_overlap = False
        role_list = list(specific_by_role.keys())
        for i, r1 in enumerate(role_list):
            for r2 in role_list[i + 1:]:
                if specific_by_role[r1] & specific_by_role[r2]:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if not has_overlap:
            return subtasks

        primary = self.coders[0]
        self.artifacts.append_open_question(
            f"bootstrap subtasks have overlapping file paths across coders; "
            f"reassigning all to {primary} to avoid merge conflicts"
        )
        for s in subtasks:
            s["role"] = primary
        return subtasks

    def _implementation_phase(self, task: TaskPack, state: HarnessState) -> None:
        self.event_logger.begin_span("implementation", name="phase.implementation", tags=["phase", "implementation"])
        planner_summary = self.artifacts.read_text(self.run_dir / "plans" / "plan.md")[:4000]
        phase_results = self._run_parallel_claim_phase(
            task=task,
            state=state,
            coordination_phase="implementation",
            driver_phase="implementation",
            planner_summary=planner_summary,
            round_index=None,
            extra_context={},
            parent_span_id="implementation",
        )
        self._record_worker_phase_outcomes(
            label="implementation phase",
            phase_results=phase_results,
        )
        self.event_logger.end_span("implementation")

    def _review_phase(self, task: TaskPack, state: HarnessState, tools) -> bool:
        public_policy = public_validate_policy(getattr(task.substrate, "public_validate_policy", None))
        accepted_public_pass = False
        last_validation = public_validation_state(
            policy=public_policy,
            state=PublicValidationState.NOT_RUN,
            stderr="not run",
        )

        for round_index in range(1, self.max_review_rounds + 1):
            state.review_round = round_index
            if self._should_rebootstrap(state):
                self._rebootstrap(task, state)
            round_span = f"review_round_{round_index}"
            self.event_logger.begin_span(
                round_span,
                name=f"review.round_{round_index}",
                tags=["review", f"round_{round_index}"],
            )
            self.artifacts.append_status(f"review round {round_index} started")

            planner_context: Dict[str, Any] = {
                "task_id": task.task_id,
                "phase": "review",
                "review_stage": "select",
                "round": round_index,
                "runtime_suffix": f"round_{round_index}",
                "coder_commits": {coder: self._list_commits(coder, self.base_commit) for coder in self.coders},
                "last_public_validation": last_validation.as_dict(),
                "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
                "coordination_summary": self.team.summary(),
                # Preserve legacy key name consumed by drivers, but include
                # recent rework/review messages so planner decisions see the
                # latest coder task lifecycle signals.
                "implementation_messages": self._planner_review_messages(round_index=round_index),
                "latest_coder_outputs": self._latest_coder_outputs(state),
                "review_ledger": self.ledger.as_context(),
            }
            if self.knowledge is not None:
                directive = self.knowledge.directive()
                if directive:
                    planner_context["reflection_directive"] = directive
                planner_context["knowledge_tool"] = build_knowledge_tool_context(
                    role_path=self.role_paths[self.planner],
                    knowledge_dir=self.run_dir / "knowledge",
                )
            # Only present unmerged commits as candidates — already-merged
            # commits waste context and can't be re-merged anyway.
            unmerged_by_role = {
                coder: [c for c in planner_context["coder_commits"].get(coder, [])
                        if c not in state.merged_commits.get(coder, set())]
                for coder in self.coders
            }
            candidate_merge_commits = build_candidate_merge_commits(
                role_paths=self.role_paths,
                coders=self.coders,
                coder_commits_by_role=unmerged_by_role,
            )
            planner_context["candidate_merge_commits"] = candidate_merge_commits
            planner_context["inline_diffs"] = extract_inline_diffs(
                role_paths=self.role_paths,
                coders=self.coders,
                candidate_merge_commits=candidate_merge_commits,
            )
            # Include diffs from already-merged commits so the reviewer sees
            # cumulative work, not just the latest rework delta.
            if state.merged_commits:
                already_merged_by_role = {
                    coder: [c for c in planner_context["coder_commits"].get(coder, [])
                            if c in state.merged_commits.get(coder, set())]
                    for coder in self.coders
                }
                merged_candidate = build_candidate_merge_commits(
                    role_paths=self.role_paths,
                    coders=self.coders,
                    coder_commits_by_role=already_merged_by_role,
                )
                merged_diffs = extract_inline_diffs(
                    role_paths=self.role_paths,
                    coders=self.coders,
                    candidate_merge_commits=merged_candidate,
                )
                if merged_diffs:
                    planner_context["already_merged_diffs"] = merged_diffs
            planner_context["review_diff_tool"] = build_review_diff_tool_context(
                role_paths=self.role_paths,
                planner=self.planner,
                round_index=round_index,
                candidate_merge_commits=candidate_merge_commits,
            )
            planner_context["scratch_merge_results"] = self._scratch_merge_test(
                state,
                nominated_commits_by_role=unmerged_by_role,
            )
            review_select_span = f"review_select_{round_index}"
            self.event_logger.begin_span(
                review_select_span,
                parent_span_id=round_span,
                name=f"step.review_select",
                tags=["step", "review_select"],
            )
            planner_result = self._invoke_role(
                self.planner,
                "review",
                planner_context,
                raise_on_failure=False,
                span_id=review_select_span,
            )
            self.event_logger.end_span(review_select_span)
            state.role_outputs[f"{self.planner}:review:{round_index}"] = planner_result.output
            self.artifacts.write_role_summary(
                role=self.planner,
                phase=f"review_round_{round_index}",
                result=planner_result,
            )
            if not planner_result.ok:
                self.artifacts.append_open_question(
                    f"planner review round {round_index} failed; continuing with merge/validation loop"
                )

            review_decision = extract_review_decision(
                planner_output=planner_result.output,
                available_commits_by_role=planner_context["coder_commits"],
                coder_roles=set(self.coders),
                resolve_symbolic_commit=self._resolve_symbolic_commit_token,
            )
            inspected_commits_by_role = inspected_review_commits_by_role(
                planner_output=planner_result.output,
                available_commits_by_role=planner_context["coder_commits"],
                coder_roles=set(self.coders),
                resolve_symbolic_commit=self._resolve_symbolic_commit_token,
            )
            # Merge inspections reported by agentic tool loop (git_show/git_diff_files calls).
            agentic_inspections = planner_result.output.get("inspected_commits", {})
            if isinstance(agentic_inspections, dict):
                for role_name, commit_list in agentic_inspections.items():
                    if not isinstance(role_name, str):
                        continue
                    bucket = inspected_commits_by_role.setdefault(role_name, set())
                    if isinstance(commit_list, (list, set)):
                        bucket.update(str(c) for c in commit_list)
            force_rework = review_decision.request_rework
            structured_valid = bool(planner_result.output.get("openrouter_structured_valid", True))
            if not structured_valid:
                self.artifacts.append_open_question(
                    f"planner review round {round_index} produced invalid structured output; "
                    "continuing with best-effort review parse."
                )
            if review_decision.merge_commits_provided and not review_decision.merge_commits_valid:
                self.artifacts.append_open_question(
                    f"planner review round {round_index} used malformed merge_commits; expected role->list[str]."
                )
            if review_decision.invalid_merge_roles:
                self.artifacts.append_open_question(
                    f"planner review round {round_index} referenced unknown merge roles: "
                    + ", ".join(review_decision.invalid_merge_roles[:8])
                    + (" ..." if len(review_decision.invalid_merge_roles) > 8 else "")
                )
            for role_name, unresolved_tokens in review_decision.unresolved_merge_tokens_by_role.items():
                if not unresolved_tokens:
                    continue
                token_preview = ", ".join(unresolved_tokens[:8])
                if len(unresolved_tokens) > 8:
                    token_preview = f"{token_preview}, ..."
                self.artifacts.append_open_question(
                    f"planner review round {round_index} referenced unresolved commits for {role_name}: {token_preview}"
                )
            if not review_decision.dynamic_checks_ran:
                self.artifacts.append_open_question(
                    f"planner review round {round_index} ran no dynamic checks; review may be shallow."
                )
            missing_inspections = nominated_commits_without_review_inspection(
                inspected_commits_by_role=inspected_commits_by_role,
                nominated_commits_by_role=review_decision.merge_commits_by_role,
                already_merged_commits_by_role=state.merged_commits,
            )
            if missing_inspections:
                review_decision.uninspected_nominated_commits_by_role = missing_inspections
                for role_name, missing_commits in missing_inspections.items():
                    commit_preview = ", ".join(commit[:12] for commit in missing_commits[:8])
                    if len(missing_commits) > 8:
                        commit_preview = f"{commit_preview}, ..."
                    self.artifacts.append_open_question(
                        f"planner review round {round_index} nominated commits without review_diff_tool evidence "
                        f"for {role_name}: {commit_preview}"
                    )
                # Only merge commits explicitly inspected in this round.
                for role_name, missing_commits in missing_inspections.items():
                    current = review_decision.merge_commits_by_role.get(role_name, [])
                    if not current:
                        continue
                    missing = set(missing_commits)
                    filtered = [sha for sha in current if sha not in missing]
                    if filtered:
                        review_decision.merge_commits_by_role[role_name] = filtered
                    else:
                        review_decision.merge_commits_by_role.pop(role_name, None)
                self.artifacts.append_open_question(
                    f"planner review round {round_index} dropped uninspected commit nominations; "
                    "re-run review with review_diff_tool show/files before nominating merges."
                )

            nominated_counts = {
                coder: len(review_decision.merge_commits_by_role.get(coder, []))
                for coder in self.coders
            }
            self.artifacts.append_status(
                "review round "
                f"{round_index} nominations: "
                + ", ".join(f"{coder}={count}" for coder, count in nominated_counts.items())
            )

            merge_span = f"merge_{round_index}"
            self.event_logger.begin_span(
                merge_span,
                parent_span_id=round_span,
                name="step.merge",
                tags=["step", "merge"],
            )
            merged_before_by_role = {
                coder: set(state.merged_commits[coder])
                for coder in self.coders
            }
            merged_before = sum(len(commits) for commits in state.merged_commits.values())
            merge_ok = self._merge_new_commits(
                state,
                nominated_commits_by_role=review_decision.merge_commits_by_role,
            )
            merged_after = sum(len(commits) for commits in state.merged_commits.values())
            merged_commits_added = merged_after > merged_before
            merged_commits_this_round: Dict[str, List[str]] = {}
            for coder in self.coders:
                new_commits = state.merged_commits[coder] - merged_before_by_role.get(coder, set())
                if new_commits:
                    merged_commits_this_round[coder] = sorted(new_commits)
            if not merge_ok:
                state.merge_conflicts += 1
                merged_commits_added = False
                self.artifacts.append_open_question(
                    f"merge conflict encountered in review round {round_index}"
                )
            self.event_logger.end_span(merge_span)

            validate_span = f"validate_{round_index}"
            self.event_logger.begin_span(
                validate_span,
                parent_span_id=round_span,
                name="step.public_validate",
                run_type="tool",
                tags=["step", "public_validate"],
            )
            validation_summary = self._run_public_validation_round(
                round_index=round_index,
                task=task,
                state=state,
                tools=tools,
                public_policy=public_policy,
            )
            last_validation = validation_summary.last_validation
            public_validation_passed = validation_summary.passed
            public_validation_available = validation_summary.available
            self.event_logger.end_span(validate_span)

            if not merge_ok:
                force_rework = True

            if public_policy == "required" and not public_validation_passed:
                force_rework = True
                if public_validation_available:
                    self.artifacts.append_open_question(
                        f"review round {round_index} policy=required and public validation failed."
                    )
                else:
                    self.artifacts.append_open_question(
                        f"review round {round_index} policy=required but public validation is unavailable/no-op."
                    )

            accepted_work_present = review_round_has_accepted_work(
                planner_output=planner_result.output,
                merged_commits_added=merged_commits_added,
            )
            if not force_rework and not accepted_work_present:
                force_rework = True
                self.artifacts.append_open_question(
                    f"review round {round_index} accepted no coder commits and planner made no direct edits; "
                    "forcing actionable rework."
                )

            # Always run verify so the reviewer (tech lead) can give targeted
            # feedback — even when the harness already detected issues like
            # merge conflicts or validation failures.
            harness_issues: List[str] = []
            if not merge_ok:
                harness_issues.append("Merge conflict: cherry-pick of nominated commits failed.")
            if public_policy == "required" and not public_validation_passed:
                harness_issues.append("Public validation failed (policy=required).")
            if not accepted_work_present:
                harness_issues.append("No coder commits were accepted or merged this round.")

            verify_span = f"verify_{round_index}"
            self.event_logger.begin_span(
                verify_span,
                parent_span_id=round_span,
                name="step.review_verify",
                tags=["step", "review_verify"],
            )
            verify_decision = self._run_review_verify(
                task=task,
                round_index=round_index,
                state=state,
                last_validation=last_validation,
                candidate_merge_commits=planner_context["candidate_merge_commits"],
                review_diff_tool=planner_context["review_diff_tool"],
                harness_issues=harness_issues,
                span_id=verify_span,
            )
            self.event_logger.end_span(verify_span)
            if verify_decision.request_rework:
                force_rework = True
            if verify_decision.coder_feedback:
                merged_feedback = dict(review_decision.coder_feedback)
                merged_feedback.update(verify_decision.coder_feedback)
                review_decision.coder_feedback = merged_feedback

            # --- Reflection phase (LLM-driven knowledge distillation) ---
            if self.knowledge is not None:
                reflect_span = f"reflect_{round_index}"
                self.event_logger.begin_span(
                    reflect_span,
                    parent_span_id=round_span,
                    name="step.reflect",
                    tags=["step", "reflect"],
                )
                self._run_reflection(
                    task=task,
                    round_index=round_index,
                    state=state,
                    last_validation=last_validation,
                    review_decision=review_decision,
                    merged_commits_this_round=merged_commits_this_round,
                    span_id=reflect_span,
                )
                self.event_logger.end_span(reflect_span)

            verify_output = state.role_outputs.get(
                f"{self.planner}:review_verify:{round_index}",
                {},
            )
            audit_kwargs = {
                "planner": self.planner,
                "round_index": round_index,
                "review_output": planner_result.output,
                "verify_output": verify_output,
                "decision": review_decision,
                "inspected_commits_by_role": inspected_commits_by_role,
                "merged_commits_this_round": merged_commits_this_round,
                "merge_ok": merge_ok,
                "public_validation": last_validation,
            }
            rework_kwargs = {
                "task": task,
                "round_index": round_index,
                "validation_stdout": last_validation.stdout,
                "validation_stderr": last_validation.stderr,
                "state": state,
                "planner_feedback_by_role": review_decision.coder_feedback,
                "public_policy": public_policy,
            }
            if not force_rework:
                self.artifacts.write_review_round_audit(
                    **audit_kwargs,
                    accepted=True,
                    force_rework=False,
                )
                self.ledger.append(ReviewLedgerEntry(
                    round_index=round_index,
                    decision="accept",
                    commits_merged=merged_commits_this_round,
                    open_issues=list(review_decision.coder_feedback.values()),
                    coder_feedback_map=dict(review_decision.coder_feedback),
                    validation_passed=public_validation_passed,
                    merge_ok=merge_ok,
                    summary=str(planner_result.output.get("summary", "")),
                    cause="accept",
                ))
                if public_validation_passed:
                    accepted_public_pass = True
                self.event_logger.end_span(round_span)
                break

            if not merged_commits_added and not review_decision.coder_feedback:
                self.artifacts.append_status(
                    "review round "
                    f"{round_index} produced no actionable merge/feedback; re-running planner review "
                    "(no merged commits and no targeted coder feedback)"
                )
                self.artifacts.append_status(
                    f"review round {round_index} dispatching coder rework despite non-actionable review "
                    "(forced by merge/public-validation constraints)"
                )
                self.artifacts.write_review_round_audit(
                    **audit_kwargs,
                    accepted=False,
                    force_rework=True,
                )
                _rework_cause = "merge_conflict" if not merge_ok else (
                    "validation_fail" if (public_policy == "required" and not public_validation_passed) else
                    "no_accepted_work"
                )
                self.ledger.append(ReviewLedgerEntry(
                    round_index=round_index,
                    decision="rework",
                    commits_merged=merged_commits_this_round,
                    open_issues=list(review_decision.coder_feedback.values()),
                    coder_feedback_map=dict(review_decision.coder_feedback),
                    validation_passed=public_validation_passed,
                    merge_ok=merge_ok,
                    summary=str(planner_result.output.get("summary", "")),
                    cause=_rework_cause,
                    validation_stderr_tail=last_validation.stderr[-1000:],
                ))
                rework_span = f"rework_{round_index}"
                self.event_logger.begin_span(
                    rework_span,
                    parent_span_id=round_span,
                    name="step.rework",
                    tags=["step", "rework"],
                )
                self._request_rework_from_coders(**rework_kwargs, parent_span_id=rework_span)
                self.event_logger.end_span(rework_span)
                self.event_logger.end_span(round_span)
                continue

            if public_policy == "required" and not public_validation_passed:
                self.artifacts.append_status(
                    f"review round {round_index} requested coder rework due to required public validation policy"
                )
            elif public_validation_passed:
                self.artifacts.append_status(
                    f"review round {round_index} requested coder rework despite public validation pass"
                )
            elif public_policy == "off":
                self.artifacts.append_status(
                    f"review round {round_index} requested coder rework with public validation disabled"
                )
            else:
                self.artifacts.append_status(
                    f"review round {round_index} requested coder rework from reviewer findings"
                )

            self.artifacts.write_review_round_audit(
                **audit_kwargs,
                accepted=False,
                force_rework=True,
            )
            _rework_cause = "merge_conflict" if not merge_ok else (
                "validation_fail" if (public_policy == "required" and not public_validation_passed) else
                "reviewer_rework"
            )
            self.ledger.append(ReviewLedgerEntry(
                round_index=round_index,
                decision="rework",
                commits_merged=merged_commits_this_round,
                open_issues=list(review_decision.coder_feedback.values()),
                coder_feedback_map=dict(review_decision.coder_feedback),
                validation_passed=public_validation_passed,
                merge_ok=merge_ok,
                summary=str(planner_result.output.get("summary", "")),
                cause=_rework_cause,
                validation_stderr_tail=last_validation.stderr[-1000:],
            ))

            rework_span = f"rework_{round_index}"
            self.event_logger.begin_span(
                rework_span,
                parent_span_id=round_span,
                name="step.rework",
                tags=["step", "rework"],
            )
            self._request_rework_from_coders(**rework_kwargs, parent_span_id=rework_span)
            self.event_logger.end_span(rework_span)
            self.event_logger.end_span(round_span)

        return accepted_public_pass

    def _run_review_verify(
        self,
        *,
        task: TaskPack,
        round_index: int,
        state: HarnessState,
        last_validation: PublicValidationRecord,
        candidate_merge_commits: Dict[str, List[Dict[str, Any]]],
        review_diff_tool: Dict[str, Any],
        harness_issues: List[str] | None = None,
        span_id: str | None = None,
    ) -> ReviewDecision:
        # Build coder_commits and inline_diffs so the verify driver can
        # inspect actual diff content (same data the review-select phase gets).
        coder_commits = {
            coder: self._list_commits(coder, self.base_commit)
            for coder in self.coders
        }
        verify_inline_diffs = extract_inline_diffs(
            role_paths=self.role_paths,
            coders=self.coders,
            candidate_merge_commits=candidate_merge_commits,
        )
        verify_context: Dict[str, Any] = {
            "task_id": task.task_id,
            "phase": "review_verify",
            "review_stage": "verify",
            "round": round_index,
            "runtime_suffix": f"round_{round_index}_verify",
            "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
            "coordination_summary": self.team.summary(),
            "last_public_validation": last_validation.as_dict(),
            "latest_coder_outputs": self._latest_coder_outputs(state),
            "review_ledger": self.ledger.as_context(),
            "candidate_merge_commits": candidate_merge_commits,
            "review_diff_tool": review_diff_tool,
            "coder_commits": coder_commits,
            "inline_diffs": verify_inline_diffs,
            "implementation_messages": self._planner_review_messages(round_index=round_index),
            "scratch_merge_results": self._scratch_merge_test(
                state,
                nominated_commits_by_role={
                    coder: [c for c in coder_commits.get(coder, [])
                            if c not in state.merged_commits[coder]]
                    for coder in self.coders
                },
            ),
        }
        # Add already_merged_diffs so verify sees cumulative state
        if state.merged_commits:
            already_merged_by_role = {
                coder: [c for c in coder_commits.get(coder, [])
                        if c in state.merged_commits.get(coder, set())]
                for coder in self.coders
            }
            merged_candidate = build_candidate_merge_commits(
                role_paths=self.role_paths,
                coders=self.coders,
                coder_commits_by_role=already_merged_by_role,
            )
            merged_diffs = extract_inline_diffs(
                role_paths=self.role_paths,
                coders=self.coders,
                candidate_merge_commits=merged_candidate,
            )
            if merged_diffs:
                verify_context["already_merged_diffs"] = merged_diffs
        if harness_issues:
            verify_context["harness_issues"] = harness_issues
        if self.knowledge is not None:
            directive = self.knowledge.directive()
            if directive:
                verify_context["reflection_directive"] = directive
            verify_context["knowledge_tool"] = build_knowledge_tool_context(
                role_path=self.role_paths[self.planner],
                knowledge_dir=self.run_dir / "knowledge",
            )
        verify_result = self._invoke_role(
            self.planner,
            "review_verify",
            verify_context,
            raise_on_failure=False,
            span_id=span_id,
        )
        state.role_outputs[f"{self.planner}:review_verify:{round_index}"] = verify_result.output
        self.artifacts.write_role_summary(
            role=self.planner,
            phase=f"review_verify_round_{round_index}",
            result=verify_result,
        )

        verify_decision = extract_review_decision(
            planner_output=verify_result.output,
            available_commits_by_role={coder: [] for coder in self.coders},
            coder_roles=set(self.coders),
            resolve_symbolic_commit=self._resolve_symbolic_commit_token,
        )
        structured_valid = bool(verify_result.output.get("openrouter_structured_valid", True))
        if not verify_result.ok:
            self.artifacts.append_open_question(
                f"planner verify round {round_index} failed; continuing without forced rework."
            )
            return verify_decision
        if not structured_valid:
            self.artifacts.append_open_question(
                f"planner verify round {round_index} produced invalid structured output; "
                "continuing without forced rework."
            )
            return verify_decision
        verification_stats = verification_command_stats(verify_result.output)
        if verification_stats["attempted"] > 0 and verification_stats["failed"] > 0:
            self.artifacts.append_open_question(
                f"planner verify round {round_index} reported failing verify commands "
                f"(failed={verification_stats['failed']}, succeeded={verification_stats['succeeded']}); "
                "verify evidence is weak."
            )
            if not verify_decision.request_rework:
                self.artifacts.append_open_question(
                    f"planner verify round {round_index} kept request_rework=false despite verify command failures."
                )
        elif verification_stats["attempted"] == 0 and is_noop_public_validation(
            last_validation.stdout
        ):
            self.artifacts.append_open_question(
                f"planner verify round {round_index} executed no verify commands while public validation is no-op; "
                "integrated-check evidence is weak."
            )
        if not verify_decision.dynamic_checks_ran:
            self.artifacts.append_open_question(
                f"planner verify round {round_index} ran no dynamic checks on integrated candidate."
            )
        if verify_decision.request_rework:
            self.artifacts.append_status(
                f"review verify round {round_index} requested coder rework"
            )
        else:
            self.artifacts.append_status(
                f"review verify round {round_index} accepted integrated candidate"
            )
        return verify_decision

    def _run_public_validation_round(
        self,
        *,
        round_index: int,
        task: TaskPack,
        state: HarnessState,
        tools,
        public_policy: str,
    ) -> PublicValidationSummary:
        if public_policy == "off":
            self.artifacts.append_status(
                f"review round {round_index} skipped public validation (policy=off)"
            )
            return PublicValidationSummary(
                passed=False,
                available=False,
                noop=False,
                last_validation=public_validation_state(
                    policy=public_policy,
                    state=PublicValidationState.SKIPPED_POLICY_OFF,
                    stdout="public validation skipped: policy=off\n",
                ),
            )

        if not task.substrate.public_validate_cmd:
            self.artifacts.append_status(
                f"review round {round_index} public validation unavailable (no command configured)"
            )
            return PublicValidationSummary(
                passed=False,
                available=False,
                noop=False,
                last_validation=public_validation_state(
                    policy=public_policy,
                    state=PublicValidationState.UNAVAILABLE_NO_COMMAND,
                    stdout="public validation unavailable: no command configured\n",
                ),
            )

        validation_result = tools.call(
            ToolCall(
                ts_ms=now_ms(),
                role=self.planner,
                tool="env.public_validate",
                args={},
            )
        )
        state.public_validation_attempts += 1
        self.artifacts.write_public_validate(
            round_index,
            validation_result.stdout,
            validation_result.stderr,
        )

        summary = evaluate_public_validation(
            policy=public_policy,
            ok=bool(validation_result.ok),
            stdout=validation_result.stdout,
            stderr=validation_result.stderr,
        )
        if summary.noop:
            state.public_validation_noop_rounds += 1
            self.artifacts.append_open_question(
                f"public validation round {round_index} appears no-op; hidden-only checks may dominate."
            )

        if summary.passed:
            self.artifacts.append_status(f"review round {round_index} passed public validation")
        elif summary.available:
            self.artifacts.append_status(f"review round {round_index} failed public validation")
        else:
            self.artifacts.append_status(
                f"review round {round_index} public validation unavailable/no-op; relying on reviewer verification"
            )

        if summary.available:
            self.team.post_public_validation(
                round_index=round_index,
                ok=summary.passed,
                stdout=validation_result.stdout,
                stderr=validation_result.stderr,
            )
        return summary

    def _finalize_phase(self, task: TaskPack, state: HarnessState) -> Path:
        self.event_logger.begin_span("finalize", name="phase.finalize", tags=["phase", "finalize"])
        context = {
            "task_id": task.task_id,
            "phase": "finalize",
            "metrics_so_far": {
                "review_round": state.review_round,
                "merge_conflicts": state.merge_conflicts,
                "public_validation_attempts": state.public_validation_attempts,
                "coordination_summary": self.team.summary(),
            },
            "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
        }
        result = self._invoke_role(self.planner, "finalize", context, raise_on_failure=False, span_id="finalize")
        state.role_outputs[f"{self.planner}:finalize"] = result.output
        self.artifacts.write_role_summary(role=self.planner, phase="finalize", result=result)
        if not result.ok:
            self.artifacts.append_open_question(
                "planner finalize failed; generating final patch from current planner branch state"
            )

        planner_path = self.role_paths[self.planner]
        final_patch = self.run_dir / "final.patch"
        diff_result = run_command(
            [
                "git",
                "-C",
                str(planner_path),
                "diff",
                "--binary",
                f"{self.base_commit}",
                "HEAD",
            ]
        )
        ensure_success(diff_result, "git diff final patch")
        final_patch.write_text(diff_result.stdout, encoding="utf-8")
        self.event_logger.end_span("finalize")
        return final_patch

    def _request_rework_from_coders(
        self,
        task: TaskPack,
        round_index: int,
        validation_stdout: str,
        validation_stderr: str,
        state: HarnessState,
        planner_feedback_by_role: Optional[Dict[str, str]] = None,
        public_policy: str = "off",
        parent_span_id: str | None = None,
    ) -> None:
        rework_seed = self.team.seed_rework(
            round_index=round_index,
            validation_stdout=validation_stdout,
            validation_stderr=validation_stderr,
            planner_feedback_by_role=planner_feedback_by_role,
        )

        full_plan = self.artifacts.read_text(self.run_dir / "plans" / "plan.md")
        # On rework, the feedback is the primary task. The plan is reference
        # context the coder has already seen — shrink it to keep feedback prominent.
        if round_index >= 2 and full_plan:
            planner_summary = full_plan[:500]
        elif full_plan:
            planner_summary = full_plan[:2000]
        else:
            planner_summary = full_plan
        phase_results = self._run_parallel_claim_phase(
            task=task,
            state=state,
            coordination_phase=rework_seed.phase,
            driver_phase="rework",
            planner_summary=planner_summary,
            round_index=round_index,
            extra_context=self._rework_extra_context(
                validation_stdout=validation_stdout,
                validation_stderr=validation_stderr,
                planner_feedback_by_role=planner_feedback_by_role,
                state=state,
                public_policy=public_policy,
            ),
            parent_span_id=parent_span_id,
        )
        self._record_worker_phase_outcomes(
            label=f"rework round {round_index}",
            phase_results=phase_results,
        )

        self.artifacts.append_status(
            f"rework tasks seeded for round {round_index}: {rework_seed.seeded_count}"
        )

    def _rework_extra_context(
        self,
        *,
        validation_stdout: str,
        validation_stderr: str,
        planner_feedback_by_role: Optional[Dict[str, str]],
        state: Optional[HarnessState] = None,
        public_policy: str = "off",
    ) -> Dict[str, Any]:
        raw_feedback = planner_feedback_by_role or {}
        ctx: Dict[str, Any] = {
            "planner_feedback_by_role": {
                role: feedback[:2000] for role, feedback in raw_feedback.items()
                if isinstance(feedback, str)
            },
        }
        # Only include validation output when it's actionable — i.e. validation
        # actually ran and produced meaningful output.  Raw build logs are capped
        # to the tail where errors typically appear.
        has_meaningful_output = (
            public_policy != "off"
            and (validation_stdout.strip() or validation_stderr.strip())
        )
        if has_meaningful_output:
            ctx["public_validate_stderr"] = validation_stderr[-4000:]
            # stdout is lower signal (progress/info); include less
            if validation_stdout.strip():
                ctx["public_validate_stdout"] = validation_stdout[-2000:]
        # Extract each coder's prior failed commands so rework coders see
        # their own build/test errors even when public_validate_policy=off.
        if state is not None:
            prior_failures_by_role: Dict[str, List[Dict[str, Any]]] = {}
            latest = self._latest_coder_outputs(state)
            for coder, outputs in latest.items():
                coder_failures: List[Dict[str, Any]] = []
                for out_entry in outputs:
                    for fail in out_entry.get("failed_commands", []):
                        coder_failures.append(fail)
                if coder_failures:
                    prior_failures_by_role[coder] = coder_failures[-5:]
            if prior_failures_by_role:
                ctx["prior_command_failures_by_role"] = prior_failures_by_role
        # Cross-coder visibility: each coder sees peer summaries
        peer_summaries: Dict[str, Dict[str, Any]] = {}
        if state is not None:
            latest = self._latest_coder_outputs(state)
            for coder, outputs in latest.items():
                if not outputs:
                    continue
                last = outputs[-1]
                peer_summaries[coder] = {
                    "summary": last.get("summary", ""),
                    "files_changed": last.get("files_changed", []),
                    "commit": last.get("commit"),
                }
        if peer_summaries:
            ctx["peer_progress"] = peer_summaries
        if self.knowledge is not None:
            directive = self.knowledge.directive()
            if directive:
                ctx["reflection_directive"] = directive
            # NOTE: knowledge_tool is NOT injected here because extra_context
            # is shared across all coders.  It is injected per-role inside
            # _run_claimed_task so that the script is deployed into each
            # coder's own worktree.
        return ctx

    def _run_reflection(
        self,
        *,
        task: TaskPack,
        round_index: int,
        state: HarnessState,
        last_validation: "PublicValidationRecord",
        review_decision: "ReviewDecision",
        merged_commits_this_round: Dict[str, List[str]],
        span_id: str | None = None,
    ) -> None:
        """Invoke the planner driver with phase='reflect' to distill knowledge."""
        assert self.knowledge is not None

        coder_output_summaries: Dict[str, Any] = {}
        for coder in self.coders:
            outputs = self._latest_coder_outputs(state).get(coder, [])
            coder_output_summaries[coder] = outputs[-2:]

        reflection_context: Dict[str, Any] = {
            "task_id": task.task_id,
            "phase": "reflect",
            "round": round_index,
            "runtime_suffix": f"round_{round_index}_reflect",
            "review_ledger": self.ledger.as_context(),
            "validation_result": {
                "stdout": (last_validation.stdout or "")[-2000:],
                "stderr": (last_validation.stderr or "")[-2000:],
                "ok": last_validation.ok,
                "state": last_validation.state.value,
            },
            "review_decision_summary": {
                "request_rework": review_decision.request_rework,
                "merge_commits_by_role": {
                    r: len(c) for r, c in review_decision.merge_commits_by_role.items()
                },
                "coder_feedback": review_decision.coder_feedback,
            },
            "coder_output_summaries": coder_output_summaries,
            "merged_commits_this_round": merged_commits_this_round,
            "current_knowledge": {
                "directive": self.knowledge.directive(),
                "task_understanding": self.knowledge.surface("task_understanding"),
                "failure_patterns": self.knowledge.surface("failure_patterns"),
                "workflow_insights": self.knowledge.surface("workflow_insights"),
            },
            "coordination_summary": self.team.summary(),
        }

        result = self._invoke_role(
            self.planner,
            "reflect",
            reflection_context,
            raise_on_failure=False,
            span_id=span_id,
        )
        state.role_outputs[f"{self.planner}:reflect:{round_index}"] = result.output
        self.artifacts.write_role_summary(
            role=self.planner,
            phase=f"reflect_round_{round_index}",
            result=result,
        )

        if result.ok and isinstance(result.output, dict):
            self.knowledge.update_from_reflection(round_index, result.output)
            self.event_logger.log(
                "reflection_update",
                {
                    "round_index": round_index,
                    "directive_chars": len(self.knowledge.directive()),
                    "superseded": result.output.get("superseded", []),
                },
            )
        else:
            self.artifacts.append_open_question(
                f"reflection round {round_index} failed; knowledge surfaces unchanged"
            )

    def _run_parallel_claim_phase(
        self,
        *,
        task: TaskPack,
        state: HarnessState,
        coordination_phase: str,
        driver_phase: str,
        planner_summary: str,
        round_index: Optional[int],
        extra_context: Dict[str, Any],
        parent_span_id: str | None = None,
    ) -> Dict[str, WorkerPhaseOutcome]:
        if not self.coders:
            return {}

        results: Dict[str, WorkerPhaseOutcome] = {}
        with ThreadPoolExecutor(max_workers=len(self.coders)) as pool:
            future_to_role = {
                pool.submit(
                    self._coder_claim_loop,
                    task=task,
                    role=coder,
                    coordination_phase=coordination_phase,
                    driver_phase=driver_phase,
                    planner_summary=planner_summary,
                    round_index=round_index,
                    extra_context=extra_context,
                    parent_span_id=parent_span_id,
                ): coder
                for coder in self.coders
            }

            for future in as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"worker loop crashed for {role} during {driver_phase}"
                    ) from exc
                results[role] = result
                state.role_outputs.update(result.role_outputs)

        return results

    def _coder_claim_loop(
        self,
        *,
        task: TaskPack,
        role: str,
        coordination_phase: str,
        driver_phase: str,
        planner_summary: str,
        round_index: Optional[int],
        extra_context: Dict[str, Any],
        parent_span_id: str | None = None,
    ) -> WorkerPhaseOutcome:
        outcome = WorkerPhaseOutcome(role=role)

        while True:
            claimed_task = self.team.claim_next_task(phase=coordination_phase, role=role)
            if claimed_task is None:
                break

            outcome.claimed += 1
            self.team.mark_claimed(
                phase=coordination_phase,
                role=role,
                task=claimed_task,
                round_index=round_index,
            )

            try:
                result = self._run_claimed_task(
                    task=task,
                    role=role,
                    claimed_task=claimed_task,
                    coordination_phase=coordination_phase,
                    driver_phase=driver_phase,
                    planner_summary=planner_summary,
                    round_index=round_index,
                    extra_context=extra_context,
                    parent_span_id=parent_span_id,
                )
            except Exception as exc:  # noqa: BLE001
                outcome.failed += 1
                self.team.mark_failed(
                    phase=coordination_phase,
                    role=role,
                    task=claimed_task,
                    round_index=round_index,
                    error={
                        "error": f"task execution crashed for {role} in {driver_phase}: {exc}",
                        "driver_phase": driver_phase,
                        "coordination_phase": coordination_phase,
                    },
                )
                raise

            self.artifacts.write_role_summary(
                role=role,
                phase=driver_phase,
                result=result,
                suffix=claimed_task.task_id,
            )

            if not result.ok:
                outcome.failed += 1
                self.team.mark_failed(
                    phase=coordination_phase,
                    role=role,
                    task=claimed_task,
                    round_index=round_index,
                    error={
                        "error": self._role_failure_message(role=role, phase=driver_phase, result=result),
                        "driver_phase": driver_phase,
                        "coordination_phase": coordination_phase,
                    },
                )
                continue

            protocol_issue = self._worker_protocol_issue(result.output)
            if protocol_issue:
                outcome.warnings.append(
                    f"worker protocol warning for {role} in {coordination_phase}:{claimed_task.task_id}: "
                    f"{protocol_issue}"
                )
                result.output["protocol_warning"] = protocol_issue

            outcome.completed += 1
            output_key = f"{role}:{driver_phase}:{claimed_task.task_id}"
            outcome.role_outputs[output_key] = result.output

            mark_ok = self.team.mark_completed(
                phase=coordination_phase,
                role=role,
                task=claimed_task,
                round_index=round_index,
                result={
                    "status": result.output.get("status", "completed"),
                    "summary": result.output.get("summary", ""),
                },
            )
            if not mark_ok:
                outcome.warnings.append(
                    f"unable to mark completed task in coordination DB: "
                    f"phase={coordination_phase} role={role} task={claimed_task.task_id}"
                )

        return outcome

    def _run_claimed_task(
        self,
        *,
        task: TaskPack,
        role: str,
        claimed_task: ClaimedTask,
        coordination_phase: str,
        driver_phase: str,
        planner_summary: str,
        round_index: Optional[int],
        extra_context: Dict[str, Any],
        parent_span_id: str | None = None,
    ) -> RoleRunResult:
        context = {
            "task_id": task.task_id,
            "phase": driver_phase,
            "coordination_phase": coordination_phase,
            "round": round_index,
            "runtime_suffix": claimed_task.task_id,
            "role": role,
            "assignment": [claimed_task.payload],
            "claimed_task": claimed_task.payload,
            "claimed_task_id": claimed_task.task_id,
            "planner_summary": planner_summary,
            "inbox": self.team.inbox(
                role=role,
                phase=coordination_phase,
                limit=30,
            ),
            "coordination": {
                "protocol": "sqlite",
                "db_path": str(self.team.db_path),
                "phase": coordination_phase,
                "task_id": claimed_task.task_id,
            },
        }
        context.update(extra_context)
        # On rework, inject the coder's own diff so it doesn't waste turns
        # rediscovering its current state.
        if driver_phase == "rework" and role in self.role_paths:
            diff_result = run_command(
                ["git", "-C", str(self.role_paths[role]), "diff", f"{self.base_commit}..HEAD"],
                timeout_sec=30,
            )
            if diff_result.ok and diff_result.stdout.strip():
                context["own_diff"] = diff_result.stdout[:6000]
        # Carry forward prior conversation so rework doesn't re-explore from scratch
        if driver_phase == "rework":
            prior_msgs = self._find_prior_conversation(role, round_index)
            if prior_msgs is not None:
                context["prior_conversation_messages"] = prior_msgs
        # Deploy knowledge_tool into the executing role's worktree so the
        # advertised command path resolves correctly for each coder.
        if self.knowledge is not None:
            context["knowledge_tool"] = build_knowledge_tool_context(
                role_path=self.role_paths[role],
                knowledge_dir=self.run_dir / "knowledge",
            )
        return self._invoke_role(role, driver_phase, context, raise_on_failure=False, span_id=parent_span_id)

    def _find_prior_conversation(self, role: str, round_index: int) -> list[dict] | None:
        """Find conversation messages from the prior phase for this role.

        For rework round 1, look for the last implementation conversation.
        For rework round N>1, look for the most recent rework conversation (= round N-1).
        Returns the raw messages list or None.
        """
        archive = self.run_dir / "role_runtime"
        if not archive.is_dir():
            return None

        if round_index == 1:
            pattern = f"{role}_implementation_*_conversation.json"
        else:
            pattern = f"{role}_rework_*_conversation.json"

        candidates = sorted(archive.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None

        prior_path = candidates[-1]  # most recent
        try:
            conv = json.loads(prior_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        # Prefer raw_messages (complete, ready to use) over reconstructed
        raw = conv.get("raw_messages")
        if raw and isinstance(raw, list) and len(raw) > 2:
            messages = raw
        else:
            # Fallback: reconstruct from conversation structure
            messages = []
            sys_msg = conv.get("system_message")
            usr_msg = conv.get("user_message")
            if sys_msg:
                messages.append(sys_msg)
            if usr_msg:
                messages.append(usr_msg)

            for turn in conv.get("turns", []):
                assistant_msg = turn.get("assistant_message")
                if assistant_msg:
                    messages.append(assistant_msg)
                for tr in turn.get("tool_results", []):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_call_id", ""),
                        "content": tr.get("content", ""),
                    })

        return messages if len(messages) > 2 else None

    def _scratch_merge_test(
        self,
        state: HarnessState,
        *,
        nominated_commits_by_role: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Test-merge on a temporary branch. Returns per-commit success/failure + conflict files."""
        planner_path = self.role_paths[self.planner]
        head_result = run_command(
            ["git", "-C", str(planner_path), "rev-parse", "HEAD"],
            timeout_sec=15,
        )
        ensure_success(head_result, "git rev-parse HEAD for scratch merge")
        original_sha = head_result.stdout.strip()
        original_branch = ""
        branch_result = run_command(
            ["git", "-C", str(planner_path), "symbolic-ref", "--quiet", "--short", "HEAD"],
            timeout_sec=15,
        )
        if branch_result.ok:
            original_branch = branch_result.stdout.strip()
        round_idx = state.review_round
        scratch_branch = f"_loopbench_scratch_{round_idx}"

        commits_tested: Dict[str, Dict[str, Any]] = {}
        all_mergeable = True

        create_result = run_command(
            ["git", "-C", str(planner_path), "checkout", "-b", scratch_branch],
            timeout_sec=30,
        )
        if not create_result.ok:
            return {"commits_tested": {}, "all_mergeable": False, "error": "failed to create scratch branch"}

        try:
            for coder, shas in nominated_commits_by_role.items():
                for sha in shas:
                    if sha in state.merged_commits.get(coder, set()):
                        continue
                    cherry = run_command(
                        ["git", "-C", str(planner_path), "cherry-pick", sha],
                        timeout_sec=120,
                    )
                    if cherry.ok:
                        commits_tested[sha] = {"ok": True, "coder": coder, "conflict_files": []}
                    else:
                        all_mergeable = False
                        # Capture conflict files
                        diff_result = run_command(
                            ["git", "-C", str(planner_path), "diff", "--name-only", "--diff-filter=U"],
                            timeout_sec=15,
                        )
                        conflict_files = [
                            f.strip() for f in diff_result.stdout.splitlines() if f.strip()
                        ] if diff_result.ok else []
                        commits_tested[sha] = {
                            "ok": False,
                            "coder": coder,
                            "conflict_files": conflict_files,
                        }
                        run_command(
                            ["git", "-C", str(planner_path), "cherry-pick", "--abort"],
                            timeout_sec=30,
                        )
        finally:
            # Always restore: checkout original branch and delete scratch
            restore_target = original_branch or original_sha
            run_command(
                ["git", "-C", str(planner_path), "checkout", restore_target],
                timeout_sec=30,
            )
            run_command(
                ["git", "-C", str(planner_path), "branch", "-D", scratch_branch],
                timeout_sec=15,
            )
            # Paranoia: verify HEAD matches original
            verify = run_command(
                ["git", "-C", str(planner_path), "rev-parse", "HEAD"],
                timeout_sec=15,
            )
            if verify.ok and verify.stdout.strip() != original_sha:
                run_command(
                    ["git", "-C", str(planner_path), "reset", "--hard", original_sha],
                    timeout_sec=30,
                )

        return {"commits_tested": commits_tested, "all_mergeable": all_mergeable}

    def _merge_new_commits(
        self,
        state: HarnessState,
        *,
        nominated_commits_by_role: Optional[Dict[str, List[str]]] = None,
    ) -> bool:
        planner_path = self.role_paths[self.planner]
        ok = True

        for coder in self.coders:
            commits = self._list_commits(coder, self.base_commit)
            unseen = [c for c in commits if c not in state.merged_commits[coder]]
            if nominated_commits_by_role is not None:
                nominated = set(nominated_commits_by_role.get(coder, []))
                unseen = [c for c in unseen if c in nominated]
            for commit in unseen:
                cherry = run_command(["git", "-C", str(planner_path), "cherry-pick", commit], timeout_sec=180)
                if not cherry.ok:
                    ok = False
                    run_command(["git", "-C", str(planner_path), "cherry-pick", "--abort"], timeout_sec=60)
                    stderr_tail = (cherry.stderr or cherry.stdout or "")[-2000:]
                    self.event_logger.log(
                        "merge_commit_failed",
                        {
                            "from_role": coder,
                            "to_role": self.planner,
                            "commit": commit,
                            "stderr_tail": stderr_tail,
                            "review_round": state.review_round,
                        },
                    )
                    self.artifacts.append_open_question(
                        f"failed to cherry-pick {commit} from {coder}: {cherry.stderr.strip() or cherry.stdout.strip()}"
                    )
                    break
                state.merged_commits[coder].add(commit)
                self.event_logger.log(
                    "merge_commit",
                    {
                        "from_role": coder,
                        "to_role": self.planner,
                        "commit": commit,
                        "review_round": state.review_round,
                    },
                )
                self.artifacts.append_status(f"merged commit {commit[:12]} from {coder}")

        return ok

    def _inject_time_context(self, context: Dict[str, Any]) -> None:
        context["run_start_epoch"] = self._run_start_epoch
        context["run_deadline_epoch"] = self._run_start_epoch + self._budget.wall_clock_sec

    def _invoke_role(
        self,
        role: str,
        phase: str,
        context: Dict[str, Any],
        *,
        raise_on_failure: bool = True,
        span_id: str | None = None,
    ) -> RoleRunResult:
        self._inject_time_context(context)
        driver = self.role_drivers[role]
        result = driver.run_phase(
            phase=phase,
            role=role,
            worktree_path=self.role_paths[role],
            run_dir=self.run_dir,
            context=context,
        )
        if result.ok:
            exec_mode = (result.output or {}).get("execution_mode", "")
            if exec_mode == "agentic_tool_loop":
                # Agentic driver already applied changes directly to worktree.
                # Auto-commit any uncommitted changes for coder roles.
                if True:  # auto-commit all roles including planner
                    commit_msg = (result.output or {}).get(
                        "commit_message", f"{role}: {phase} update"
                    )
                    commit_sha = self._commit_role_if_dirty(
                        role=role, commit_message=commit_msg
                    )
                    if commit_sha:
                        result.output["commit"] = commit_sha
                        result.output["changed"] = True
                # Emit driver_tool events from the conversation artifact so the
                # canonical event stream reflects the agent's actual action space.
                self._emit_driver_tool_events(role=role, phase=phase, span_id=span_id)
            else:
                result.output = self._materialize_role_actions(
                    role=role,
                    phase=phase,
                    context=context,
                    output=result.output,
                )

        self.event_logger.log(
            "role_phase",
            {
                "role": role,
                "phase": phase,
                "ok": result.ok,
                "exit_code": result.exit_code,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
                "output": result.output,
            },
            span_id=span_id,
        )

        if not result.ok and raise_on_failure:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            raise RuntimeError(
                f"role '{role}' failed during '{phase}' (exit_code={result.exit_code}). "
                f"stderr_tail={stderr_tail!r}"
            )

        return result

    def _emit_driver_tool_events(self, *, role: str, phase: str, span_id: str | None) -> None:
        """Read conversation.json for this role/phase and emit driver_tool events.

        This bridges the gap between the driver's local tool loop and the
        harness event stream, making driver tool calls first-class in
        events.jsonl, hodoscope exports, and aggregate telemetry.
        """
        rt_dir = self.run_dir / "role_runtime"
        if not rt_dir.exists():
            return

        # Find conversation files matching this role and phase
        pattern = f"{role}_{phase}_*_conversation.json"
        conv_files = sorted(rt_dir.glob(pattern))
        if not conv_files:
            # Try without wildcard suffix for simpler naming conventions
            pattern = f"{role}_{phase}_conversation.json"
            conv_files = sorted(rt_dir.glob(pattern))

        for conv_path in conv_files:
            try:
                conv = json.loads(conv_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            turns = conv.get("turns") or []
            for turn in turns:
                turn_idx = turn.get("turn", 0)
                assistant = turn.get("assistant_message") or {}
                tool_calls = assistant.get("tool_calls") or []
                tool_results = turn.get("tool_results") or []

                # Build result lookup by tool_call_id
                result_by_id = {}
                for tr in tool_results:
                    tcid = tr.get("tool_call_id")
                    if tcid:
                        result_by_id[tcid] = tr

                for tc in tool_calls:
                    func = tc.get("function") or {}
                    tool_name = func.get("name", "unknown")
                    tc_id = tc.get("id", "")

                    # Truncate args for the event (avoid bloating events.jsonl)
                    args_raw = func.get("arguments", "{}")
                    if isinstance(args_raw, str) and len(args_raw) > 500:
                        args_summary = args_raw[:500] + "..."
                    else:
                        args_summary = args_raw

                    # Get result summary
                    tr = result_by_id.get(tc_id)
                    result_content = (tr.get("content", "") if tr else "")
                    if len(result_content) > 500:
                        result_summary = result_content[:500] + "..."
                    else:
                        result_summary = result_content

                    self.event_logger.log(
                        "driver_tool",
                        {
                            "role": role,
                            "phase": phase,
                            "turn": turn_idx,
                            "tool": tool_name,
                            "args_summary": args_summary,
                            "result_summary": result_summary,
                        },
                        span_id=span_id,
                    )

    def _role_failure_message(self, *, role: str, phase: str, result: RoleRunResult) -> str:
        output_error = result.output.get("error") if isinstance(result.output, dict) else None
        if isinstance(output_error, str) and output_error.strip():
            return output_error.strip()
        stderr_tail = result.stderr[-2000:] if result.stderr else ""
        return (
            f"role '{role}' failed during '{phase}' (exit_code={result.exit_code}). "
            f"stderr_tail={stderr_tail!r}"
        )

    def _materialize_role_actions(
        self,
        *,
        role: str,
        phase: str,
        context: Dict[str, Any],
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(output, dict):
            return {}
        if phase == "bootstrap":
            return output
        has_action_intents = any(
            key in output
            for key in (
                "intent_file_updates",
                "intent_patch",
                "intent_run_commands",
                "file_updates",
                "patch",
                "run_commands",
            )
        )
        if not has_action_intents:
            return output
        if self._tools is None:
            raise RuntimeError("tool router is not initialized")

        plan = extract_role_action_plan(role=role, phase=phase, output=output)
        requested_file_updates = list(plan.file_updates)
        requested_patch = plan.patch_text
        requested_commands = list(plan.run_commands)
        intent_file_update_preview = [
            {"path": update.path, "content_len": len(update.content)}
            for update in requested_file_updates
        ]
        suppress_commands = plan.planner_non_mutating and phase not in {"review", "review_verify"}
        assignment_paths = list(iter_assignment_paths(context))
        # Planner has no assignment during review — detect unguarded
        # mutations so they appear in the audit trail.
        planner_mutation_audit = role == self.planner and not assignment_paths

        patch_text = "" if plan.planner_non_mutating else requested_patch
        patch_requested_paths = paths_from_unified_patch(requested_patch) if requested_patch else []
        patch_applied_paths: List[str] = []
        patch_rejected_paths: List[str] = []
        patch_apply_error = ""
        if patch_text and assignment_paths:
            patch_rejected_paths = [
                path
                for path in patch_requested_paths
                if not path_allowed_by_assignment(path=path, allowed_paths=assignment_paths)
            ]
            if patch_rejected_paths:
                preview = ", ".join(patch_rejected_paths[:8])
                if len(patch_rejected_paths) > 8:
                    preview = f"{preview}, ..."
                patch_apply_error = f"patch paths outside assignment: {preview}"
                patch_text = ""
        if patch_text and not patch_text.endswith("\n"):
            patch_text = f"{patch_text}\n"
        if patch_text:
            patch_result = self._tool_call(
                role=role,
                tool="repo.apply_patch",
                args={"patch_text": patch_text},
            )
            if patch_result.ok:
                patch_applied_paths = paths_from_unified_patch(patch_text)
            else:
                patch_apply_error = (
                    patch_result.stderr.strip()
                    or patch_result.stdout.strip()
                    or "failed to apply unified patch"
                )

        file_updates = [] if plan.planner_non_mutating else requested_file_updates
        applied_paths: List[str] = []
        rejected_paths: List[str] = []
        for path in patch_rejected_paths:
            if path not in rejected_paths:
                rejected_paths.append(path)
        for update in file_updates:
            if assignment_paths and not path_allowed_by_assignment(path=update.path, allowed_paths=assignment_paths):
                rejected_paths.append(update.path)
                continue
            write_result = self._tool_call(
                role=role,
                tool="repo.write_file",
                args={"path": update.path, "content": update.content},
            )
            if write_result.ok:
                applied_paths.append(update.path)
            else:
                rejected_paths.append(update.path)

        merged_applied_paths = list(applied_paths)
        for path in patch_applied_paths:
            if path not in merged_applied_paths:
                merged_applied_paths.append(path)

        assignment_deviation_paths = assignment_deviations(
            changed_paths=merged_applied_paths,
            assignment_paths=assignment_paths,
        )
        if planner_mutation_audit and merged_applied_paths:
            assignment_deviation_paths = list(merged_applied_paths)

        commands_to_run = [] if suppress_commands else requested_commands
        command_results: List[Dict[str, Any]] = []
        command_trace: List[Dict[str, Any]] = []
        for cmd in commands_to_run:
            command_result = self._tool_call(
                role=role,
                tool="repo.exec",
                args={"cmd": cmd, "timeout_sec": plan.command_policy_timeout_sec},
            )
            command_trace.append(
                {
                    "cmd": cmd,
                    "ok": bool(command_result.ok),
                    "exit_code": command_result.exit_code,
                    "stdout": command_result.stdout,
                    "stderr": command_result.stderr,
                }
            )
            command_results.append(
                {
                    "cmd": cmd,
                    "ok": bool(command_result.ok),
                    "exit_code": command_result.exit_code,
                    "stdout_tail": command_result.stdout[-1200:],
                    "stderr_tail": command_result.stderr[-1200:],
                }
            )

        commit_sha = None
        should_try_commit = bool(patch_text or file_updates or commands_to_run)
        if not plan.planner_non_mutating and should_try_commit:
            commit_sha = self._commit_role_if_dirty(
                role=role,
                commit_message=plan.commit_message,
            )

        notes = plan.notes
        if plan.planner_non_mutating and (requested_file_updates or suppress_commands):
            ignored_commands = len(requested_commands) if suppress_commands else 0
            notes = (
                f"{notes} planner_non_mutating_mode: ignored "
                f"{len(requested_file_updates)} file_updates and {ignored_commands} run_commands."
            ).strip()
        if plan.planner_non_mutating and requested_patch:
            notes = f"{notes} planner_non_mutating_mode: ignored patch output.".strip()
        if patch_apply_error:
            notes = f"{notes} patch_apply_error: {patch_apply_error}".strip()
        if rejected_paths:
            rejected_note = (
                "file_update_rejections: "
                + ", ".join(rejected_paths[:8])
                + (" ..." if len(rejected_paths) > 8 else "")
            )
            notes = f"{notes} {rejected_note}".strip()
        if assignment_deviation_paths:
            assignment_note = (
                "assignment_deviation_paths: "
                + ", ".join(assignment_deviation_paths[:8])
                + (" ..." if len(assignment_deviation_paths) > 8 else "")
            )
            notes = f"{notes} {assignment_note}".strip()

        for key in ("intent_patch", "patch", "file_updates"):
            output.pop(key, None)
        output.update(
            {
                "summary": plan.summary,
                "notes": notes,
                "intent_file_updates": intent_file_update_preview,
                "intent_patch_chars": len(requested_patch),
                "intent_run_commands": requested_commands,
                "intent_commit_message": plan.commit_message,
                "file_updates_attempted": len(requested_file_updates),
                "file_updates_applied": len(merged_applied_paths),
                "file_updates_rejected": len(rejected_paths),
                "patch_attempted": bool(requested_patch),
                "patch_applied": bool(patch_applied_paths),
                "patch_applied_paths": patch_applied_paths,
                "assignment_deviation_paths": assignment_deviation_paths,
                "rejected_paths": rejected_paths,
                "applied_paths": merged_applied_paths,
                "run_commands_attempted": len(requested_commands),
                "commands_run": command_results,
                "command_policy_max_commands": plan.command_policy_max_commands,
                "command_policy_timeout_sec": plan.command_policy_timeout_sec,
                "planner_non_mutating": plan.planner_non_mutating,
                "commit": commit_sha,
                "changed": bool(commit_sha),
                "execution_mode": "harness_tool_router",
            }
        )

        if command_trace:
            trace_path = self._write_command_trace(
                role=role,
                phase=phase,
                context=context,
                command_trace=command_trace,
            )
            output["command_trace_path"] = str(trace_path)
        return output

    def _commit_role_if_dirty(self, *, role: str, commit_message: str) -> str | None:
        result = self._tool_call(
            role=role,
            tool="repo.git_add_commit",
            args={"message": commit_message},
        )
        if not result.ok:
            return None
        return next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)

    def _tool_call(self, *, role: str, tool: str, args: Dict[str, Any]) -> Any:
        if self._tools is None:
            raise RuntimeError("tool router is not initialized")
        return self._tools.call(
            ToolCall(
                ts_ms=now_ms(),
                role=role,
                tool=tool,
                args=args,
            )
        )

    def _write_command_trace(
        self,
        *,
        role: str,
        phase: str,
        context: Dict[str, Any],
        command_trace: List[Dict[str, Any]],
    ) -> Path:
        runtime_suffix_raw = context.get("runtime_suffix")
        runtime_suffix = safe_path_component(runtime_suffix_raw if isinstance(runtime_suffix_raw, str) else None)
        stem = f"{role}_{phase}" if not runtime_suffix else f"{role}_{phase}_{runtime_suffix}"
        trace_path = self.run_dir / "role_runtime" / f"{stem}_command_trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps({"commands": command_trace}, indent=2),
            encoding="utf-8",
        )
        return trace_path

    def _list_commits(self, role: str, since_commit: str) -> List[str]:
        path = self.role_paths[role]
        result = run_command(
            ["git", "-C", str(path), "rev-list", "--reverse", f"{since_commit}..HEAD"],
            timeout_sec=60,
        )
        ensure_success(result, f"git rev-list {role}")
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def _record_worker_phase_outcomes(
        self,
        *,
        label: str,
        phase_results: Dict[str, WorkerPhaseOutcome],
    ) -> None:
        for coder, outcome in phase_results.items():
            self.artifacts.append_status(
                f"{label} finished for {coder}: "
                f"claimed={outcome.claimed} completed={outcome.completed} failed={outcome.failed}"
            )
            for warning in outcome.warnings:
                self.artifacts.append_open_question(warning)

    def _planner_review_messages(self, *, round_index: int) -> List[Dict[str, Any]]:
        # Only include implementation + the immediately preceding round
        # to cap token growth.  Previously this accumulated ALL prior
        # phases, causing O(rounds) context bloat.
        phases = ["implementation"]
        if round_index > 1:
            phases.append(TeamProtocol.rework_phase(round_index - 1))
            phases.append(f"review_round_{round_index - 1}")

        by_message_id: Dict[int, Dict[str, Any]] = {}
        fallback_rows: List[Dict[str, Any]] = []
        for phase in phases:
            rows = self.team.inbox(
                role=self.planner,
                phase=phase,
                limit=40,
            )
            for row in rows:
                message_id = row.get("message_id")
                if isinstance(message_id, int):
                    by_message_id[message_id] = row
                else:
                    fallback_rows.append(row)

        merged = list(by_message_id.values()) + fallback_rows
        merged.sort(
            key=lambda row: (
                int(row.get("ts_ms", 0)),
                int(row.get("message_id", 0)) if isinstance(row.get("message_id"), int) else 0,
            )
        )
        _MAX_MSG_BODY = 500
        _MAX_MSG_BODY_REWORK = 2000
        for row in merged:
            body = row.get("body", "")
            if isinstance(body, dict):
                # Structured body (rework_request etc.) — serialize and use higher limit
                body = json.dumps(body, indent=2)
                limit = _MAX_MSG_BODY_REWORK
            elif isinstance(body, str):
                # Check message type for rework requests
                msg_type = row.get("type", "")
                limit = _MAX_MSG_BODY_REWORK if msg_type == "rework_request" else _MAX_MSG_BODY
            else:
                continue
            if len(body) > limit:
                row["body"] = body[:limit] + "..."
        return merged[-40:]

    def _resolve_symbolic_commit_token(self, role: str, token: str) -> Optional[str]:
        role_path = self.role_paths.get(role)
        if role_path is None:
            return None
        result = run_command(
            ["git", "-C", str(role_path), "rev-parse", "--verify", token],
            timeout_sec=15,
        )
        if not result.ok:
            return None
        return next((line.strip() for line in result.stdout.splitlines() if line.strip()), None)

    def _worker_protocol_issue(self, output: Dict[str, Any]) -> str | None:
        if not isinstance(output, dict):
            return "missing worker output payload"
        if bool(output.get("openrouter_structured_valid", True)):
            return None

        commit = output.get("commit")
        if (isinstance(commit, str) and commit.strip()) or bool(output.get("changed")):
            return None

        for key in ("applied_paths", "commands_run"):
            value = output.get(key)
            if isinstance(value, list) and value:
                return None

        return "invalid structured reply with no applied edits or executed commands"

    def _latest_coder_outputs(self, state: HarnessState) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {coder: [] for coder in self.coders}
        for key, output in state.role_outputs.items():
            if not isinstance(output, dict):
                continue
            coder = str(key).split(":", 1)[0]
            if coder not in out:
                continue
            # Extract failed command results for planner visibility
            failed_commands: List[Dict[str, Any]] = []
            commands_run = output.get("commands_run", [])
            if isinstance(commands_run, list):
                for cmd in commands_run:
                    if isinstance(cmd, dict) and cmd.get("exit_code", 0) != 0:
                        failed_commands.append({
                            "command": str(cmd.get("command", cmd.get("cmd", "")))[:200],
                            "exit_code": cmd.get("exit_code"),
                            "stderr_tail": str(cmd.get("stderr", cmd.get("stderr_tail", "")))[-500:],
                        })
            entry: Dict[str, Any] = {
                    "output_key": key,
                    "phase": str(output.get("phase") or ""),
                    "summary": str(output.get("summary") or "")[:1000],
                    "notes": str(output.get("notes") or "")[:1000],
                    "commit": output.get("commit"),
                    "applied_paths": output.get("applied_paths"),
                    "run_commands_attempted": output.get("run_commands_attempted"),
                    "failed_commands": failed_commands[:5],
            }
            # PR packet fields produced by the DSPy driver — pass through
            # so the reviewer pipeline can consume structured coder metadata.
            for pr_key in ("files_changed", "key_decisions", "tests_run", "known_risks"):
                val = output.get(pr_key)
                if val:
                    entry[pr_key] = val
            out[coder].append(entry)
        def _compress_older(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if len(outputs) <= 2:
                return outputs
            compressed = []
            for entry in outputs[:-2]:
                compressed.append({
                    "output_key": entry.get("output_key"),
                    "phase": entry.get("phase"),
                    "summary": str(entry.get("summary", ""))[:200],
                    "commit": entry.get("commit"),
                    "failed_commands": entry.get("failed_commands", []),
                })
            return compressed + outputs[-2:]

        return {coder: _compress_older(outputs[-4:]) for coder, outputs in out.items()}

    def _should_rebootstrap(self, state: HarnessState) -> bool:
        """Detect stuck loops: 3+ consecutive rework rounds with no new merges."""
        entries = self.ledger._entries  # direct access for read-only check
        if len(entries) < 3:
            return False
        last_3 = entries[-3:]
        # All rework, no commits merged in any of them
        return all(
            e.decision == "rework" and not any(e.commits_merged.values())
            for e in last_3
        )

    def _rebootstrap(self, task: TaskPack, state: HarnessState) -> None:
        """Re-plan: run bootstrap again with failure context from the ledger."""
        self.artifacts.append_status(
            f"re-bootstrap triggered at round {state.review_round}: "
            "3 consecutive rework rounds with no merged commits"
        )
        context: Dict[str, Any] = {
            "task_id": task.task_id,
            "task_kind": task.kind,
            "phase": "bootstrap",
            "role": self.planner,
            "coders": self.coders,
            "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
            "review_ledger": self.ledger.as_context(),
            "is_rebootstrap": True,
        }
        result = self._invoke_role(self.planner, "bootstrap", context, raise_on_failure=False, span_id=None)
        state.role_outputs[f"{self.planner}:rebootstrap:{state.review_round}"] = result.output
        self.artifacts.write_role_summary(
            role=self.planner,
            phase=f"rebootstrap_round_{state.review_round}",
            result=result,
        )

        if not result.ok:
            self.artifacts.append_open_question(
                "re-bootstrap failed; continuing with existing plan"
            )
            return

        # Re-seed subtasks (same logic as _bootstrap_phase)
        plan_path = self.run_dir / "plans" / "plan.md"
        subtasks_path = self.run_dir / "plans" / "subtasks.yaml"

        plan_md = result.output.get("plan_markdown")
        if isinstance(plan_md, str) and plan_md.strip():
            plan_path.write_text(plan_md, encoding="utf-8")

        subtasks = result.output.get("subtasks")
        if isinstance(subtasks, list) and subtasks:
            subtasks = self._normalize_subtask_roles(subtasks)
            subtasks = self._collapse_overlapping_paths(subtasks)
            subtasks_path.write_text(
                yaml.safe_dump({"subtasks": subtasks}, sort_keys=False), encoding="utf-8"
            )
            if self.knowledge is not None:
                self.knowledge.seed_from_bootstrap(
                    plan_md=plan_path.read_text(encoding="utf-8"),
                    subtasks=subtasks,
                )
            self.team.seed_implementation(
                task_id=task.task_id,
                plan_path=plan_path,
                subtasks_path=subtasks_path,
                subtasks=subtasks,
            )

    def _pick_planner(self, roles: Iterable[str]) -> str:
        role_list = list(roles)
        if not role_list:
            raise ValueError("no roles provided")
        return "planner_reviewer" if "planner_reviewer" in role_list else role_list[0]
