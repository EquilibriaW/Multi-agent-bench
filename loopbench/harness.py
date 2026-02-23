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
from .review_diff_context import build_candidate_merge_commits, build_review_diff_tool_context
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
from .run_artifacts import RunArtifacts
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
    ):
        self.run_id = run_id
        self.run_dir = run_dir
        self.role_paths = role_paths
        self.role_drivers = role_drivers
        self.base_commit = base_commit
        self.max_review_rounds = max_review_rounds
        self.event_logger = event_logger
        self._tools: ToolRouter | None = None

        self.planner = self._pick_planner(role_paths)
        self.coders = [r for r in role_paths if r != self.planner]

        self.artifacts = RunArtifacts(self.run_dir)
        self.team = TeamProtocol(
            db=CoordinationDB(self.run_dir / "coordination" / "coordination.db"),
            planner=self.planner,
            coders=self.coders,
        )

    def run(self, task: TaskPack, budget: Budget, tools) -> Dict[str, Any]:
        state = HarnessState(merged_commits={coder: set() for coder in self.coders})
        start_monotonic = time.monotonic()
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
        context = {
            "task_id": task.task_id,
            "task_kind": task.kind,
            "task_entrypoint": str(Path(task.root_dir) / "public" / task.workspace.entrypoint),
            "phase": "bootstrap",
            "role": self.planner,
            "coders": self.coders,
            "coordination_db_path": str(self.team.db_path),
            "coordination": {"protocol": "sqlite", "db_path": str(self.team.db_path)},
        }
        result = self._invoke_role(self.planner, "bootstrap", context, raise_on_failure=False)
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

        subtasks_path.write_text(yaml.safe_dump({"subtasks": subtasks}, sort_keys=False), encoding="utf-8")

        seeded_count = self.team.seed_implementation(
            task_id=task.task_id,
            plan_path=plan_path,
            subtasks_path=subtasks_path,
            subtasks=subtasks,
        )
        self.artifacts.append_status(
            f"bootstrap finished: plan + subtasks written, seeded implementation tasks={seeded_count}"
        )

    def _implementation_phase(self, task: TaskPack, state: HarnessState) -> None:
        planner_summary = self.artifacts.read_text(self.run_dir / "plans" / "plan.md")
        phase_results = self._run_parallel_claim_phase(
            task=task,
            state=state,
            coordination_phase="implementation",
            driver_phase="implementation",
            planner_summary=planner_summary,
            round_index=None,
            extra_context={},
        )
        self._record_worker_phase_outcomes(
            label="implementation phase",
            phase_results=phase_results,
        )

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
            self.artifacts.append_status(f"review round {round_index} started")

            planner_context = {
                "task_id": task.task_id,
                "phase": "review",
                "review_stage": "select",
                "round": round_index,
                "runtime_suffix": f"round_{round_index}",
                "coder_commits": {coder: self._list_commits(coder, self.base_commit) for coder in self.coders},
                "last_public_validation": last_validation.as_dict(),
                "coordination_db_path": str(self.team.db_path),
                "coordination_summary": self.team.summary(),
                # Preserve legacy key name consumed by drivers, but include
                # recent rework/review messages so planner decisions see the
                # latest coder task lifecycle signals.
                "implementation_messages": self._planner_review_messages(round_index=round_index),
                "latest_coder_outputs": self._latest_coder_outputs(state),
            }
            # Only present unmerged commits as candidates — already-merged
            # commits waste context and can't be re-merged anyway.
            unmerged_by_role = {
                coder: [c for c in planner_context["coder_commits"].get(coder, [])
                        if c not in state.merged_commits.get(coder, set())]
                for coder in self.coders
            }
            planner_context["candidate_merge_commits"] = build_candidate_merge_commits(
                role_paths=self.role_paths,
                coders=self.coders,
                coder_commits_by_role=unmerged_by_role,
            )
            planner_context["review_diff_tool"] = build_review_diff_tool_context(
                role_paths=self.role_paths,
                planner=self.planner,
                round_index=round_index,
                candidate_merge_commits=planner_context["candidate_merge_commits"],
            )
            planner_result = self._invoke_role(
                self.planner,
                "review",
                planner_context,
                raise_on_failure=False,
            )
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

            if not force_rework:
                verify_decision = self._run_review_verify(
                    task=task,
                    round_index=round_index,
                    state=state,
                    last_validation=last_validation,
                    candidate_merge_commits=planner_context["candidate_merge_commits"],
                    review_diff_tool=planner_context["review_diff_tool"],
                )
                if verify_decision.request_rework:
                    force_rework = True
                if verify_decision.coder_feedback:
                    merged_feedback = dict(review_decision.coder_feedback)
                    merged_feedback.update(verify_decision.coder_feedback)
                    review_decision.coder_feedback = merged_feedback

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
            }
            if not force_rework:
                self.artifacts.write_review_round_audit(
                    **audit_kwargs,
                    accepted=True,
                    force_rework=False,
                )
                if public_validation_passed:
                    accepted_public_pass = True
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
                self._request_rework_from_coders(**rework_kwargs)
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

            self._request_rework_from_coders(**rework_kwargs)

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
    ) -> ReviewDecision:
        verify_context = {
            "task_id": task.task_id,
            "phase": "review_verify",
            "review_stage": "verify",
            "round": round_index,
            "runtime_suffix": f"round_{round_index}_verify",
            "coordination_db_path": str(self.team.db_path),
            "coordination_summary": self.team.summary(),
            "last_public_validation": last_validation.as_dict(),
            "latest_coder_outputs": self._latest_coder_outputs(state),
            "candidate_merge_commits": candidate_merge_commits,
            "review_diff_tool": review_diff_tool,
        }
        verify_result = self._invoke_role(
            self.planner,
            "review_verify",
            verify_context,
            raise_on_failure=False,
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
        context = {
            "task_id": task.task_id,
            "phase": "finalize",
            "metrics_so_far": {
                "review_round": state.review_round,
                "merge_conflicts": state.merge_conflicts,
                "public_validation_attempts": state.public_validation_attempts,
                "coordination_summary": self.team.summary(),
            },
            "coordination_db_path": str(self.team.db_path),
        }
        result = self._invoke_role(self.planner, "finalize", context, raise_on_failure=False)
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
        return final_patch

    def _request_rework_from_coders(
        self,
        task: TaskPack,
        round_index: int,
        validation_stdout: str,
        validation_stderr: str,
        state: HarnessState,
        planner_feedback_by_role: Optional[Dict[str, str]] = None,
    ) -> None:
        rework_seed = self.team.seed_rework(
            round_index=round_index,
            validation_stdout=validation_stdout,
            validation_stderr=validation_stderr,
            planner_feedback_by_role=planner_feedback_by_role,
        )

        planner_summary = self.artifacts.read_text(self.run_dir / "plans" / "plan.md")
        phase_results = self._run_parallel_claim_phase(
            task=task,
            state=state,
            coordination_phase=rework_seed.phase,
            driver_phase="rework",
            planner_summary=planner_summary,
            round_index=round_index,
            extra_context={
                "public_validate_stdout": validation_stdout[-8000:],
                "public_validate_stderr": validation_stderr[-8000:],
                "planner_feedback_by_role": planner_feedback_by_role or {},
            },
        )
        self._record_worker_phase_outcomes(
            label=f"rework round {round_index}",
            phase_results=phase_results,
        )

        self.artifacts.append_status(
            f"rework tasks seeded for round {round_index}: {rework_seed.seeded_count}"
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
                    "driver_phase": driver_phase,
                    "status": result.output.get("status", "completed"),
                    "output_keys": sorted(result.output.keys()),
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
            "coordination_db_path": str(self.team.db_path),
            "coordination": {
                "protocol": "sqlite",
                "db_path": str(self.team.db_path),
                "phase": coordination_phase,
                "task_id": claimed_task.task_id,
            },
        }
        context.update(extra_context)
        return self._invoke_role(role, driver_phase, context, raise_on_failure=False)

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

    def _invoke_role(
        self,
        role: str,
        phase: str,
        context: Dict[str, Any],
        *,
        raise_on_failure: bool = True,
    ) -> RoleRunResult:
        driver = self.role_drivers[role]
        result = driver.run_phase(
            phase=phase,
            role=role,
            worktree_path=self.role_paths[role],
            run_dir=self.run_dir,
            context=context,
        )
        if result.ok:
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
        )

        if not result.ok and raise_on_failure:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            raise RuntimeError(
                f"role '{role}' failed during '{phase}' (exit_code={result.exit_code}). "
                f"stderr_tail={stderr_tail!r}"
            )

        return result

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
        return merged[-120:]

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
            out[coder].append(
                {
                    "output_key": key,
                    "phase": str(output.get("phase") or ""),
                    "summary": str(output.get("summary") or ""),
                    "notes": str(output.get("notes") or ""),
                    "commit": output.get("commit"),
                    "applied_paths": output.get("applied_paths"),
                    "run_commands_attempted": output.get("run_commands_attempted"),
                }
            )
        return {coder: outputs[-4:] for coder, outputs in out.items()}

    def _pick_planner(self, roles: Iterable[str]) -> str:
        role_list = list(roles)
        if not role_list:
            raise ValueError("no roles provided")
        return "planner_reviewer" if "planner_reviewer" in role_list else role_list[0]
