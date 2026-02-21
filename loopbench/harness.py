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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import yaml

from .agents import RoleDriver, RoleRunResult
from .coordination import ClaimedTask, CoordinationDB
from .events import EventLogger
from .interfaces import MultiAgentHarness
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
    role_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coordination_summary: Dict[str, int] = field(default_factory=dict)


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

        self.artifacts.append_status(f"run {self.run_id} started")
        self._init_coordination()

        self._bootstrap_phase(task, state)
        self._implementation_phase(task, state)

        public_pass = self._review_phase(task, state, tools)

        final_patch_path = self._finalize_phase(task, state)
        state.coordination_summary = self.team.summary()

        elapsed = time.monotonic() - start_monotonic
        metrics = {
            "review_iterations": state.review_round,
            "merge_conflicts": state.merge_conflicts,
            "public_validation_attempts": state.public_validation_attempts,
            "wall_clock_sec": elapsed,
        }
        for key, value in state.coordination_summary.items():
            metrics[f"coordination_{key}"] = value

        self.artifacts.append_status(f"run {self.run_id} completed; public_pass={public_pass}")

        return {
            "final_patch_path": str(final_patch_path),
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
        result = self._invoke_role(self.planner, "bootstrap", context)
        state.role_outputs[f"{self.planner}:bootstrap"] = result.output
        self.artifacts.write_role_summary(
            role=self.planner,
            phase="bootstrap",
            result=result,
        )

        plan_path = self.run_dir / "plans" / "plan.md"
        subtasks_path = self.run_dir / "plans" / "subtasks.yaml"

        plan_md = result.output.get("plan_markdown")
        if isinstance(plan_md, str) and plan_md.strip():
            plan_path.write_text(plan_md, encoding="utf-8")
        else:
            plan_path.write_text(self._default_plan_markdown(task), encoding="utf-8")

        subtasks = result.output.get("subtasks")
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = self._default_subtasks(task)

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
        public_pass = False
        last_validation = {"ok": False, "stdout": "", "stderr": "not run"}

        for round_index in range(1, self.max_review_rounds + 1):
            state.review_round = round_index
            self.artifacts.append_status(f"review round {round_index} started")

            planner_context = {
                "task_id": task.task_id,
                "phase": "review",
                "round": round_index,
                "runtime_suffix": f"round_{round_index}",
                "coder_commits": {coder: self._list_commits(coder, self.base_commit) for coder in self.coders},
                "last_public_validation": last_validation,
                "coordination_db_path": str(self.team.db_path),
                "coordination_summary": self.team.summary(),
                "implementation_messages": self.team.inbox(
                    role=self.planner,
                    phase="implementation",
                    limit=40,
                ),
            }
            planner_result = self._invoke_role(self.planner, "review", planner_context)
            state.role_outputs[f"{self.planner}:review:{round_index}"] = planner_result.output
            self.artifacts.write_role_summary(
                role=self.planner,
                phase=f"review_round_{round_index}",
                result=planner_result,
            )

            merge_ok = self._merge_new_commits(state)
            if not merge_ok:
                state.merge_conflicts += 1
                self.artifacts.append_open_question(
                    f"merge conflict encountered in review round {round_index}"
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

            last_validation = {
                "ok": validation_result.ok,
                "stdout": validation_result.stdout,
                "stderr": validation_result.stderr,
            }
            if validation_result.ok:
                public_pass = True
                self.artifacts.append_status(f"review round {round_index} passed public validation")
                self.team.post_public_validation(
                    round_index=round_index,
                    ok=True,
                    stdout=validation_result.stdout,
                    stderr=validation_result.stderr,
                )
                break

            self.artifacts.append_status(f"review round {round_index} failed public validation")
            self.team.post_public_validation(
                round_index=round_index,
                ok=False,
                stdout=validation_result.stdout,
                stderr=validation_result.stderr,
            )
            self._request_rework_from_coders(
                task,
                round_index,
                validation_result.stdout,
                validation_result.stderr,
                state,
            )

        return public_pass

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
        result = self._invoke_role(self.planner, "finalize", context)
        state.role_outputs[f"{self.planner}:finalize"] = result.output
        self.artifacts.write_role_summary(role=self.planner, phase="finalize", result=result)

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
    ) -> None:
        rework_seed = self.team.seed_rework(
            round_index=round_index,
            validation_stdout=validation_stdout,
            validation_stderr=validation_stderr,
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
                result = future.result()
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
                        "error": str(exc),
                        "driver_phase": driver_phase,
                        "coordination_phase": coordination_phase,
                    },
                )
                raise

            outcome.completed += 1
            output_key = f"{role}:{driver_phase}:{claimed_task.task_id}"
            outcome.role_outputs[output_key] = result.output

            self.artifacts.write_role_summary(
                role=role,
                phase=driver_phase,
                result=result,
                suffix=claimed_task.task_id,
            )

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
        return self._invoke_role(role, driver_phase, context)

    def _merge_new_commits(self, state: HarnessState) -> bool:
        planner_path = self.role_paths[self.planner]
        ok = True

        for coder in self.coders:
            commits = self._list_commits(coder, self.base_commit)
            unseen = [c for c in commits if c not in state.merged_commits[coder]]
            for commit in unseen:
                cherry = run_command(["git", "-C", str(planner_path), "cherry-pick", commit], timeout_sec=180)
                if not cherry.ok:
                    ok = False
                    run_command(["git", "-C", str(planner_path), "cherry-pick", "--abort"], timeout_sec=60)
                    self.artifacts.append_open_question(
                        f"failed to cherry-pick {commit} from {coder}: {cherry.stderr.strip() or cherry.stdout.strip()}"
                    )
                    break
                state.merged_commits[coder].add(commit)

        return ok

    def _invoke_role(self, role: str, phase: str, context: Dict[str, Any]) -> RoleRunResult:
        driver = self.role_drivers[role]
        result = driver.run_phase(
            phase=phase,
            role=role,
            worktree_path=self.role_paths[role],
            run_dir=self.run_dir,
            context=context,
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

        if not result.ok:
            stderr_tail = result.stderr[-2000:] if result.stderr else ""
            raise RuntimeError(
                f"role '{role}' failed during '{phase}' (exit_code={result.exit_code}). "
                f"stderr_tail={stderr_tail!r}"
            )

        return result

    def _default_plan_markdown(self, task: TaskPack) -> str:
        return (
            f"# Plan for {task.task_id}\n\n"
            "1. Inspect task requirements and affected modules.\n"
            "2. Implement core functional changes in coder_a.\n"
            "3. Implement supporting tests, env, and validation updates in coder_b.\n"
            "4. Review and merge coder commits in planner_reviewer.\n"
            "5. Run public validation and iterate until green.\n"
        )

    def _default_subtasks(self, task: TaskPack) -> List[Dict[str, Any]]:
        planner_path = self.role_paths[self.planner]
        files = self._list_files(planner_path)
        top_levels = sorted({p.split("/", 1)[0] for p in files if "/" in p})

        a_paths: List[str] = []
        b_paths: List[str] = []
        for idx, path in enumerate(top_levels):
            if idx % 2 == 0:
                a_paths.append(path)
            else:
                b_paths.append(path)

        if not a_paths:
            a_paths = ["."]
        if not b_paths:
            b_paths = ["tests", "infra", "."]

        coder_a = self.coders[0] if self.coders else "coder_a"
        coder_b = self.coders[1] if len(self.coders) > 1 else coder_a

        return [
            {
                "id": "S1",
                "role": coder_a,
                "title": "Core implementation changes",
                "paths": a_paths,
                "acceptance": "Primary functionality implemented and locally validated.",
            },
            {
                "id": "S2",
                "role": coder_b,
                "title": "Validation and integration polish",
                "paths": b_paths,
                "acceptance": "Tests/env updates aligned with runtime behavior.",
            },
        ]

    def _list_commits(self, role: str, since_commit: str) -> List[str]:
        path = self.role_paths[role]
        result = run_command(
            ["git", "-C", str(path), "rev-list", "--reverse", f"{since_commit}..HEAD"],
            timeout_sec=60,
        )
        ensure_success(result, f"git rev-list {role}")
        commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return commits

    def _list_files(self, repo_path: Path) -> List[str]:
        result = run_command(["git", "-C", str(repo_path), "ls-files"], timeout_sec=30)
        ensure_success(result, "git ls-files")
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

    def _pick_planner(self, roles: Iterable[str]) -> str:
        for role in roles:
            if role == "planner_reviewer":
                return role
        first = next(iter(roles), None)
        if not first:
            raise ValueError("no roles provided")
        return first
