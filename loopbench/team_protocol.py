"""
loopbench.team_protocol

Canonical DB-backed team coordination protocol for planner/coder orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .coordination import ClaimedTask, CoordinationDB


@dataclass(frozen=True)
class ReworkSeed:
    phase: str
    seeded_count: int


class TeamProtocol:
    def __init__(
        self,
        *,
        db: CoordinationDB,
        planner: str,
        coders: Sequence[str],
    ):
        self.db = db
        self.planner = planner
        self.coders = list(coders)
        self._implementation_paths_by_role: Dict[str, List[str]] = {}

    @property
    def db_path(self) -> Path:
        return self.db.db_path

    def initialize(self) -> None:
        self.db.initialize()

    def summary(self) -> Dict[str, int]:
        return self.db.summary()

    def inbox(self, *, role: str, phase: str, limit: int) -> List[Dict[str, Any]]:
        return self.db.latest_messages(role=role, phase=phase, limit=limit)

    def seed_implementation(
        self,
        *,
        task_id: str,
        plan_path: Path,
        subtasks_path: Path,
        subtasks: List[Dict[str, Any]],
    ) -> int:
        self._implementation_paths_by_role = self._collect_paths_by_role(subtasks)
        seeded = self.db.seed_tasks(phase="implementation", subtasks=subtasks)
        self.db.post_message(
            phase="implementation",
            from_role=self.planner,
            to_role=None,
            kind="plan_published",
            body={
                "task_id": task_id,
                "seeded_subtasks": len(seeded),
                "plan_path": str(plan_path),
                "subtasks_path": str(subtasks_path),
            },
        )
        self._post_assignments(phase="implementation", subtasks=subtasks, round_index=None)
        return len(seeded)

    def seed_rework(
        self,
        *,
        round_index: int,
        validation_stdout: str,
        validation_stderr: str,
        planner_feedback_by_role: Optional[Dict[str, str]] = None,
    ) -> ReworkSeed:
        phase = self.rework_phase(round_index)
        feedback_by_role = planner_feedback_by_role or {}
        subtasks = [
            {
                "id": f"R{round_index}_{coder}",
                "role": coder,
                "title": f"Rework for review round {round_index}",
                "paths": list(self._implementation_paths_by_role.get(coder) or []),
                "acceptance": "Address planner feedback and unblock public validation.",
            }
            for coder in self.coders
        ]

        seeded = self.db.seed_tasks(phase=phase, subtasks=subtasks)
        for subtask in subtasks:
            self.db.post_message(
                phase=phase,
                from_role=self.planner,
                to_role=subtask["role"],
                kind="rework_request",
                round_index=round_index,
                body={
                    "subtask": subtask,
                    "public_validate_stdout_tail": validation_stdout[-3000:],
                    "public_validate_stderr_tail": validation_stderr[-3000:],
                    "planner_feedback": feedback_by_role.get(str(subtask["role"]), ""),
                },
            )

        return ReworkSeed(phase=phase, seeded_count=len(seeded))

    def post_public_validation(
        self,
        *,
        round_index: int,
        ok: bool,
        stdout: str,
        stderr: str,
    ) -> None:
        phase = f"review_round_{round_index}"
        kind = "public_validation_passed" if ok else "public_validation_failed"
        body: Dict[str, Any] = {"public_validation_ok": ok}
        if not ok:
            body["stdout_tail"] = stdout[-2000:]
            body["stderr_tail"] = stderr[-2000:]
        self.db.post_message(
            phase=phase,
            from_role=self.planner,
            to_role=None,
            kind=kind,
            round_index=round_index,
            body=body,
        )

    def claim_next_task(self, *, phase: str, role: str) -> Optional[ClaimedTask]:
        return self.db.claim_next_task(phase=phase, role=role)

    def mark_claimed(self, *, phase: str, role: str, task: ClaimedTask, round_index: Optional[int]) -> None:
        self.db.post_message(
            phase=phase,
            from_role=role,
            to_role=self.planner,
            kind="task_claimed",
            round_index=round_index,
            body={"task_id": task.task_id, "title": task.title},
        )

    def mark_completed(
        self,
        *,
        phase: str,
        role: str,
        task: ClaimedTask,
        round_index: Optional[int],
        result: Dict[str, Any],
    ) -> bool:
        ok = self.db.complete_task(
            task_id=task.task_id,
            role=role,
            claim_token=task.claim_token,
            result=result,
        )
        self.db.post_message(
            phase=phase,
            from_role=role,
            to_role=self.planner,
            kind="task_completed",
            round_index=round_index,
            body={
                "task_id": task.task_id,
                "driver_phase": result.get("driver_phase"),
                "status": result.get("status"),
                "output_keys": result.get("output_keys"),
            },
        )
        return ok

    def mark_failed(
        self,
        *,
        phase: str,
        role: str,
        task: ClaimedTask,
        round_index: Optional[int],
        error: Dict[str, Any],
    ) -> None:
        self.db.fail_task(
            task_id=task.task_id,
            role=role,
            claim_token=task.claim_token,
            error=error,
        )
        self.db.post_message(
            phase=phase,
            from_role=role,
            to_role=self.planner,
            kind="task_failed",
            round_index=round_index,
            body={"task_id": task.task_id, "error": error.get("error", "")},
        )

    @staticmethod
    def rework_phase(round_index: int) -> str:
        return f"rework_round_{round_index}"

    def _post_assignments(
        self,
        *,
        phase: str,
        subtasks: List[Dict[str, Any]],
        round_index: Optional[int],
    ) -> None:
        for item in subtasks:
            role = str(item.get("role") or "")
            if not role:
                continue
            self.db.post_message(
                phase=phase,
                from_role=self.planner,
                to_role=role,
                kind="assignment",
                round_index=round_index,
                body={"subtask": item},
            )

    def _collect_paths_by_role(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for item in subtasks:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if not role:
                continue
            raw_paths = item.get("paths")
            if not isinstance(raw_paths, list):
                continue
            bucket = out.setdefault(role, [])
            for raw_path in raw_paths:
                if not isinstance(raw_path, str):
                    continue
                path = raw_path.strip()
                if not path or path in bucket:
                    continue
                bucket.append(path)
        return out
