"""
loopbench.run_artifacts

Run-scoped artifact writer helpers used by orchestration components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .agents import RoleRunResult
from .path_utils import safe_path_component
from .review_logic import PublicValidationRecord, ReviewDecision


@dataclass
class ReviewLedgerEntry:
    round_index: int
    decision: str  # "accept" | "rework"
    commits_merged: Dict[str, List[str]] = field(default_factory=dict)
    open_issues: List[str] = field(default_factory=list)
    validation_passed: bool = False
    merge_ok: bool = True
    summary: str = ""
    cause: str = ""  # "validation_fail" | "merge_conflict" | "reviewer_rework" | "no_accepted_work" | "accept"
    validation_stderr_tail: str = ""  # last 1000 chars of validation stderr


class ReviewLedger:
    def __init__(self, path: Path):
        self._path = path
        self._entries: List[ReviewLedgerEntry] = []
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for raw in data:
                    self._entries.append(ReviewLedgerEntry(**raw))
            except (json.JSONDecodeError, TypeError):
                pass

    def append(self, entry: ReviewLedgerEntry) -> None:
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps([self._serialize(e) for e in self._entries], indent=2),
            encoding="utf-8",
        )

    def as_context(self) -> Dict[str, Any]:
        """Last 3 entries in full, earlier rounds as one-line summaries."""
        if not self._entries:
            return {"entries": [], "prior_summaries": []}
        recent = self._entries[-3:]
        earlier = self._entries[:-3]
        return {
            "entries": [self._serialize(e) for e in recent],
            "prior_summaries": [
                f"round {e.round_index}: {e.decision} cause={e.cause}, merge_ok={e.merge_ok}, validation={e.validation_passed}"
                for e in earlier
            ],
        }

    @staticmethod
    def _serialize(entry: ReviewLedgerEntry) -> Dict[str, Any]:
        return {
            "round_index": entry.round_index,
            "decision": entry.decision,
            "commits_merged": entry.commits_merged,
            "open_issues": entry.open_issues,
            "validation_passed": entry.validation_passed,
            "merge_ok": entry.merge_ok,
            "summary": entry.summary,
            "cause": entry.cause,
            "validation_stderr_tail": entry.validation_stderr_tail,
        }


class RunArtifacts:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()

    def append_status(self, line: str) -> None:
        path = self.run_dir / "status.md"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"- {line}\n")

    def append_open_question(self, line: str) -> None:
        path = self.run_dir / "open_questions.md"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"- {line}\n")

    def write_public_validate(self, round_index: int, stdout: str, stderr: str) -> None:
        path = self.run_dir / "public_validate" / f"round_{round_index}.log"
        path.write_text(f"STDOUT\n{stdout}\n\nSTDERR\n{stderr}\n", encoding="utf-8")

    def write_role_summary(
        self,
        *,
        role: str,
        phase: str,
        result: RoleRunResult,
        suffix: Optional[str] = None,
    ) -> None:
        safe_phase = phase.replace("/", "_")
        safe_suffix = safe_path_component(suffix)
        stem = f"{role}_{safe_phase}" if not safe_suffix else f"{role}_{safe_phase}_{safe_suffix}"
        filename = f"{stem}.md"
        summary_path = self.run_dir / "role_summaries" / filename
        stdout_path = self.run_dir / "role_stdio" / f"{stem}.stdout.log"
        stderr_path = self.run_dir / "role_stdio" / f"{stem}.stderr.log"

        payload = json.dumps(result.output, indent=2)
        summary_path.write_text(
            "\n".join(
                [
                    f"# {role} {phase}",
                    "",
                    f"- ok: {result.ok}",
                    f"- exit_code: {result.exit_code}",
                    "",
                    "## stdout",
                    "```",
                    result.stdout[-6000:],
                    "```",
                    "",
                    "## stderr",
                    "```",
                    result.stderr[-6000:],
                    "```",
                    "",
                    "## output",
                    "```json",
                    payload,
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        stdout_path.write_text(result.stdout, encoding="utf-8")
        stderr_path.write_text(result.stderr, encoding="utf-8")

    def read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def write_review_round_audit(
        self,
        *,
        planner: str,
        round_index: int,
        review_output: Dict[str, Any],
        verify_output: Dict[str, Any],
        decision: ReviewDecision,
        inspected_commits_by_role: Dict[str, Set[str]],
        merged_commits_this_round: Dict[str, List[str]],
        merge_ok: bool,
        public_validation: PublicValidationRecord,
        accepted: bool,
        force_rework: bool,
    ) -> Path:
        out_path = self.run_dir / "artifacts" / "review_audit" / f"round_{round_index}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        review_trace_path = self.run_dir / "role_runtime" / f"{planner}_review_round_{round_index}_command_trace.json"
        verify_trace_path = self.run_dir / "role_runtime" / (
            f"{planner}_review_verify_round_{round_index}_verify_command_trace.json"
        )
        payload = {
            "round_index": round_index,
            "accepted": bool(accepted),
            "force_rework": bool(force_rework),
            "request_rework": bool(decision.request_rework),
            "merge_ok": bool(merge_ok),
            "merge_commits_by_role": {role: list(commits) for role, commits in decision.merge_commits_by_role.items()},
            "invalid_merge_roles": list(decision.invalid_merge_roles),
            "uninspected_nominated_commits_by_role": {
                role: list(commits) for role, commits in decision.uninspected_nominated_commits_by_role.items()
            },
            "inspected_commits_by_role": {role: sorted(commits) for role, commits in inspected_commits_by_role.items()},
            "merged_commits_this_round": {role: list(commits) for role, commits in merged_commits_this_round.items()},
            "review_dynamic_checks_ran": bool(decision.dynamic_checks_ran),
            "public_validation": {
                "ok": bool(public_validation.ok),
                "noop": bool(public_validation.noop),
                "state": public_validation.state.value,
                "policy": public_validation.policy,
                "stdout_tail": public_validation.stdout[-3000:],
                "stderr_tail": public_validation.stderr[-3000:],
            },
            "review_command_trace_path": str(review_trace_path) if review_trace_path.exists() else None,
            "verify_command_trace_path": str(verify_trace_path) if verify_trace_path.exists() else None,
            "review_summary": str(review_output.get("summary") or ""),
            "review_notes": str(review_output.get("notes") or ""),
            "verify_summary": str(verify_output.get("summary") or ""),
            "verify_notes": str(verify_output.get("notes") or ""),
            "review_commands_run": review_output.get("commands_run"),
            "verify_commands_run": verify_output.get("commands_run"),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    def write_workflow_summary(
        self,
        *,
        task_id: str,
        planner_role: str,
        coder_roles: List[str],
        coordination_db_path: str,
        plan_path: str,
        subtasks_path: str,
        review_rounds_executed: int,
        public_validate_policy: str,
        public_validation_attempts: int,
        public_validation_noop_rounds: int,
        merge_conflicts: int,
        merged_commits: Dict[str, Set[str]],
        role_outputs: Dict[str, Dict[str, Any]],
        coordination_summary: Dict[str, int],
        public_pass: bool,
        final_patch_path: str,
    ) -> Path:
        planner_mutations: Dict[str, List[str]] = {}
        for key, output in role_outputs.items():
            if not key.startswith(f"{planner_role}:") or not isinstance(output, dict):
                continue
            applied_paths = output.get("applied_paths")
            if not isinstance(applied_paths, list):
                continue
            clean_paths = [str(p) for p in applied_paths if isinstance(p, str)]
            if clean_paths:
                planner_mutations[key] = clean_paths

        out_path = self.run_dir / "artifacts" / "workflow_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "planner_role": planner_role,
            "coder_roles": list(coder_roles),
            "coordination_db_path": coordination_db_path,
            "plan_path": plan_path,
            "subtasks_path": subtasks_path,
            "review_rounds_executed": review_rounds_executed,
            "public_validate_policy": public_validate_policy,
            "public_validation_attempts": public_validation_attempts,
            "public_validation_noop_rounds": public_validation_noop_rounds,
            "merge_conflicts": merge_conflicts,
            "merged_commits": {role: sorted(commits) for role, commits in merged_commits.items()},
            "planner_mutations": planner_mutations,
            "planner_mutation_count": sum(len(paths) for paths in planner_mutations.values()),
            "coordination_summary": dict(coordination_summary),
            "public_pass": bool(public_pass),
            "final_patch_path": final_patch_path,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path
