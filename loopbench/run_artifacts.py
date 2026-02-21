"""
loopbench.run_artifacts

Run-scoped artifact writer helpers used by orchestration components.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .agents import RoleRunResult


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
        safe_suffix = self._safe_suffix(suffix)
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

    def _safe_suffix(self, raw: Optional[str]) -> str:
        if not raw:
            return ""
        chars = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_", "."}:
                chars.append(ch)
            else:
                chars.append("_")
        return "".join(chars)[:120]
