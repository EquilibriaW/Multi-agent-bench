"""
loopbench.sandbox

Local sandbox implementation bound to a single role worktree.
"""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Dict, List, Optional

from .path_utils import resolve_within_root
from .schema import ToolResult
from .shell import run_command
from .time_utils import now_ms


class LocalSandbox:
    def __init__(self, sandbox_name: str, root: str | Path):
        self._sandbox_name = sandbox_name
        self.root = Path(root).resolve()
        self._cancel_event = threading.Event()

    def name(self) -> str:
        return self._sandbox_name

    def exec(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout_sec: int = 600,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        if self._cancel_event.is_set():
            return ToolResult(
                ts_ms=now_ms(),
                ok=False,
                tool="repo.exec",
                stdout="",
                stderr="sandbox closed",
                exit_code=130,
                data={"cwd": str(self.root), "elapsed_sec": 0.0},
            )
        working_dir = self.root if cwd is None else self._resolve_path(cwd)
        result = run_command(
            cmd,
            cwd=working_dir,
            timeout_sec=timeout_sec,
            env=env,
            cancel_event=self._cancel_event,
        )
        return ToolResult(
            ts_ms=now_ms(),
            ok=result.ok,
            tool="repo.exec",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            data={"cwd": str(working_dir), "elapsed_sec": result.elapsed_sec},
        )

    def read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        return resolved.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def apply_patch(self, patch_text: str) -> None:
        # Apply relative to worktree root to match unified diff semantics.
        result = run_command(
            ["git", "-C", str(self.root), "apply", "--whitespace=nowarn", "-"],
            stdin_text=patch_text,
            cancel_event=self._cancel_event,
        )
        if not result.ok:
            raise RuntimeError(f"patch apply failed: {result.stderr.strip()}")

    def close(self) -> None:
        self._cancel_event.set()

    def _resolve_path(self, raw: str) -> Path:
        return resolve_within_root(root=self.root, raw_path=raw)
