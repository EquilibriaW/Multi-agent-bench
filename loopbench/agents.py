"""
loopbench.agents

Role-driver abstraction for integrating external coding agents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import shlex
from typing import Any, Dict, Protocol

from .config import RoleConfig
from .interfaces import Sandbox


@dataclass
class RoleRunResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    output: Dict[str, Any]


class RoleDriver(Protocol):
    def run_phase(
        self,
        phase: str,
        role: str,
        worktree_path: Path,
        run_dir: Path,
        context: Dict[str, Any],
    ) -> RoleRunResult:
        ...


class NoopRoleDriver:
    def run_phase(
        self,
        phase: str,
        role: str,
        worktree_path: Path,
        run_dir: Path,
        context: Dict[str, Any],
    ) -> RoleRunResult:
        output = {
            "status": "noop",
            "phase": phase,
            "role": role,
            "notes": "Noop driver performed no direct edits.",
        }
        return RoleRunResult(ok=True, stdout="", stderr="", exit_code=0, output=output)


class ShellRoleDriver:
    def __init__(
        self,
        command: str,
        env: Dict[str, str] | None = None,
        model: str | None = None,
        sandbox: Sandbox | None = None,
    ):
        if not command:
            raise ValueError("shell role driver requires a non-empty command")
        if sandbox is None:
            raise ValueError("shell role driver requires a sandbox")
        self.command = command
        self.extra_env = env or {}
        self.model = model
        self.sandbox = sandbox

    def run_phase(
        self,
        phase: str,
        role: str,
        worktree_path: Path,
        run_dir: Path,
        context: Dict[str, Any],
    ) -> RoleRunResult:
        state_dir = worktree_path / ".loopbench" / "role_runtime"
        archive_dir = run_dir / "role_runtime"
        state_dir.mkdir(parents=True, exist_ok=True)
        archive_dir.mkdir(parents=True, exist_ok=True)

        runtime_suffix_raw = context.get("runtime_suffix")
        runtime_suffix = self._safe_suffix(runtime_suffix_raw if isinstance(runtime_suffix_raw, str) else None)
        stem = f"{role}_{phase}" if not runtime_suffix else f"{role}_{phase}_{runtime_suffix}"

        context_path = state_dir / f"{stem}_context.json"
        output_path = state_dir / f"{stem}_output.json"

        context_path.write_text(json.dumps(context, indent=2), encoding="utf-8")
        if output_path.exists():
            output_path.unlink()

        env = {
            "LB_ROLE": role,
            "LB_PHASE": phase,
            "LB_WORKTREE": str(worktree_path),
            "LB_RUN_DIR": str(run_dir),
            "LB_CONTEXT_JSON": str(context_path),
            "LB_OUTPUT_JSON": str(output_path),
            "LB_ROLE_EXECUTION_CONTEXT": "sandbox",
        }
        if self.model:
            env["LB_MODEL"] = self.model
        coordination_db = context.get("coordination_db_path")
        if not coordination_db:
            coordination_meta = context.get("coordination")
            if isinstance(coordination_meta, dict):
                coordination_db = coordination_meta.get("db_path")
        if isinstance(coordination_db, str) and coordination_db.strip():
            env["LB_COORD_DB"] = coordination_db
        env.update(self.extra_env)

        sandbox_command = self._prepare_sandbox_command(worktree_path)
        sandbox_result = self.sandbox.exec(
            ["bash", "-lc", sandbox_command],
            cwd=str(worktree_path),
            timeout_sec=3600,
            env=env,
        )
        ok = sandbox_result.ok
        stdout = sandbox_result.stdout
        stderr = sandbox_result.stderr
        exit_code = int(sandbox_result.exit_code) if sandbox_result.exit_code is not None else 1

        parsed_output: Dict[str, Any] = {}
        if output_path.exists():
            try:
                parsed_output = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parsed_output = {"error": f"invalid output json: {exc}"}
        self._archive_runtime_outputs(state_dir=state_dir, archive_dir=archive_dir, stem=stem)

        return RoleRunResult(
            ok=ok,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            output=parsed_output,
        )

    def _archive_runtime_outputs(self, *, state_dir: Path, archive_dir: Path, stem: str) -> None:
        pattern = f"{stem}_*"
        for path in sorted(state_dir.glob(pattern)):
            if not path.is_file():
                continue
            shutil.copy2(path, archive_dir / path.name)

    def _prepare_sandbox_command(self, worktree_path: Path) -> str:
        parts = shlex.split(self.command)
        if not parts:
            return self.command

        first = Path(parts[0])
        if first.is_absolute() and first.exists() and first.is_file():
            staged_dir = worktree_path / ".loopbench" / "role_driver_bin"
            staged_dir.mkdir(parents=True, exist_ok=True)
            staged_path = staged_dir / first.name
            staged_path.write_bytes(first.read_bytes())
            staged_path.chmod(0o755)
            parts[0] = str(staged_path.relative_to(worktree_path))
            return " ".join(shlex.quote(part) for part in parts)

        return self.command

    def _safe_suffix(self, raw: str | None) -> str:
        if not raw:
            return ""
        chars = []
        for ch in raw:
            if ch.isalnum() or ch in {"-", "_", "."}:
                chars.append(ch)
            else:
                chars.append("_")
        return "".join(chars)[:120]


def build_role_driver(role_cfg: RoleConfig, *, sandbox: Sandbox | None = None) -> RoleDriver:
    if role_cfg.driver == "noop":
        return NoopRoleDriver()
    if role_cfg.driver == "shell":
        return ShellRoleDriver(
            command=role_cfg.command or "",
            env=role_cfg.env,
            model=role_cfg.model,
            sandbox=sandbox,
        )
    raise ValueError(f"Unsupported role driver: {role_cfg.driver}")
