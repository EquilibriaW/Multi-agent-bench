"""
loopbench.shell

Small subprocess helpers used by runtime modules.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CommandResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int
    elapsed_sec: float


class CommandError(RuntimeError):
    pass


def run_command(
    cmd: List[str],
    cwd: Optional[str | Path] = None,
    timeout_sec: int = 600,
    env: Optional[Dict[str, str]] = None,
    stdin_text: Optional[str] = None,
) -> CommandResult:
    started = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=merged_env,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        timeout_msg = f"command timed out after {timeout_sec}s"
        if stderr:
            stderr = f"{stderr}\n{timeout_msg}"
        else:
            stderr = timeout_msg
        return CommandResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            exit_code=124,
            elapsed_sec=elapsed,
        )

    elapsed = time.monotonic() - started
    return CommandResult(
        ok=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
        exit_code=proc.returncode,
        elapsed_sec=elapsed,
    )


def run_shell(
    command: str,
    cwd: Optional[str | Path] = None,
    timeout_sec: int = 600,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    return run_command(["bash", "-lc", command], cwd=cwd, timeout_sec=timeout_sec, env=env)


def ensure_success(result: CommandResult, context: str) -> None:
    if result.ok:
        return
    cmd = context.strip()
    out = result.stdout.strip()
    err = result.stderr.strip()
    message = f"{cmd} failed (exit={result.exit_code})"
    if out:
        message += f"\nstdout:\n{out}"
    if err:
        message += f"\nstderr:\n{err}"
    raise CommandError(message)


def shell_quote(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)
