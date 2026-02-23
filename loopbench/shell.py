"""
loopbench.shell

Small subprocess helpers used by runtime modules.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
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
    cancel_event: threading.Event | None = None,
) -> CommandResult:
    started = time.monotonic()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdin=subprocess.PIPE if stdin_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if stdin_text is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_text)
        except BrokenPipeError:
            pass
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            proc.stdin = None
    stdout = ""
    stderr = ""
    cancelled = False
    timed_out = False

    while True:
        elapsed = time.monotonic() - started
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            _terminate_process_tree(proc)
            stdout, stderr = _safe_communicate(proc)
            break
        if elapsed >= timeout_sec:
            timed_out = True
            _terminate_process_tree(proc)
            stdout, stderr = _safe_communicate(proc)
            break
        poll_timeout = min(0.2, max(0.01, timeout_sec - elapsed))
        try:
            stdout, stderr = proc.communicate(timeout=poll_timeout)
            break
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            _terminate_process_tree(proc)
            _safe_communicate(proc)
            raise

    elapsed = time.monotonic() - started
    if cancelled:
        cancel_msg = "command cancelled"
        if stderr:
            stderr = f"{stderr}\n{cancel_msg}"
        else:
            stderr = cancel_msg
        return CommandResult(
            ok=False,
            stdout=stdout,
            stderr=stderr,
            exit_code=130,
            elapsed_sec=elapsed,
        )
    if timed_out:
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

    exit_code = int(proc.returncode) if proc.returncode is not None else 1
    return CommandResult(
        ok=exit_code == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        elapsed_sec=elapsed,
    )


def run_shell(
    command: str,
    cwd: Optional[str | Path] = None,
    timeout_sec: int = 600,
    env: Optional[Dict[str, str]] = None,
    cancel_event: threading.Event | None = None,
) -> CommandResult:
    return run_command(
        ["bash", "-lc", command],
        cwd=cwd,
        timeout_sec=timeout_sec,
        env=env,
        cancel_event=cancel_event,
    )


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


def _safe_communicate(proc: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(proc)
        try:
            out, err = proc.communicate(timeout=1)
        except Exception:  # noqa: BLE001
            out, err = "", ""
    except Exception:  # noqa: BLE001
        out, err = "", ""
    return out or "", err or ""


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return
    except Exception:  # noqa: BLE001
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
