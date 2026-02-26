"""
loopbench.e2b_sandbox

E2B Firecracker-backed sandbox implementation.

This backend keeps a local worktree as source-of-truth for git orchestration,
and syncs state to/from an E2B sandbox for command execution isolation.
"""
from __future__ import annotations

import io
import sys
import tarfile
import threading
from dataclasses import dataclass
from pathlib import Path
import shlex
import time
from typing import Dict, List, Optional

from .path_utils import resolve_within_root
from .schema import ToolResult
from .shell import run_command, shell_quote
from .time_utils import now_ms

try:
    from e2b.sandbox.commands.command_handle import CommandExitException as _CommandExitException
except ImportError:
    _CommandExitException = type(None)  # noqa: will never match if e2b not installed

# Process-level registry of active E2B sandbox IDs for shutdown cleanup.
_active_sandbox_ids: set[str] = set()
_sandbox_registry_lock = threading.Lock()


@dataclass
class E2BOptions:
    api_key: str
    template: Optional[str]
    timeout_sec: int
    allow_internet_access: bool


class E2BFirecrackerSandbox:
    def __init__(self, sandbox_name: str, root: str | Path, options: E2BOptions):
        self._sandbox_name = sandbox_name
        self.root = Path(root).resolve()
        self.options = options
        self._file_request_timeout_sec = float(max(30, min(options.timeout_sec, 120)))
        self._control_request_timeout_sec = float(max(30, min(options.timeout_sec, 300)))
        # Sync budget should follow the configured sandbox timeout budget to avoid
        # premature sync aborts on large/slow repo transfers.
        self._sync_budget_sec = float(max(90, options.timeout_sec))

        try:
            from e2b_code_interpreter import Sandbox as E2BSandbox
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "e2b backend requested but e2b-code-interpreter is not installed. "
                "Install with: pip install e2b-code-interpreter"
            ) from exc

        create_kwargs: Dict[str, object] = {
            "timeout": options.timeout_sec,
            "allow_internet_access": options.allow_internet_access,
            "api_key": options.api_key,
        }
        if options.template:
            create_kwargs["template"] = options.template

        self._sandbox = E2BSandbox.create(**create_kwargs)
        self._last_timeout_refresh = time.monotonic()
        with _sandbox_registry_lock:
            _active_sandbox_ids.add(self._sandbox.sandbox_id)
        # E2B sandboxes run as an unprivileged user; top-level /workspace is not writable.
        self._remote_root = "/home/user/workspace"

        try:
            self._run_remote(f"mkdir -p {shell_quote([self._remote_root])}", cwd=None)
            self._sync_local_to_remote()
            # Initialize a proper git repo in the remote workspace so that
            # git commands work inside the sandbox.  The local worktree has a
            # .git file (gitdir pointer) that references host paths unreachable
            # from inside E2B; we exclude it from sync and create a fresh repo.
            self._run_remote(
                f"cd {shlex.quote(self._remote_root)} && "
                "git init -q && "
                "git config user.email 'loopbench@sandbox' && "
                "git config user.name 'loopbench' && "
                "git add -A && "
                "git commit -q -m 'initial sync' --allow-empty",
                cwd=self._remote_root,
            )
        except BaseException:
            # If post-create setup fails, kill the E2B sandbox so it
            # doesn't leak.  The caller never gets a reference to this
            # object (the constructor raised), so nobody else can close it.
            try:
                self._sandbox.kill()
            except Exception:  # noqa: BLE001
                pass
            raise

    # Only call set_timeout if more than this fraction of the TTL has elapsed
    # since the last refresh.  Avoids redundant API calls on rapid exec sequences.
    _REFRESH_AFTER_FRACTION = 0.25

    def name(self) -> str:
        return self._sandbox_name

    def _refresh_timeout(self) -> None:
        """Reset the E2B sandbox TTL if enough time has elapsed since last refresh."""
        elapsed = time.monotonic() - self._last_timeout_refresh
        threshold = self.options.timeout_sec * self._REFRESH_AFTER_FRACTION
        if elapsed < threshold:
            return
        try:
            self._sandbox.set_timeout(self.options.timeout_sec)
            self._last_timeout_refresh = time.monotonic()
        except Exception:  # noqa: BLE001
            # If set_timeout fails the sandbox may already be dead.
            # Let the subsequent exec call surface the real error.
            pass

    def exec(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout_sec: int = 600,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        # Refresh sandbox TTL before each command to prevent expiry during
        # long-running rollouts that span multiple phases.
        self._refresh_timeout()

        cmd_str = shell_quote(cmd)
        if env:
            remote_env = self._translate_env(env)
            env_pairs = " ".join(f"{k}={shlex.quote(v)}" for k, v in remote_env.items())
            cmd_str = f"env {env_pairs} {cmd_str}"
        local_cwd = self.root if cwd is None else self._resolve_local_path(cwd)
        remote_cwd = self._to_remote_dir(local_cwd)
        command_request_timeout = float(max(timeout_sec + 30, int(self._control_request_timeout_sec)))

        self._sync_local_to_remote()
        try:
            result = self._commands_run(
                cmd_str,
                cwd=remote_cwd,
                timeout=float(timeout_sec),
                request_timeout=command_request_timeout,
            )
        except Exception as run_exc:  # noqa: BLE001
            # The E2B SDK streaming connection can drop mid-execution
            # (h11 RemoteProtocolError: "peer closed connection without
            # sending complete message body").  The command may have
            # completed inside the sandbox — try to recover files.
            recovered = self._try_recover_after_stream_drop(run_exc)
            if not recovered:
                raise
            return ToolResult(
                ts_ms=now_ms(),
                ok=False,
                tool="repo.exec",
                stdout="",
                stderr=f"e2b stream dropped (files recovered): {run_exc}",
                exit_code=1,
                data={
                    "cwd": str(local_cwd),
                    "backend": "e2b_firecracker",
                    "sandbox_id": getattr(self._sandbox, "sandbox_id", None),
                    "stream_recovery": True,
                },
            )
        self._sync_remote_to_local()

        stderr = result.stderr or ""
        if result.error:
            if stderr:
                stderr = f"{stderr}\n{result.error}"
            else:
                stderr = result.error

        return ToolResult(
            ts_ms=now_ms(),
            ok=(result.exit_code == 0 and not result.error),
            tool="repo.exec",
            stdout=result.stdout or "",
            stderr=stderr,
            exit_code=int(result.exit_code),
            data={
                "cwd": str(local_cwd),
                "backend": "e2b_firecracker",
                "sandbox_id": getattr(self._sandbox, "sandbox_id", None),
            },
        )

    def read_file(self, path: str) -> str:
        self._sync_remote_to_local()
        resolved = self._resolve_local_path(path)
        return resolved.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        resolved = self._resolve_local_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

        remote_path = self._to_remote_file(resolved)
        self._files_write(remote_path, content, request_timeout=self._file_request_timeout_sec)

    def apply_patch(self, patch_text: str) -> None:
        # Keep local git history authoritative for merge orchestration.
        result = run_command(
            ["git", "-C", str(self.root), "apply", "--whitespace=nowarn", "-"],
            stdin_text=patch_text,
        )
        if not result.ok:
            raise RuntimeError(f"patch apply failed: {result.stderr.strip()}")
        self._sync_local_to_remote()

    def close(self) -> None:
        try:
            sid = getattr(self._sandbox, "sandbox_id", None)
            self._sandbox.kill()
            if sid:
                with _sandbox_registry_lock:
                    _active_sandbox_ids.discard(sid)
        except Exception:  # noqa: BLE001
            pass

    def _is_within_root(self, path: Path) -> bool:
        """Return True only if *resolved* path is inside self.root."""
        try:
            path.resolve().relative_to(self.root)
            return True
        except ValueError:
            return False

    def _try_recover_after_stream_drop(self, original_exc: Exception) -> bool:
        """Attempt to sync files back from a sandbox whose streaming connection dropped.

        The E2B SDK's HTTP/1.1 chunked stream to the sandbox can close
        mid-command (h11 RemoteProtocolError).  The sandbox VM is often
        still alive — the driver script inside may have finished and written
        its output files.  This method tries to sync those files back so the
        harness can read them, preventing total work loss.

        Returns True if file recovery succeeded (caller should return a
        degraded ToolResult), False if recovery failed (caller should re-raise).
        """
        exc_str = str(original_exc).lower()
        is_stream_error = any(
            marker in exc_str
            for marker in ("peer closed", "incomplete chunked", "remoteprotocolerror", "connection reset")
        )
        if not is_stream_error:
            return False

        try:
            self._sync_remote_to_local()
            return True
        except Exception:  # noqa: BLE001
            # Sandbox is truly dead — nothing to recover.
            return False

    def _sync_local_to_remote(self) -> None:
        start = time.monotonic()
        # Create a tar archive of the local root in memory, upload it
        # as a single blob, and extract on the remote side.  This
        # replaces hundreds of individual file-write API calls with one
        # write + one command, cutting sync time from minutes to seconds
        # for large repos.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in sorted(self.root.rglob("*")):
                if path.is_symlink():
                    continue
                if not self._is_within_root(path):
                    continue
                if not (path.is_file() or path.is_dir()):
                    continue
                rel = path.relative_to(self.root)
                # Skip .git file/dir — the local worktree's .git has a gitdir
                # pointer to host paths that don't exist inside E2B.  A fresh
                # git repo is initialized on the remote side instead.
                if rel.parts and rel.parts[0] == ".git":
                    continue
                # rglob already walks descendants; avoid recursively re-adding
                # directory contents for every parent directory.
                tar.add(str(path), arcname=rel.as_posix(), recursive=False)
        tar_bytes = buf.getvalue()

        self._ensure_sync_budget(start=start, operation="upload tar")
        remote_tar = "/tmp/.sync_upload.tar.gz"
        upload_timeout = float(max(self._sync_budget_sec, self._file_request_timeout_sec))
        self._files_write(remote_tar, tar_bytes, request_timeout=upload_timeout)

        self._ensure_sync_budget(start=start, operation="extract tar")
        extract_timeout = float(max(120, self._sync_budget_sec - (time.monotonic() - start)))
        extract_result = self._commands_run(
            f"tar xzf {shlex.quote(remote_tar)} -C {shlex.quote(self._remote_root)} && rm -f {shlex.quote(remote_tar)}",
            cwd=self._remote_root,
            timeout=extract_timeout,
            request_timeout=extract_timeout + 30,
        )
        if extract_result.exit_code != 0 or extract_result.error:
            raise RuntimeError(
                f"e2b sync upload-extract failed: exit_code={extract_result.exit_code}; "
                f"stderr={extract_result.stderr}; error={extract_result.error}"
            )

    def _sync_remote_to_local(self) -> None:
        start = time.monotonic()
        # Tar the remote workspace into a single blob, download it,
        # and extract locally.  Mirrors the tar-based upload approach.
        remote_tar = "/tmp/.sync_download.tar.gz"
        self._ensure_sync_budget(start=start, operation="create remote tar")
        pack_timeout = float(max(120, self._sync_budget_sec - (time.monotonic() - start)))
        pack_result = self._commands_run(
            f"tar czf {shlex.quote(remote_tar)}"
            f" --exclude=.git"
            f" -C {shlex.quote(self._remote_root)} .",
            cwd=self._remote_root,
            timeout=pack_timeout,
            request_timeout=pack_timeout + 30,
        )
        if pack_result.exit_code != 0 or pack_result.error:
            raise RuntimeError(
                f"e2b sync download-pack failed: exit_code={pack_result.exit_code}; "
                f"stderr={pack_result.stderr}; error={pack_result.error}"
            )

        self._ensure_sync_budget(start=start, operation="download tar")
        download_timeout = float(max(self._sync_budget_sec, self._file_request_timeout_sec))
        tar_bytes = bytes(self._files_read(
            remote_tar,
            format="bytes",
            request_timeout=download_timeout,
        ))

        # Clean up remote tar.
        try:
            self._commands_run(
                f"rm -f {shlex.quote(remote_tar)}",
                cwd=self._remote_root,
                timeout=10.0,
                request_timeout=30.0,
            )
        except Exception:  # noqa: BLE001
            pass

        # Replace local tree with remote contents.
        # Remove existing files first, then extract.  Preserve .git
        # (file or directory) — the local worktree's .git pointer
        # references host paths needed by the harness for git ops.
        for path in sorted(self.root.rglob("*"), reverse=True):
            rel = path.relative_to(self.root)
            if rel.parts and rel.parts[0] == ".git":
                continue
            if path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_file() and self._is_within_root(path):
                path.unlink(missing_ok=True)

        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(path=str(self.root))

    def _to_remote_dir(self, local_dir: Path) -> str:
        rel = local_dir.relative_to(self.root)
        if not rel.parts:
            return self._remote_root
        return f"{self._remote_root}/{rel.as_posix()}"

    def _to_remote_file(self, local_path: Path) -> str:
        rel = local_path.relative_to(self.root)
        return f"{self._remote_root}/{rel.as_posix()}"

    def _resolve_local_path(self, raw: str) -> Path:
        return resolve_within_root(root=self.root, raw_path=raw)

    def _run_remote(self, command: str, *, cwd: str | None) -> None:
        result = self._commands_run(
            command,
            cwd=cwd,
            timeout=120.0,
            request_timeout=max(self._control_request_timeout_sec, 150.0),
        )
        if result.exit_code != 0 or result.error:
            raise RuntimeError(f"e2b command failed: {command}; stderr={result.stderr}; error={result.error}")

    def _commands_run(
        self,
        command: str,
        *,
        cwd: str | None,
        timeout: float,
        request_timeout: float,
    ):
        try:
            return self._sandbox.commands.run(
                command,
                cwd=cwd,
                timeout=timeout,
                request_timeout=request_timeout,
            )
        except TypeError:
            try:
                return self._sandbox.commands.run(command, cwd=cwd, timeout=timeout)
            except _CommandExitException as exc:
                return exc  # CommandExitException is a CommandResult subclass
        except _CommandExitException as exc:
            return exc  # CommandExitException is a CommandResult subclass

    def _files_write(self, path: str, data: str | bytes, *, request_timeout: float):
        try:
            return self._sandbox.files.write(path, data, request_timeout=request_timeout)
        except TypeError:
            return self._sandbox.files.write(path, data)

    def _files_read(self, path: str, *, format: str, request_timeout: float):
        try:
            return self._sandbox.files.read(path, format=format, request_timeout=request_timeout)
        except TypeError:
            return self._sandbox.files.read(path, format=format)

    def _translate_env(self, env: Dict[str, str]) -> Dict[str, str]:
        translated: Dict[str, str] = {}
        for key, value in env.items():
            translated[key] = self._translate_env_value(value)
        return translated

    def _translate_env_value(self, value: str) -> str:
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            rel = path.resolve().relative_to(self.root)
        except Exception:  # noqa: BLE001
            return value
        if not rel.parts:
            return self._remote_root
        return f"{self._remote_root}/{rel.as_posix()}"

    def _ensure_sync_budget(self, *, start: float, operation: str) -> None:
        elapsed = time.monotonic() - start
        if elapsed <= self._sync_budget_sec:
            return
        raise TimeoutError(
            f"e2b sync exceeded {self._sync_budget_sec:.1f}s while {operation}"
        )


def kill_all_active_sandboxes() -> None:
    """Kill all E2B sandboxes created by this process. For shutdown cleanup."""
    with _sandbox_registry_lock:
        ids = list(_active_sandbox_ids)
    if not ids:
        return
    try:
        from e2b_code_interpreter import Sandbox as E2BSandbox
        for sid in ids:
            try:
                E2BSandbox.kill(sid)
            except Exception:  # noqa: BLE001
                pass
        with _sandbox_registry_lock:
            _active_sandbox_ids.difference_update(ids)
        print(f"[loopbench] shutdown: killed {len(ids)} E2B sandbox(es)", file=sys.stderr)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[loopbench] warning: E2B cleanup failed: {exc}", file=sys.stderr)
