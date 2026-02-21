"""
loopbench.e2b_sandbox

E2B Firecracker-backed sandbox implementation.

This backend keeps a local worktree as source-of-truth for git orchestration,
and syncs state to/from an E2B sandbox for command execution isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Dict, List, Optional

from .path_utils import resolve_within_root
from .schema import ToolResult
from .shell import run_command, shell_quote
from .time_utils import now_ms


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

        try:
            from e2b_code_interpreter import Sandbox as E2BSandbox
            from e2b.sandbox.filesystem.filesystem import FileType
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "e2b backend requested but e2b-code-interpreter is not installed. "
                "Install with: pip install e2b-code-interpreter"
            ) from exc

        self._file_type = FileType
        create_kwargs: Dict[str, object] = {
            "timeout": options.timeout_sec,
            "allow_internet_access": options.allow_internet_access,
            "api_key": options.api_key,
        }
        if options.template:
            create_kwargs["template"] = options.template

        self._sandbox = E2BSandbox.create(**create_kwargs)
        # E2B sandboxes run as an unprivileged user; top-level /workspace is not writable.
        self._remote_root = "/home/user/workspace"

        self._run_remote(f"mkdir -p {shell_quote([self._remote_root])}", cwd=None)
        self._sync_local_to_remote()

    def name(self) -> str:
        return self._sandbox_name

    def exec(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout_sec: int = 600,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        cmd_str = shell_quote(cmd)
        if env:
            remote_env = self._translate_env(env)
            env_pairs = " ".join(f"{k}={shlex.quote(v)}" for k, v in remote_env.items())
            cmd_str = f"env {env_pairs} {cmd_str}"
        local_cwd = self.root if cwd is None else self._resolve_local_path(cwd)
        remote_cwd = self._to_remote_dir(local_cwd)

        self._sync_local_to_remote()
        result = self._sandbox.commands.run(
            cmd_str,
            cwd=remote_cwd,
            timeout=float(timeout_sec),
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
        self._sandbox.files.write(remote_path, content)

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
            self._sandbox.kill()
        except Exception:  # noqa: BLE001
            pass

    def _sync_local_to_remote(self) -> None:
        executable_paths: List[str] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                continue
            rel = path.relative_to(self.root)
            remote_path = f"{self._remote_root}/{rel.as_posix()}"

            if path.is_dir():
                self._sandbox.files.make_dir(remote_path)
                continue

            if not path.is_file():
                continue

            data = path.read_bytes()
            self._sandbox.files.write(remote_path, data)
            if path.stat().st_mode & 0o111:
                executable_paths.append(remote_path)

        for remote_path in executable_paths:
            self._run_remote(f"chmod +x {shell_quote([remote_path])}", cwd=self._remote_root)

    def _sync_remote_to_local(self) -> None:
        entries = self._sandbox.files.list(self._remote_root, depth=128)

        remote_dirs = set()
        remote_files = set()

        for entry in entries:
            entry_path = str(entry.path)
            if not entry_path.startswith(self._remote_root + "/"):
                continue
            rel = Path(entry_path[len(self._remote_root) + 1 :])
            if not rel.parts:
                continue

            entry_type = entry.type
            is_dir = entry_type == self._file_type.DIR or str(entry_type) == "dir"
            is_file = entry_type == self._file_type.FILE or str(entry_type) == "file"

            if is_dir:
                remote_dirs.add(rel)
            elif is_file:
                remote_files.add(rel)

        # Remove local files that no longer exist remotely.
        for path in sorted(self.root.rglob("*"), reverse=True):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(self.root)
            if rel not in remote_files:
                path.unlink(missing_ok=True)

        for rel in sorted(remote_dirs):
            (self.root / rel).mkdir(parents=True, exist_ok=True)

        for rel in sorted(remote_files):
            local_path = self.root / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            data = self._sandbox.files.read(f"{self._remote_root}/{rel.as_posix()}", format="bytes")
            local_path.write_bytes(bytes(data))

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
        result = self._sandbox.commands.run(command, cwd=cwd, timeout=120)
        if result.exit_code != 0 or result.error:
            raise RuntimeError(f"e2b command failed: {command}; stderr={result.stderr}; error={result.error}")

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
