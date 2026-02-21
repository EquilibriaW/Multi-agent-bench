"""
loopbench.reproducibility

Durable run snapshots for debugging and reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import shlex
import sys
from typing import Any, Dict, Mapping

from .config import AgentsConfig, RuntimeConfig
from .schema import TaskPack
from .shell import run_command

_SECRET_ENV_KEY_RE = re.compile(r"(key|token|secret|password|passwd|credential|auth)", re.IGNORECASE)


def write_run_inputs_snapshot(
    *,
    run_dir: str | Path,
    run_id: str,
    runtime_config_path: str | Path,
    runtime_cfg: RuntimeConfig,
    agents_config_path: str | Path,
    agents_cfg: AgentsConfig,
    task: TaskPack,
    base_commit: str,
) -> None:
    run_dir_path = Path(run_dir).resolve()
    inputs_dir = run_dir_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    runtime_source = Path(runtime_config_path).resolve()
    agents_source = Path(agents_config_path).resolve()
    task_yaml_path = Path(task.task_yaml).resolve()
    task_root = Path(task.root_dir).resolve()
    task_base_repo = task_root / task.workspace.base_repo

    runtime_payload = {
        "source_path": str(runtime_source),
        "source_sha256": _sha256_if_file(runtime_source),
        "resolved": runtime_cfg.model_dump(),
    }
    _write_json(inputs_dir / "runtime_config.json", runtime_payload)

    agents_payload = {
        "source_path": str(agents_source),
        "source_sha256": _sha256_if_file(agents_source),
        "resolved": _sanitize_agents_for_artifacts(agents_cfg.model_dump()),
        "role_driver_fingerprints": _role_driver_fingerprints(agents_cfg),
    }
    _write_json(inputs_dir / "agents_config.json", agents_payload)

    task_payload = {
        "task_id": task.task_id,
        "task_yaml_path": str(task_yaml_path),
        "task_yaml_sha256": _sha256_if_file(task_yaml_path),
        "task_root": str(task_root),
        "task_pack": task.model_dump(),
        "workspace_base_repo_path": str(task_base_repo.resolve()),
        "workspace_base_repo_sha256": _sha256_if_file(task_base_repo),
        "workspace_base_repo_head": _git_head_if_repo(task_base_repo),
        "base_commit": base_commit,
    }
    _write_json(inputs_dir / "task_snapshot.json", task_payload)

    environment_payload = {
        "run_id": run_id,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_version": _binary_version(["git", "--version"]),
        "docker_version": _binary_version(["docker", "--version"]),
    }
    _write_json(inputs_dir / "environment.json", environment_payload)


def snapshot_role_repo_state(
    *,
    run_dir: str | Path,
    role_paths: Mapping[str, Path],
    base_commit: str,
) -> Dict[str, str]:
    run_dir_path = Path(run_dir).resolve()
    repo_state_dir = run_dir_path / "repo_state"
    repo_state_dir.mkdir(parents=True, exist_ok=True)
    (repo_state_dir / "base_commit.txt").write_text(f"{base_commit}\n", encoding="utf-8")

    role_heads: Dict[str, str] = {}
    for role, role_path in sorted(role_paths.items()):
        role_dir = repo_state_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)

        head = _git_query(role_path, ["rev-parse", "HEAD"])
        if head:
            role_heads[role] = head
            (role_dir / "head.txt").write_text(f"{head}\n", encoding="utf-8")
        else:
            (role_dir / "head.txt").write_text("", encoding="utf-8")

        _write_git_capture(role_path, ["status", "--short", "--branch"], role_dir / "status.txt")
        _write_git_capture(
            role_path,
            ["log", "--reverse", "--oneline", f"{base_commit}..HEAD"],
            role_dir / "log_from_base.txt",
        )
        _write_git_capture(
            role_path,
            ["diff", "--name-status", f"{base_commit}..HEAD"],
            role_dir / "changed_files.txt",
        )
        _write_git_capture(
            role_path,
            ["diff", "--binary", base_commit, "HEAD"],
            role_dir / "diff_from_base.patch",
        )
        _write_git_bundle(role_path, role_dir / "role.bundle")

    return role_heads


def _write_git_capture(repo_path: Path, args: list[str], out_path: Path) -> None:
    result = run_command(["git", "-C", str(repo_path), *args], timeout_sec=120)
    if result.ok:
        out_path.write_text(result.stdout, encoding="utf-8")
        return
    out_path.write_text(
        "\n".join(
            [
                f"# command: git -C {repo_path} {' '.join(args)}",
                f"# exit_code: {result.exit_code}",
                "",
                "STDOUT",
                result.stdout,
                "",
                "STDERR",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )


def _sanitize_agents_for_artifacts(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    roles = out.get("roles")
    if not isinstance(roles, list):
        return out

    sanitized_roles = []
    for role in roles:
        if not isinstance(role, dict):
            sanitized_roles.append(role)
            continue
        role_copy = dict(role)
        env = role_copy.get("env")
        if isinstance(env, dict):
            role_copy["env"] = {
                str(k): _sanitize_env_value(str(k), v)
                for k, v in env.items()
            }
        sanitized_roles.append(role_copy)
    out["roles"] = sanitized_roles
    return out


def _write_git_bundle(repo_path: Path, out_path: Path) -> None:
    result = run_command(
        ["git", "-C", str(repo_path), "bundle", "create", str(out_path), "HEAD"],
        timeout_sec=120,
    )
    if result.ok:
        return
    out_path.with_suffix(".bundle.error.txt").write_text(
        "\n".join(
            [
                f"# command: git -C {repo_path} bundle create {out_path} HEAD",
                f"# exit_code: {result.exit_code}",
                "",
                "STDOUT",
                result.stdout,
                "",
                "STDERR",
                result.stderr,
            ]
        ),
        encoding="utf-8",
    )


def _sanitize_env_value(key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if _SECRET_ENV_KEY_RE.search(key):
        return {
            "redacted": True,
            "sha256": _sha256_text(value),
            "length": len(value),
        }
    return value


def _role_driver_fingerprints(agents_cfg: AgentsConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in agents_cfg.roles:
        row: dict[str, Any] = {
            "role": role.name,
            "driver": role.driver,
            "command": role.command,
            "model": role.model,
        }

        first = _resolve_first_command_path(role.command)
        if first is not None:
            row["command_executable_path"] = str(first)
            row["command_executable_sha256"] = _sha256_if_file(first)
        rows.append(row)
    return rows


def _resolve_first_command_path(command: str | None) -> Path | None:
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    if not parts:
        return None
    first = Path(parts[0])
    if first.is_absolute() and first.exists() and first.is_file():
        return first.resolve()
    return None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_if_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_if_repo(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    if not (path / ".git").exists():
        return None
    return _git_query(path, ["rev-parse", "HEAD"])


def _git_query(repo_path: Path, args: list[str]) -> str | None:
    result = run_command(["git", "-C", str(repo_path), *args], timeout_sec=30)
    if not result.ok:
        return None
    value = result.stdout.strip()
    return value or None


def _binary_version(cmd: list[str]) -> str | None:
    if not cmd:
        return None
    if shutil.which(cmd[0]) is None:
        return None
    result = run_command(cmd, timeout_sec=10)
    if not result.ok:
        return None
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    return line[0]


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
