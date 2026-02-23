"""
loopbench.role_actions

Typed role-action intent parsing and assignment-path helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class FileUpdateAction:
    path: str
    content: str


@dataclass(frozen=True)
class RoleActionPlan:
    summary: str
    notes: str
    patch_text: str
    file_updates: List[FileUpdateAction] = field(default_factory=list)
    run_commands: List[str] = field(default_factory=list)
    commit_message: str = ""
    planner_non_mutating: bool = False
    command_policy_max_commands: int = 8
    command_policy_timeout_sec: int = 180


def extract_role_action_plan(
    *,
    role: str,
    phase: str,
    output: Dict[str, Any],
) -> RoleActionPlan:
    command_max = _to_int(output.get("command_policy_max_commands"), default=8, min_value=1, max_value=20)
    timeout_sec = _to_int(output.get("command_policy_timeout_sec"), default=180, min_value=30, max_value=3600)
    patch_text = _safe_text(output.get("intent_patch"), fallback="")
    if not patch_text:
        patch_text = _safe_text(output.get("patch"), fallback="")

    run_commands = _normalize_commands(output.get("intent_run_commands"))
    if not run_commands:
        run_commands = _normalize_commands(output.get("run_commands"))
    run_commands = run_commands[:command_max]

    file_updates = _normalize_file_updates(output.get("intent_file_updates"))
    if not file_updates:
        file_updates = _normalize_file_updates(output.get("file_updates"))

    commit_message = _safe_text(output.get("intent_commit_message"), fallback="")
    if not commit_message:
        commit_message = _safe_text(output.get("commit_message"), fallback=f"{role}: {phase} update")

    return RoleActionPlan(
        summary=_safe_text(output.get("summary"), fallback=f"{role} {phase} complete"),
        notes=_safe_text(output.get("notes"), fallback=""),
        patch_text=patch_text,
        file_updates=file_updates,
        run_commands=run_commands,
        commit_message=commit_message,
        planner_non_mutating=bool(output.get("planner_non_mutating")),
        command_policy_max_commands=command_max,
        command_policy_timeout_sec=timeout_sec,
    )


def iter_assignment_paths(context: Dict[str, Any]) -> Iterable[str]:
    assignment = context.get("assignment")
    if not isinstance(assignment, list):
        return []
    out: List[str] = []
    for item in assignment:
        if not isinstance(item, dict):
            continue
        paths = item.get("paths")
        if not isinstance(paths, list):
            continue
        for raw in paths:
            if not isinstance(raw, str):
                continue
            path = raw.strip()
            if path and path not in out:
                out.append(path)
    return out


def assignment_deviations(*, changed_paths: List[str], assignment_paths: List[str]) -> List[str]:
    if not assignment_paths:
        return []
    out: List[str] = []
    for path in changed_paths:
        if path_allowed_by_assignment(path=path, allowed_paths=assignment_paths):
            continue
        if path not in out:
            out.append(path)
    return out


def path_allowed_by_assignment(*, path: str, allowed_paths: List[str]) -> bool:
    normalized = _normalize_repo_relative_path(path)
    if not normalized or normalized == ".":
        return False
    for raw_root in allowed_paths:
        raw_root = raw_root.strip()
        if raw_root in {"", "."}:
            return True
        root = _normalize_repo_relative_path(raw_root)
        if not root:
            continue
        if root == ".":
            return True
        if normalized == root or normalized.startswith(f"{root}/"):
            return True
    return False


def paths_from_unified_patch(patch_text: str) -> List[str]:
    paths: List[str] = []
    for raw_line in patch_text.splitlines():
        line = raw_line.strip()
        if not line.startswith(("+++ ", "--- ")):
            continue
        path = line[4:].strip()
        normalized = _normalize_repo_relative_path(path)
        if not normalized or normalized == ".":
            continue
        if normalized not in paths:
            paths.append(normalized)
    return paths


def _normalize_file_updates(raw: Any) -> List[FileUpdateAction]:
    if not isinstance(raw, list):
        return []
    out: List[FileUpdateAction] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path or path.startswith("/") or ".." in path.split("/"):
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        out.append(FileUpdateAction(path=path, content=content))
    return out


def _normalize_commands(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        cmd = item.strip()
        if cmd and cmd not in out:
            out.append(cmd)
    return out


def _safe_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _to_int(raw: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(raw)
    except Exception:  # noqa: BLE001
        value = default
    return max(min_value, min(max_value, value))


def _normalize_repo_relative_path(raw: str) -> str | None:
    path = str(raw or "").strip().replace("\\", "/")
    if not path:
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    if path == "/dev/null":
        return None
    if path.startswith("/"):
        return None

    parts: List[str] = []
    for token in path.split("/"):
        token = token.strip()
        if token in {"", "."}:
            continue
        if token == "..":
            return None
        parts.append(token)

    if not parts:
        return "."
    return "/".join(parts)
