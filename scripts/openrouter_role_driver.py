#!/usr/bin/env python3
"""
OpenRouter-backed role driver for LoopBench team experiments.

Expected env from harness:
- LB_ROLE
- LB_PHASE
- LB_WORKTREE
- LB_CONTEXT_JSON
- LB_OUTPUT_JSON
- LB_MODEL

Optional env:
- OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY
- OPENROUTER_API_KEY_ENV (env var name containing the API key)
- OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
- OPENROUTER_HTTP_REFERER
- OPENROUTER_APP_TITLE
- OPENROUTER_TEMPERATURE (default: 0.2)
- OPENROUTER_MAX_TOKENS (default: 4096)
- LOOPBENCH_SANDBOX_BACKEND (injected by harness; used for default policy)
- LOOPBENCH_MAX_COMMANDS (default: 8 in e2b, 3 otherwise)
- LOOPBENCH_COMMAND_TIMEOUT_SEC (default: 1200 in e2b, 180 otherwise)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_MODEL = "moonshotai/kimi-k2.5"
MAX_FILES_IN_CONTEXT = 24
MAX_FILE_CHARS = 7000
MAX_TOTAL_SNIPPET_CHARS = 70000


TEAM_BASE_PROMPT = (
    "You are part of a 3-agent coding team in a benchmark harness. "
    "Return JSON only. No markdown fences, no prose outside JSON. "
    "Prefer minimal, correct edits. Keep files syntactically valid."
)

PLANNER_ROLE_PROMPT = (
    "Role: planner_reviewer. You own decomposition, review guidance, and final coherence. "
    "In bootstrap, produce a concrete plan and subtask split for coder_a and coder_b. "
    "In review/finalize, focus on merge readiness, validation feedback triage, and clear status notes."
)

GENERIC_CODER_ROLE_PROMPT = (
    "Role: coder (applies equally to coder_a and coder_b). "
    "You execute only assigned work from planner-reviewer messages/assignment payload. "
    "Do not invent a new work split. If assignment is missing/ambiguous, report blocker in notes."
)


def main() -> int:
    role = _required_env("LB_ROLE")
    phase = _required_env("LB_PHASE")
    worktree = Path(_required_env("LB_WORKTREE")).resolve()
    context_path = Path(_required_env("LB_CONTEXT_JSON")).resolve()
    output_path = Path(_required_env("LB_OUTPUT_JSON")).resolve()
    model = os.environ.get("LB_MODEL") or DEFAULT_MODEL

    context = _read_json_file(context_path)
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is missing. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY, "
            "or set OPENROUTER_API_KEY_ENV to the env-var name."
        )

    repo_ctx = _collect_repo_context(worktree=worktree, context=context)
    payload = _build_payload(
        role=role,
        phase=phase,
        model=model,
        context=context,
        repo_ctx=repo_ctx,
    )
    _write_json_file(_role_trace_path(output_path, "openrouter_request", ".json"), payload)

    reply_text = _call_openrouter(payload=payload, api_key=api_key)
    _write_text_file(_role_trace_path(output_path, "openrouter_response", ".txt"), reply_text)
    parsed = _parse_json_object(reply_text)

    output: Dict[str, Any] = {
        "status": "completed",
        "role": role,
        "phase": phase,
        "model": model,
        "openrouter_raw_chars": len(reply_text),
        "coordination_phase": context.get("coordination_phase"),
    }

    if phase == "bootstrap":
        plan_md = parsed.get("plan_markdown")
        subtasks = parsed.get("subtasks")
        if not isinstance(plan_md, str) or not plan_md.strip():
            plan_md = _default_plan_markdown(task_id=str(context.get("task_id") or "task"))
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = _default_subtasks(context=context)
        output["plan_markdown"] = plan_md
        output["subtasks"] = subtasks
        output["summary"] = _safe_text(parsed.get("summary"), fallback="bootstrap plan created")
        _write_json_file(output_path, output)
        return 0

    file_updates = _normalize_file_updates(parsed.get("file_updates"))
    applied_paths = _apply_file_updates(worktree=worktree, file_updates=file_updates)

    command_policy = _get_command_policy()
    command_results = []
    command_trace = []
    for cmd in _normalize_commands(parsed.get("run_commands"))[: command_policy["max_commands"]]:
        result = _run_shell(cmd, cwd=worktree, timeout_sec=command_policy["command_timeout_sec"])
        command_trace.append(
            {
                "cmd": cmd,
                "ok": result["ok"],
                "exit_code": result["exit_code"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }
        )
        command_results.append(
            {
                "cmd": cmd,
                "ok": result["ok"],
                "exit_code": result["exit_code"],
                "stdout_tail": result["stdout"][-1200:],
                "stderr_tail": result["stderr"][-1200:],
            }
        )

    commit_message = _safe_text(
        parsed.get("commit_message"),
        fallback=f"{role}: {phase} update",
    )
    commit_sha = _commit_if_dirty(worktree=worktree, commit_message=commit_message)

    output.update(
        {
            "summary": _safe_text(parsed.get("summary"), fallback=f"{role} {phase} complete"),
            "notes": _safe_text(parsed.get("notes"), fallback=""),
            "file_updates_attempted": len(file_updates),
            "file_updates_applied": len(applied_paths),
            "applied_paths": applied_paths,
            "commands_run": command_results,
            "command_policy_max_commands": command_policy["max_commands"],
            "command_policy_timeout_sec": command_policy["command_timeout_sec"],
            "commit": commit_sha,
            "changed": bool(commit_sha),
        }
    )
    if command_trace:
        trace_path = _role_trace_path(output_path, "command_trace", ".json")
        _write_json_file(trace_path, {"commands": command_trace})
        output["command_trace_path"] = str(trace_path)

    _write_json_file(output_path, output)
    return 0


def _build_payload(
    *,
    role: str,
    phase: str,
    model: str,
    context: Dict[str, Any],
    repo_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    system_prompt = _build_system_prompt(role=role, phase=phase)

    user_prompt = {
        "role": role,
        "phase": phase,
        "task_id": context.get("task_id"),
        "assignment": context.get("assignment"),
        "claimed_task": context.get("claimed_task"),
        "public_validate_stderr": context.get("public_validate_stderr"),
        "public_validate_stdout": context.get("public_validate_stdout"),
        "planner_summary": context.get("planner_summary"),
        "repo_context": repo_ctx,
        "required_json_schema": _schema_hint_for_phase(phase),
    }

    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096"))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    return payload


def _build_system_prompt(*, role: str, phase: str) -> str:
    role_prompt = PLANNER_ROLE_PROMPT if role == "planner_reviewer" else GENERIC_CODER_ROLE_PROMPT
    phase_hint = f"Current phase: {phase}."
    return " ".join([TEAM_BASE_PROMPT, role_prompt, phase_hint])


def _schema_hint_for_phase(phase: str) -> Dict[str, Any]:
    if phase == "bootstrap":
        return {
            "status": "completed",
            "summary": "short summary",
            "plan_markdown": "# Plan ...",
            "subtasks": [
                {
                    "id": "S1",
                    "role": "coder_a",
                    "title": "subtask title",
                    "paths": ["path/or/dir"],
                    "acceptance": "done when ...",
                },
                {
                    "id": "S2",
                    "role": "coder_b",
                    "title": "subtask title",
                    "paths": ["path/or/dir"],
                    "acceptance": "done when ...",
                },
            ],
        }

    return {
        "status": "completed",
        "summary": "what changed",
        "notes": "optional caveats",
        "file_updates": [
            {
                "path": "relative/path.ext",
                "content": "full replacement file content",
            }
        ],
        "run_commands": [
            "pytest -q",
            "npm test -- --runInBand",
        ],
        "commit_message": "concise commit message",
    }


def _collect_repo_context(*, worktree: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    task_readme = _safe_read_text(worktree / "public" / "README.task.md", max_chars=8000)
    tracked = _git_ls_files(worktree)
    candidate_paths = _select_candidate_paths(worktree=worktree, tracked_files=tracked, context=context)

    snippets = []
    total = 0
    for rel in candidate_paths:
        if len(snippets) >= MAX_FILES_IN_CONTEXT or total >= MAX_TOTAL_SNIPPET_CHARS:
            break
        content = _safe_read_text(worktree / rel, max_chars=MAX_FILE_CHARS)
        if content is None:
            continue
        snippets.append({"path": rel, "content": content})
        total += len(content)

    return {
        "readme_task_md": task_readme,
        "tracked_files_count": len(tracked),
        "tracked_files_head": tracked[:200],
        "file_snippets": snippets,
    }


def _select_candidate_paths(
    *,
    worktree: Path,
    tracked_files: List[str],
    context: Dict[str, Any],
) -> List[str]:
    scored: Dict[str, int] = {}
    tracked_set = set(tracked_files)

    for idx, path in enumerate(tracked_files):
        if idx < 40:
            scored[path] = max(scored.get(path, 0), 1)
        lower = path.lower()
        if lower.endswith((".rb", ".js", ".ts", ".py", ".go", ".java", ".rs", ".php", ".yaml", ".yml")):
            scored[path] = max(scored.get(path, 0), 2)
        if any(token in lower for token in ("app", "index", "main", "route", "link", "controller", "server")):
            scored[path] = max(scored.get(path, 0), 3)

    for p in _iter_assignment_paths(context):
        if p in tracked_set:
            scored[p] = max(scored.get(p, 0), 8)
            continue
        prefix = p.rstrip("/") + "/"
        matched = [f for f in tracked_files if f.startswith(prefix)]
        for m in matched[:8]:
            scored[m] = max(scored.get(m, 0), 7)

    # Lightweight grep signal for likely edit locations.
    if shutil.which("rg"):
        grep_cmd = [
            "rg",
            "-n",
            "--max-count",
            "2",
            "TODO|links|opensearch|secret bot|hello world|route|Link",
        ]
        grep = _run_cmd(grep_cmd, cwd=worktree, timeout_sec=15)
        if grep["ok"]:
            for line in grep["stdout"].splitlines():
                m = re.match(r"([^:]+):\d+:", line)
                if not m:
                    continue
                path = m.group(1)
                if path in tracked_set:
                    scored[path] = max(scored.get(path, 0), 10)

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [path for path, _ in ranked[:MAX_FILES_IN_CONTEXT]]


def _iter_assignment_paths(context: Dict[str, Any]) -> Iterable[str]:
    assignment = context.get("assignment")
    if not isinstance(assignment, list):
        return []

    out: List[str] = []
    for item in assignment:
        if not isinstance(item, dict):
            continue
        paths = item.get("paths")
        if isinstance(paths, list):
            for p in paths:
                if isinstance(p, str) and p.strip():
                    out.append(p.strip())
    return out


def _normalize_file_updates(raw: Any) -> List[Dict[str, str]]:
    if not isinstance(raw, list):
        return []
    updates: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content = item.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        path = path.strip()
        if not path:
            continue
        updates.append({"path": path, "content": content})
    return updates


def _normalize_commands(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def _apply_file_updates(*, worktree: Path, file_updates: List[Dict[str, str]]) -> List[str]:
    applied = []
    for update in file_updates:
        path = update["path"]
        dest = _resolve_within_root(worktree, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(update["content"], encoding="utf-8")
        applied.append(path)
    return applied


def _resolve_within_root(root: Path, raw_path: str) -> Path:
    path = (root / raw_path).resolve()
    if path == root or root in path.parents:
        return path
    raise RuntimeError(f"path escapes worktree root: {raw_path}")


def _commit_if_dirty(*, worktree: Path, commit_message: str) -> str | None:
    status = _run_cmd(["git", "status", "--porcelain"], cwd=worktree, timeout_sec=30)
    if not status["ok"] or not status["stdout"].strip():
        return None

    add = _run_cmd(["git", "add", "-A"], cwd=worktree, timeout_sec=30)
    if not add["ok"]:
        return None

    # Never commit harness-staged task assets.
    _run_cmd(["git", "reset", "-q", "HEAD", "--", ".loopbench"], cwd=worktree, timeout_sec=15)
    public_link = worktree / "public"
    if public_link.is_symlink():
        try:
            target = public_link.resolve()
        except Exception:  # noqa: BLE001
            target = None
        if target and ".loopbench" in target.parts:
            _run_cmd(["git", "reset", "-q", "HEAD", "--", "public"], cwd=worktree, timeout_sec=15)

    staged = _run_cmd(["git", "diff", "--cached", "--name-only"], cwd=worktree, timeout_sec=20)
    if not staged["ok"] or not staged["stdout"].strip():
        return None

    commit = _run_cmd(["git", "commit", "-m", commit_message], cwd=worktree, timeout_sec=45)
    if not commit["ok"]:
        return None

    rev = _run_cmd(["git", "rev-parse", "HEAD"], cwd=worktree, timeout_sec=15)
    if not rev["ok"]:
        return None
    return rev["stdout"].strip() or None


def _get_command_policy() -> Dict[str, Any]:
    if _is_e2b_backend():
        default_max_commands = 8
        default_timeout_sec = 1200
    else:
        default_max_commands = 3
        default_timeout_sec = 180

    max_commands = _read_int_env(
        name="LOOPBENCH_MAX_COMMANDS",
        default=default_max_commands,
        min_value=1,
        max_value=20,
    )
    command_timeout_sec = _read_int_env(
        name="LOOPBENCH_COMMAND_TIMEOUT_SEC",
        default=default_timeout_sec,
        min_value=30,
        max_value=3600,
    )

    return {
        "max_commands": max_commands,
        "command_timeout_sec": command_timeout_sec,
    }


def _is_e2b_backend() -> bool:
    backend = os.environ.get("LOOPBENCH_SANDBOX_BACKEND", "").strip().lower()
    return backend == "e2b_firecracker"


def _read_int_env(*, name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _default_plan_markdown(task_id: str) -> str:
    return (
        f"# Plan for {task_id}\n\n"
        "1. Identify affected files and required endpoint behavior.\n"
        "2. coder_a implements core runtime logic changes.\n"
        "3. coder_b handles env/test/integration adjustments.\n"
        "4. planner_reviewer merges commits and runs validation.\n"
    )


def _default_subtasks(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": "S1",
            "role": "coder_a",
            "title": "Core implementation",
            "paths": ["."],
            "acceptance": "Core behavior implemented according to task prompt.",
        },
        {
            "id": "S2",
            "role": "coder_b",
            "title": "Validation and integration",
            "paths": ["tests", "Dockerfile", "."],
            "acceptance": "Validation or environment adjustments aligned with implementation.",
        },
    ]


def _resolve_openrouter_api_key() -> str:
    key_env = os.environ.get("OPENROUTER_API_KEY_ENV") or os.environ.get("OPEN_ROUTER_API_KEY_ENV")
    if key_env:
        value = os.environ.get(key_env)
        if value:
            return value

    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _call_openrouter(*, payload: Dict[str, Any], api_key: str) -> str:
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    app_title = os.environ.get("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title

    body = json.dumps(payload).encode("utf-8")
    err_text = ""

    for attempt in range(1, 4):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                response = json.loads(resp.read().decode("utf-8", errors="replace"))
                content = response["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    return "\n".join(parts).strip()
                if isinstance(content, str):
                    return content.strip()
                return json.dumps(content)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            err_text = f"http {exc.code}: {raw}"
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
        time.sleep(attempt)

    raise RuntimeError(f"openrouter request failed after retries: {err_text}")


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    return {}


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _read_json_file(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"context JSON must be an object: {path}")
    return data


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _role_trace_path(output_path: Path, suffix: str, extension: str) -> Path:
    stem = output_path.stem
    if stem.endswith("_output"):
        stem = stem[:-7]
    return output_path.with_name(f"{stem}_{suffix}{extension}")


def _safe_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _safe_read_text(path: Path, *, max_chars: int) -> str | None:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    if len(data) > max_chars:
        return data[:max_chars] + "\n...[truncated]..."
    return data


def _git_ls_files(worktree: Path) -> List[str]:
    result = _run_cmd(["git", "ls-files"], cwd=worktree, timeout_sec=30)
    if not result["ok"]:
        return []
    return [line.strip() for line in result["stdout"].splitlines() if line.strip()]


def _run_shell(command: str, *, cwd: Path, timeout_sec: int) -> Dict[str, Any]:
    return _run_cmd(["bash", "-lc", command], cwd=cwd, timeout_sec=timeout_sec)


def _run_cmd(cmd: List[str], *, cwd: Path, timeout_sec: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") + f"\ncommand timed out after {timeout_sec}s",
            "exit_code": 124,
        }


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        lb_output = os.environ.get("LB_OUTPUT_JSON")
        if lb_output:
            payload = {
                "status": "error",
                "error": str(exc),
                "role": os.environ.get("LB_ROLE"),
                "phase": os.environ.get("LB_PHASE"),
            }
            Path(lb_output).parent.mkdir(parents=True, exist_ok=True)
            Path(lb_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
