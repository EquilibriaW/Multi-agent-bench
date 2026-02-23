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
- OPENROUTER_REASONING_ENABLED (optional bool)
- OPENROUTER_REASONING_EFFORT (optional string, e.g. none/low/medium/high)
- OPENROUTER_REASONING_MAX_TOKENS (optional int >= 0)
- OPENROUTER_REASONING_EXCLUDE (optional bool)
- OPENROUTER_HTTP_TIMEOUT_SEC (default: 90)
- OPENROUTER_HTTP_RETRIES (default: 2)
- OPENROUTER_STRUCTURED_RETRIES (default: 2)
- LOOPBENCH_SANDBOX_BACKEND (injected by harness; used for default policy)
- LOOPBENCH_MAX_COMMANDS (default: 8 in e2b, 3 otherwise)
- LOOPBENCH_COMMAND_TIMEOUT_SEC (default: 1200 in e2b, 180 otherwise)
- LOOPBENCH_PLANNER_NON_MUTATING_PHASES (optional comma-separated phases; e.g. review,finalize)
"""
from __future__ import annotations

import json
import os
import re
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
DEFAULT_STRUCTURED_REPLY_ATTEMPTS = 2
DEFAULT_OPENROUTER_HTTP_RETRIES = 2
DEFAULT_OPENROUTER_HTTP_TIMEOUT_SEC = 90
MIN_STRUCTURED_REPLY_CHARS = 24


TEAM_BASE_PROMPT = (
    "You are part of a 3-agent coding team in a benchmark harness. "
    "Prefer minimal, correct edits. Keep files syntactically valid."
)

PLANNER_ROLE_PROMPT = (
    "Role: planner_reviewer. You own decomposition, review guidance, and final coherence. "
    "Do not execute commands yourself; emit tool intents (run_commands/file_updates/patch) for the harness to execute. "
    "In bootstrap, produce a concrete plan and subtask split for coder_a and coder_b. "
    "In review (review_stage=select), inspect coder diffs and choose commits to merge. "
    "Before nominating a commit for merge, run review_diff_tool show/files for that commit sha. "
    "Return merge_commits as role->commit list for only the coder commits you nominate for merge. "
    "Use candidate_merge_commits and coder_commits from the prompt as your merge source of truth. "
    "Use review_diff_tool commands to inspect full commit patches when needed (list/files/show). "
    "Use commit hashes from coder_commits; do not use symbolic tokens like HEAD. "
    "In review_verify (review_stage=verify), run dynamic checks on the integrated candidate and decide whether rework is needed. "
    "If any verification command fails, set request_rework=true and explain next fixes. "
    "Set request_rework=true and provide coder_feedback when coders must revise. "
    "In finalize, focus on final coherence and ship readiness."
)

REFLECTION_PROMPT = (
    "Role: reflection analyst. You analyze the execution trace of the most recent review round "
    "and produce structured knowledge for the next round. Your output is DIRECTIVE — it tells "
    "agents exactly what to do differently, not what happened. "
    "Rules: "
    "1. Write a concise directive (200-500 chars) with embedded fix instructions — like a lint "
    "error with a fix suggestion. Example: 'coder_a: add flask to requirements.txt before "
    "running pytest. coder_b: verify app/routes.py imports match installed packages.' "
    "2. OVERWRITE each knowledge surface completely. Do not append to previous content. "
    "3. Explicitly list which previous insights are superseded in the 'superseded' array. "
    "4. Focus on actionable patterns, not raw data. What should change, not what happened. "
    "5. Keep task_understanding updated with refined understanding of what the task requires. "
    "6. In failure_patterns, only keep CURRENT failures. Remove resolved ones (backpressure). "
    "7. In workflow_insights, capture strategic observations about what's working vs not."
)

GENERIC_CODER_ROLE_PROMPT = (
    "Role: coder (applies equally to coder_a and coder_b). "
    "Do not execute commands yourself; emit tool intents (run_commands/file_updates/patch) for the harness to execute. "
    "You execute only assigned work from planner-reviewer messages/assignment payload. "
    "Do not invent a new work split. If assignment is missing/ambiguous, report blocker in notes. "
    "Prefer normal engineering output: concise summary plus unified diff patch blocks (```diff ... ```)."
)

FILE_UPDATE_SCHEMA = {
    "path": "relative/path.ext",
    "content": "full replacement file content",
}
CODER_FEEDBACK_SCHEMA = {
    "coder_a": "what to revise",
    "coder_b": "what to revise",
}
DEFAULT_BOOTSTRAP_SUBTASKS = [
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

    require_structured = _require_structured_reply(role=role, phase=phase)
    command_policy: Dict[str, Any] | None = None
    if phase != "bootstrap":
        command_policy = _get_command_policy()

    request_result = _request_structured_reply(
        payload=payload,
        api_key=api_key,
        phase=phase,
        require_structured=require_structured,
    )
    reply_text = request_result["reply_text"]
    parsed = request_result["parsed"]
    if role != "planner_reviewer":
        parsed = _coerce_coder_reply(parsed=parsed, reply_text=reply_text)
    _write_text_file(_role_trace_path(output_path, "openrouter_response", ".txt"), reply_text)
    if request_result["attempts"]:
        _write_json_file(
            _role_trace_path(output_path, "openrouter_attempts", ".json"),
            {
                "selected_attempt": request_result.get("selected_attempt"),
                "attempts": request_result["attempts"],
            },
        )

    output: Dict[str, Any] = {
        "status": "completed",
        "role": role,
        "phase": phase,
        "model": model,
        "openrouter_raw_chars": len(reply_text),
        "openrouter_attempt_count": len(request_result["attempts"]),
        "openrouter_selected_attempt": request_result.get("selected_attempt"),
        "openrouter_structured_valid": bool(request_result["structured_valid"]),
        "openrouter_usage": request_result["usage"],
        "openrouter_turn_count": 1,
        "coordination_phase": context.get("coordination_phase"),
    }

    if phase == "reflect":
        output.update(
            {
                "directive": _safe_text(parsed.get("directive"), fallback=""),
                "task_understanding": _safe_text(parsed.get("task_understanding"), fallback=""),
                "failure_patterns": _safe_text(parsed.get("failure_patterns"), fallback=""),
                "workflow_insights": _safe_text(parsed.get("workflow_insights"), fallback=""),
                "superseded": parsed.get("superseded") if isinstance(parsed.get("superseded"), list) else [],
                "summary": _safe_text(parsed.get("directive"), fallback="reflection complete"),
            }
        )
        _write_json_file(output_path, output)
        return 0

    if phase == "bootstrap":
        plan_md = parsed.get("plan_markdown")
        subtasks = parsed.get("subtasks")
        if not isinstance(plan_md, str) or not plan_md.strip():
            task_id = str(context.get("task_id") or "task")
            plan_md = (
                f"# Plan for {task_id}\n\n"
                "1. Identify affected files and required endpoint behavior.\n"
                "2. coder_a implements core runtime logic changes.\n"
                "3. coder_b handles env/test/integration adjustments.\n"
                "4. planner_reviewer merges commits and runs validation.\n"
            )
        if not isinstance(subtasks, list) or not subtasks:
            subtasks = DEFAULT_BOOTSTRAP_SUBTASKS
        output["plan_markdown"] = plan_md
        output["subtasks"] = subtasks
        output["summary"] = _safe_text(parsed.get("summary"), fallback="bootstrap plan created")
        _write_json_file(output_path, output)
        return 0

    command_policy = command_policy or _get_command_policy()
    requested_file_updates = _normalize_file_updates(parsed.get("file_updates"))
    requested_patch = _safe_text(parsed.get("patch"), fallback="")
    requested_commands = _normalize_commands(parsed.get("run_commands"))[: command_policy["max_commands"]]
    planner_non_mutating = False
    if role == "planner_reviewer":
        raw = (os.environ.get("LOOPBENCH_PLANNER_NON_MUTATING_PHASES") or "").strip()
        if raw:
            phases = {token.strip().lower() for token in raw.split(",") if token.strip()}
            phase_name = phase.strip().lower()
            planner_non_mutating = "all" in phases or phase_name in phases
    suppress_commands = planner_non_mutating and not _is_planner_review_phase(phase)
    notes = _safe_text(parsed.get("notes"), fallback="")
    if planner_non_mutating and (requested_file_updates or suppress_commands):
        ignored_commands = len(requested_commands) if suppress_commands else 0
        notes = (
            f"{notes} planner_non_mutating_mode: ignored "
            f"{len(requested_file_updates)} file_updates and {ignored_commands} run_commands."
        ).strip()
    if planner_non_mutating and requested_patch:
        notes = f"{notes} planner_non_mutating_mode: ignored patch output.".strip()

    commit_message = _safe_text(parsed.get("commit_message"), fallback=f"{role}: {phase} update")
    output.update(
        {
            "summary": _safe_text(parsed.get("summary"), fallback=f"{role} {phase} complete"),
            "notes": notes,
            # Intents are materialized by the harness through ToolRouter.
            "intent_file_updates": requested_file_updates,
            "intent_patch": requested_patch,
            "intent_run_commands": requested_commands,
            "intent_commit_message": commit_message,
            "file_updates_attempted": len(requested_file_updates),
            "file_updates_applied": 0,
            "file_updates_rejected": 0,
            "patch_attempted": bool(requested_patch),
            "patch_applied": False,
            "patch_applied_paths": [],
            "assignment_deviation_paths": [],
            "rejected_paths": [],
            "applied_paths": [],
            "run_commands_attempted": len(requested_commands),
            "commands_run": [],
            "command_policy_max_commands": command_policy["max_commands"],
            "command_policy_timeout_sec": command_policy["command_timeout_sec"],
            "planner_non_mutating": planner_non_mutating,
            "commit": None,
            "changed": False,
            "execution_mode": "harness_tool_router",
        }
    )
    if role == "planner_reviewer" and phase == "review":
        output["merge_commits"] = _normalize_merge_commits(parsed.get("merge_commits"))
        output["request_rework"] = bool(parsed.get("request_rework"))
        output["coder_feedback"] = _normalize_coder_feedback(parsed.get("coder_feedback"))
    elif role == "planner_reviewer" and phase == "review_verify":
        output["request_rework"] = bool(parsed.get("request_rework"))
        output["coder_feedback"] = _normalize_coder_feedback(parsed.get("coder_feedback"))

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
    require_structured = _require_structured_reply(role=role, phase=phase)

    prompt_keys = (
        "reflection_directive",
        "review_stage",
        "task_id",
        "round",
        "coordination_phase",
        "assignment",
        "claimed_task",
        "public_validate_stderr",
        "public_validate_stdout",
        "planner_summary",
        "coder_commits",
        "candidate_merge_commits",
        "latest_coder_outputs",
        "implementation_messages",
        "last_public_validation",
        "coordination_summary",
        "review_diff_tool",
        "knowledge_tool",
    )
    user_prompt = {
        "role": role,
        "phase": phase,
        **{key: context.get(key) for key in prompt_keys},
        "repo_context": repo_ctx,
    }
    if require_structured:
        user_prompt["required_json_schema"] = _schema_hint_for_phase(phase)

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
    }
    if require_structured:
        payload["response_format"] = {"type": "json_object"}
    reasoning = _reasoning_payload_from_env()
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return payload


def _build_system_prompt(*, role: str, phase: str) -> str:
    if phase == "reflect":
        return " ".join([TEAM_BASE_PROMPT, REFLECTION_PROMPT, f"Current phase: {phase}."])
    role_prompt = PLANNER_ROLE_PROMPT if role == "planner_reviewer" else GENERIC_CODER_ROLE_PROMPT
    phase_hint = f"Current phase: {phase}."
    return " ".join([TEAM_BASE_PROMPT, role_prompt, phase_hint])


def _require_structured_reply(*, role: str, phase: str) -> bool:
    if phase == "reflect":
        return True
    if role == "planner_reviewer":
        return True
    # Coders can answer in normal freeform/diff style for implementation/rework.
    if phase in {"implementation", "rework"}:
        return False
    return True


def _is_planner_review_phase(phase: str) -> bool:
    phase_name = str(phase or "").strip().lower()
    return phase_name in {"review", "review_verify"}


def _reasoning_payload_from_env() -> Dict[str, Any] | None:
    payload: Dict[str, Any] = {}
    for name, key in (
        ("OPENROUTER_REASONING_ENABLED", "enabled"),
        ("OPENROUTER_REASONING_EXCLUDE", "exclude"),
    ):
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            payload[key] = True
        elif value in {"0", "false", "no", "off"}:
            payload[key] = False

    effort = (os.environ.get("OPENROUTER_REASONING_EFFORT") or "").strip()
    if effort:
        payload["effort"] = effort

    raw_max_tokens = (os.environ.get("OPENROUTER_REASONING_MAX_TOKENS") or "").strip()
    if raw_max_tokens:
        try:
            payload["max_tokens"] = max(0, min(65536, int(raw_max_tokens)))
        except ValueError:
            pass

    return payload or None


def _schema_hint_for_phase(phase: str) -> Dict[str, Any]:
    worker_schema = {
        "status": "completed",
        "summary": "what changed",
        "notes": "optional caveats",
        "file_updates": [FILE_UPDATE_SCHEMA],
        "run_commands": ["pytest -q", "npm test -- --runInBand"],
        "commit_message": "concise commit message",
    }
    if phase == "reflect":
        return {
            "status": "completed",
            "directive": "concise coaching instruction for next round (200-500 chars)",
            "task_understanding": "refined understanding of what the task requires",
            "failure_patterns": "current failure patterns (rewritten, not appended)",
            "workflow_insights": "strategic observations about what's working",
            "superseded": ["list of previous insights that are no longer valid"],
        }

    if phase == "bootstrap":
        return {
            "status": "completed",
            "summary": "short summary",
            "plan_markdown": "# Plan ...",
            "subtasks": DEFAULT_BOOTSTRAP_SUBTASKS,
        }

    if phase == "review":
        return {
            **worker_schema,
            "summary": "review summary",
            "notes": "static + dynamic findings",
            "run_commands": ["pytest -q"],
            "merge_commits": {
                "coder_a": ["<commit_sha_from_coder_commits>"],
                "coder_b": ["<commit_sha_from_coder_commits>"],
            },
            "request_rework": False,
            "coder_feedback": CODER_FEEDBACK_SCHEMA,
        }

    if phase == "review_verify":
        return {
            **worker_schema,
            "summary": "verification summary on integrated candidate",
            "notes": "what passed/failed and why",
            "run_commands": ["pytest -q"],
            "request_rework": False,
            "coder_feedback": {
                "coder_a": "targeted fixes for coder_a",
                "coder_b": "targeted fixes for coder_b",
            },
            "commit_message": "optional reviewer integration fix",
        }

    return worker_schema


def _collect_repo_context(*, worktree: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    task_readme = _safe_read_text(worktree / "public" / "README.task.md", max_chars=8000)
    if task_readme is None:
        # In E2B sandboxes, `public` is a host symlink and may not be present remotely.
        task_readme = _safe_read_text(worktree / ".loopbench" / "public" / "README.task.md", max_chars=8000)
    tracked = _git_ls_files(worktree)
    candidate_paths = _select_candidate_paths(tracked_files=tracked, context=context)

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


def _normalize_merge_commits(raw: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out

    for role, commits in raw.items():
        role_name = str(role or "").strip()
        if not role_name:
            continue
        if not isinstance(commits, list):
            continue
        commit_list: List[str] = []
        for item in commits:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if token and token not in commit_list:
                commit_list.append(token)
        if commit_list:
            out[role_name] = commit_list
    return out


def _normalize_coder_feedback(raw: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for role, message in raw.items():
            role_name = str(role or "").strip()
            if not role_name or not isinstance(message, str):
                continue
            text = message.strip()
            if text:
                out[role_name] = text
        return out

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            role_name = str(item.get("role") or "").strip()
            message = item.get("feedback")
            if not role_name or not isinstance(message, str):
                continue
            text = message.strip()
            if text:
                out[role_name] = text
    return out


def _coerce_coder_reply(*, parsed: Dict[str, Any], reply_text: str) -> Dict[str, Any]:
    text = str(reply_text or "").strip()
    patch = _extract_fenced_diff_patch(text)
    commands = _extract_fenced_shell_commands(text)

    if parsed:
        out = dict(parsed)
        existing_patch = out.get("patch")
        if patch and (not isinstance(existing_patch, str) or not existing_patch.strip()):
            out["patch"] = patch

        existing_commands = out.get("run_commands")
        has_existing_commands = (
            isinstance(existing_commands, list)
            and any(isinstance(item, str) and item.strip() for item in existing_commands)
        )
        if commands and not has_existing_commands:
            out["run_commands"] = commands

        status = out.get("status")
        if not isinstance(status, str) or not status.strip():
            out["status"] = "completed"

        summary = out.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            out["summary"] = _first_nonempty_line(text, limit=400) or "coder update"

        notes = out.get("notes")
        if (not isinstance(notes, str) or not notes.strip()) and text:
            out["notes"] = text[:8000]
        return out

    summary = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            summary = stripped[:400]
            break
    if not summary:
        summary = "coder update"

    out: Dict[str, Any] = {
        "status": "completed",
        "summary": summary,
        "notes": text[:8000],
    }
    if patch:
        out["patch"] = patch
    if commands:
        out["run_commands"] = commands
    return out


def _first_nonempty_line(text: str, *, limit: int) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:limit]
    return ""


def _extract_fenced_diff_patch(text: str) -> str:
    blocks: List[str] = []
    for match in re.finditer(r"```diff\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        block = str(match.group(1) or "").strip("\n")
        if not block:
            continue
        blocks.append(_normalize_unified_patch_block(block))
    if not blocks:
        return ""
    return "\n".join(blocks).strip() + "\n"


def _normalize_unified_patch_block(block: str) -> str:
    lines = block.splitlines()
    normalized: List[str] = []
    in_hunk = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git "):
            in_hunk = False
            normalized.append(line)
            continue
        if line.startswith("@@"):
            in_hunk = True
            normalized.append(line)
            continue
        if line.startswith(("index ", "--- ", "+++ ", "new file mode ", "deleted file mode ")):
            normalized.append(line)
            continue
        if in_hunk:
            if line.startswith(("+", "-", " ", "\\")):
                normalized.append(line)
            else:
                normalized.append(f" {line}")
            continue
        normalized.append(line)

    return "\n".join(normalized)


def _extract_fenced_shell_commands(text: str) -> List[str]:
    commands: List[str] = []
    for match in re.finditer(r"```(?:bash|sh|shell)\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        block = str(match.group(1) or "")
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("$ "):
                line = line[2:].strip()
            if line and line not in commands:
                commands.append(line)
    return commands


def _get_command_policy() -> Dict[str, Any]:
    backend = os.environ.get("LOOPBENCH_SANDBOX_BACKEND", "").strip().lower()
    if backend == "e2b_firecracker":
        default_max_commands = 12
        default_timeout_sec = 1200
    else:
        default_max_commands = 8
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


def _read_int_env(*, name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


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


def _call_openrouter(*, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    http_retries = _read_int_env(
        name="OPENROUTER_HTTP_RETRIES",
        default=DEFAULT_OPENROUTER_HTTP_RETRIES,
        min_value=1,
        max_value=5,
    )
    http_timeout_sec = _read_int_env(
        name="OPENROUTER_HTTP_TIMEOUT_SEC",
        default=DEFAULT_OPENROUTER_HTTP_TIMEOUT_SEC,
        min_value=10,
        max_value=600,
    )

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

    for attempt in range(1, http_retries + 1):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=http_timeout_sec) as resp:
                response = json.loads(resp.read().decode("utf-8", errors="replace"))
                choice = _first_choice(response)
                return {
                    "reply_text": _extract_openrouter_reply_text(response),
                    "usage": _normalize_usage_metadata(response.get("usage")),
                    "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                    "native_finish_reason": (
                        choice.get("native_finish_reason") if isinstance(choice, dict) else None
                    ),
                    "response_shape": _summarize_openrouter_response_shape(response),
                }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            err_text = f"http {exc.code}: {raw}"
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
        time.sleep(attempt)

    raise RuntimeError(f"openrouter request failed after retries ({http_retries}): {err_text}")


def _request_structured_reply(
    *,
    payload: Dict[str, Any],
    api_key: str,
    phase: str,
    require_structured: bool,
) -> Dict[str, Any]:
    current_payload = json.loads(json.dumps(payload))
    attempts: List[Dict[str, Any]] = []
    attempt_records: List[Dict[str, Any]] = []
    total_usage = _zero_usage_metadata()
    structured_retries = _read_int_env(
        name="OPENROUTER_STRUCTURED_RETRIES",
        default=DEFAULT_STRUCTURED_REPLY_ATTEMPTS,
        min_value=1,
        max_value=5,
    )
    if require_structured and phase in {"review", "review_verify"}:
        structured_retries = max(structured_retries, 2)

    for attempt in range(1, structured_retries + 1):
        call_result = _call_openrouter(payload=current_payload, api_key=api_key)
        reply_text = str(call_result.get("reply_text") or "")
        attempt_usage = _normalize_usage_metadata(call_result.get("usage"))
        _accumulate_usage(total_usage, attempt_usage)
        parsed = _parse_json_object(reply_text)
        if require_structured:
            valid = _is_structured_reply_valid(phase=phase, reply_text=reply_text, parsed=parsed)
        else:
            valid = bool(reply_text.strip())

        response_shape = call_result.get("response_shape")
        if not isinstance(response_shape, dict):
            response_shape = {}
        attempt_payload = {
            "attempt": attempt,
            "reply_text": reply_text,
            "parsed": parsed,
            "structured_valid": valid,
            "usage": attempt_usage,
            "finish_reason": call_result.get("finish_reason"),
            "native_finish_reason": call_result.get("native_finish_reason"),
            "response_shape": response_shape,
        }
        attempt_records.append(attempt_payload)
        attempts.append(
            {
                "attempt": attempt,
                "raw_chars": len(reply_text),
                "structured_valid": valid,
                "parsed_keys": sorted(parsed.keys()),
                "usage": attempt_usage,
                "finish_reason": call_result.get("finish_reason"),
                "native_finish_reason": call_result.get("native_finish_reason"),
                "response_shape": response_shape,
            }
        )

        if valid:
            break

        if require_structured and attempt < structured_retries:
            current_payload = _build_repair_payload(base_payload=payload, phase=phase, bad_reply=reply_text)

    def attempt_key(record: Dict[str, Any]) -> tuple[int, int, int, int, int]:
        parsed = record.get("parsed")
        parsed_keys = len(parsed.keys()) if isinstance(parsed, dict) else 0
        reply_text = str(record.get("reply_text") or "")
        summary = parsed.get("summary") if isinstance(parsed, dict) else None
        has_summary = int(isinstance(summary, str) and bool(summary.strip()))
        return (
            int(bool(record.get("structured_valid"))),
            parsed_keys,
            has_summary,
            int(bool(reply_text.strip())),
            len(reply_text),
        )

    selected_attempt = max(attempt_records, key=attempt_key, default={})
    final_reply = selected_attempt.get("reply_text", "") if isinstance(selected_attempt, dict) else ""
    final_parsed = selected_attempt.get("parsed", {}) if isinstance(selected_attempt, dict) else {}
    if not isinstance(final_parsed, dict):
        final_parsed = {}
    final_valid = bool(selected_attempt.get("structured_valid")) if isinstance(selected_attempt, dict) else False
    selected_attempt_index = (
        int(selected_attempt.get("attempt"))
        if isinstance(selected_attempt, dict) and isinstance(selected_attempt.get("attempt"), int)
        else None
    )

    return {
        "reply_text": final_reply,
        "parsed": final_parsed,
        "attempts": attempts,
        "structured_valid": final_valid,
        "usage": total_usage,
        "selected_attempt": selected_attempt_index,
    }

def _extract_openrouter_reply_text(response: Dict[str, Any]) -> str:
    choice = _first_choice(response)
    if not isinstance(choice, dict):
        return ""

    message = choice.get("message")
    if isinstance(message, dict):
        parsed = message.get("parsed")
        if isinstance(parsed, dict) and parsed:
            return json.dumps(parsed, ensure_ascii=False)

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, dict):
            for key in ("text", "content", "output_text", "value"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                for key in ("text", "content", "output_text", "value"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
            if parts:
                return "\n".join(parts).strip()

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                fn = item.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str) and args.strip():
                    return args.strip()

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()

        # Keep non-empty fallback text for diagnostics when content is present but non-standard.
        fallback = json.dumps(message, ensure_ascii=False)
        return fallback.strip()

    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return ""


def _first_choice(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def _summarize_openrouter_response_shape(response: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    choice = _first_choice(response)
    if not choice:
        return summary

    summary["finish_reason"] = choice.get("finish_reason")
    summary["native_finish_reason"] = choice.get("native_finish_reason")

    message = choice.get("message")
    if isinstance(message, dict):
        summary["message_keys"] = sorted(str(key) for key in message.keys())
        content = message.get("content")
        summary["content_type"] = type(content).__name__
        if isinstance(content, list):
            summary["content_len"] = len(content)
            summary["content_item_types"] = [
                (
                    str(item.get("type"))
                    if isinstance(item, dict) and item.get("type") is not None
                    else ("dict" if isinstance(item, dict) else type(item).__name__)
                )
                for item in content[:12]
            ]
        elif isinstance(content, str):
            summary["content_len"] = len(content)
        summary["has_tool_calls"] = isinstance(message.get("tool_calls"), list)

    return summary


def _zero_usage_metadata() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _normalize_usage_metadata(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        return _zero_usage_metadata()

    input_tokens = _read_non_negative_int(raw.get("input_tokens"), raw.get("prompt_tokens"))
    output_tokens = _read_non_negative_int(raw.get("output_tokens"), raw.get("completion_tokens"))
    total_tokens = _read_non_negative_int(raw.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _accumulate_usage(target: Dict[str, int], delta: Dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0)) + int(delta.get(key, 0))


def _read_non_negative_int(*values: Any) -> int:
    for value in values:
        try:
            parsed = int(value)
        except Exception:  # noqa: BLE001
            continue
        if parsed >= 0:
            return parsed
    return 0


def _is_structured_reply_valid(*, phase: str, reply_text: str, parsed: Dict[str, Any]) -> bool:
    compact = reply_text.strip()
    if len(compact) < MIN_STRUCTURED_REPLY_CHARS:
        return False
    if not isinstance(parsed, dict) or not parsed:
        return False

    status = parsed.get("status")
    if not isinstance(status, str) or not status.strip():
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_ -]{1,31}", status.strip()) is None:
        return False

    if phase == "reflect":
        directive = parsed.get("directive")
        return isinstance(directive, str) and bool(directive.strip())

    if phase == "bootstrap":
        plan_md = parsed.get("plan_markdown")
        subtasks = parsed.get("subtasks")
        return isinstance(plan_md, str) and bool(plan_md.strip()) and isinstance(subtasks, list) and bool(subtasks)

    if phase == "review":
        summary = parsed.get("summary")
        merge_commits = parsed.get("merge_commits")
        request_rework = parsed.get("request_rework")
        return (
            isinstance(summary, str)
            and bool(summary.strip())
            and isinstance(merge_commits, dict)
            and isinstance(request_rework, bool)
        )

    if phase == "review_verify":
        summary = parsed.get("summary")
        request_rework = parsed.get("request_rework")
        return isinstance(summary, str) and bool(summary.strip()) and isinstance(request_rework, bool)

    summary = parsed.get("summary")
    notes = parsed.get("notes")
    file_updates = parsed.get("file_updates")
    run_commands = parsed.get("run_commands")
    return (
        (isinstance(summary, str) and bool(summary.strip()))
        or (isinstance(notes, str) and bool(notes.strip()))
        or isinstance(file_updates, list)
        or isinstance(run_commands, list)
    )


def _build_repair_payload(*, base_payload: Dict[str, Any], phase: str, bad_reply: str) -> Dict[str, Any]:
    payload = json.loads(json.dumps(base_payload))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload

    excerpt = bad_reply.strip()
    if len(excerpt) > 600:
        excerpt = excerpt[:600] + "\n...[truncated]..."
    if not excerpt:
        excerpt = "(empty reply)"
    repair_text = (
        f"Your previous reply for phase '{phase}' did not satisfy the required JSON schema. "
        "Return exactly one valid JSON object with all required keys and no extra text. "
        "Do not emit placeholder status values like ':' and do not emit markdown. "
        "Keep it compact and executable; avoid long explanations. "
        f"Previous invalid reply excerpt:\n{excerpt}"
    )
    if bad_reply.strip():
        messages.append({"role": "assistant", "content": excerpt})
    messages.append({"role": "user", "content": repair_text})
    payload["messages"] = messages
    return payload


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
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
    out: List[str] = []
    for path in sorted(worktree.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        try:
            rel = path.relative_to(worktree).as_posix()
        except Exception:  # noqa: BLE001
            continue
        if rel.startswith(".loopbench/"):
            continue
        out.append(rel)
    return out


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
