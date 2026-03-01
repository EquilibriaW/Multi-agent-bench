"""
Shared extraction functions for run artifact analysis.

Used by inspect_run.py and harness_usage_report.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_role_phases(run_dir: Path) -> list[dict]:
    """Parse events.jsonl, return list of role_phase payloads with output."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    phases = []
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "role_phase":
                continue
            payload = event.get("payload") or {}
            payload["_ts_ms"] = event.get("ts_ms")
            phases.append(payload)
    return phases


def load_conversations(run_dir: Path) -> dict[str, dict]:
    """Glob role_runtime/*_conversation.json, return {filename_stem: parsed_json}."""
    rt_dir = run_dir / "role_runtime"
    if not rt_dir.exists():
        return {}
    result = {}
    for p in sorted(rt_dir.glob("*_conversation.json")):
        try:
            result[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return result


def load_contexts(run_dir: Path) -> dict[str, dict]:
    """Glob role_runtime/*_context.json, return {filename_stem: parsed_json}."""
    rt_dir = run_dir / "role_runtime"
    if not rt_dir.exists():
        return {}
    result = {}
    for p in sorted(rt_dir.glob("*_context.json")):
        try:
            result[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return result


def load_manifest(run_dir: Path) -> dict | None:
    """Load manifest.json if present."""
    p = run_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_review_ledger(run_dir: Path) -> list[dict]:
    """Load review_ledger.json if present."""
    p = run_dir / "review_ledger.json"
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def extract_tool_usage_from_conversation(conv: dict) -> dict:
    """From a conversation.json, extract tool usage metrics.

    Returns:
        tool_call_count: int
        tool_distribution: dict[str, int] (tool_name -> count)
        anti_patterns: list[dict] (exec_command calls that duplicate dedicated tools)
        write_count: int (write_file + create_file calls)
        turns_total: int
        continued_from_prior: bool
        total_input_tokens: int
        total_output_tokens: int
    """
    turns = conv.get("turns") or []
    tool_dist: dict[str, int] = {}
    anti_patterns: list[dict] = []
    write_count = 0
    total_input = 0
    total_output = 0

    for turn in turns:
        usage = turn.get("usage") or {}
        total_input += usage.get("input_tokens", 0) or 0
        total_output += usage.get("output_tokens", 0) or 0

        assistant = turn.get("assistant_message") or {}
        tool_calls = assistant.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function") or {}
            name = func.get("name", "unknown")
            tool_dist[name] = tool_dist.get(name, 0) + 1

            if name in ("write_file", "create_file"):
                write_count += 1

            # Check for anti-patterns in execute calls
            if name == "execute":
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                label = classify_anti_pattern("execute", args)
                if label:
                    cmd = args.get("command", "")
                    if isinstance(cmd, str):
                        cmd = cmd[:120]
                    anti_patterns.append({"label": label, "command": cmd})

    return {
        "tool_call_count": sum(tool_dist.values()),
        "tool_distribution": tool_dist,
        "anti_patterns": anti_patterns,
        "write_count": write_count,
        "turns_total": len(turns),
        "continued_from_prior": bool(conv.get("continued_from_prior")),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
    }


def extract_context_summary(ctx: dict) -> dict:
    """From a context.json, extract key metadata.

    Returns:
        has_prior_conversation: bool
        prior_message_count: int
        has_rework_feedback: bool
        has_reflection_directive: bool
        context_keys: list[str]
    """
    keys = sorted(ctx.keys())

    # prior_conversation_messages is set when a rework phase carries forward
    prior_msgs = ctx.get("prior_conversation_messages") or []
    has_prior = bool(prior_msgs)
    prior_count = len(prior_msgs) if isinstance(prior_msgs, list) else 0

    # Rework feedback comes via inbox messages of kind "rework_request"
    inbox = ctx.get("inbox") or []
    has_rework = any(
        (isinstance(m, dict) and m.get("kind") == "rework_request")
        for m in inbox
    )

    # Reflection directive appears in reflect phases
    has_reflection = bool(ctx.get("reflection_directive")) or ctx.get("phase") == "reflect"

    return {
        "has_prior_conversation": has_prior,
        "prior_message_count": prior_count,
        "has_rework_feedback": has_rework,
        "has_reflection_directive": has_reflection,
        "context_keys": keys,
    }


# Patterns for exec_command anti-pattern detection
_EXEC_ANTI_PATTERNS: list[tuple[str, list[str]]] = [
    ("find_via_exec", ["find ", "find."]),
    ("cat_via_exec", ["cat "]),
    ("grep_via_exec", ["grep ", "rg "]),
    ("ls_via_exec", ["ls "]),
    ("head_via_exec", ["head "]),
    ("tail_via_exec", ["tail "]),
    ("sed_via_exec", ["sed "]),
    ("awk_via_exec", ["awk "]),
    ("wc_via_exec", ["wc "]),
]


def classify_anti_pattern(tool_name: str, args: dict) -> str | None:
    """Return anti-pattern label if exec_command duplicates a dedicated tool.

    E.g. execute('find ...') -> 'find_via_exec', execute('cat ...') -> 'cat_via_exec'
    Returns None if not an anti-pattern.
    """
    if tool_name != "execute":
        return None

    command = args.get("command", "")
    if not isinstance(command, str):
        return None

    cmd = command.strip()
    for label, prefixes in _EXEC_ANTI_PATTERNS:
        for prefix in prefixes:
            if cmd.startswith(prefix):
                return label
    return None
