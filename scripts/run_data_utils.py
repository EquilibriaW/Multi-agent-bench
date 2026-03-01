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

            if name in ("write_file", "create_file", "apply_patch"):
                write_count += 1

            # Check for anti-patterns in execute/exec calls
            if name in ("execute", "exec"):
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                label = classify_anti_pattern(name, args)
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


def load_review_audits(run_dir: Path) -> list[dict]:
    """Load review audit JSON files from artifacts/review_audit/."""
    audit_dir = run_dir / "artifacts" / "review_audit"
    if not audit_dir.exists():
        return []
    audits = []
    for p in sorted(audit_dir.glob("round_*.json")):
        try:
            audits.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return audits


def load_driver_tool_events(run_dir: Path) -> list[dict]:
    """Parse events.jsonl, return list of driver_tool event payloads."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with events_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") != "driver_tool":
                continue
            payload = event.get("payload") or {}
            payload["_ts_ms"] = event.get("ts_ms")
            events.append(payload)
    return events


def load_ambiguity_reports(run_dir: Path) -> list[dict]:
    """Load ambiguity report JSONs if present."""
    reports_dir = run_dir / "ambiguity_reports"
    if not reports_dir.exists():
        return []
    reports = []
    for p in sorted(reports_dir.glob("*.json")):
        try:
            reports.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return reports


def extract_harness_design_metrics(
    conversations: dict[str, dict],
    contexts: dict[str, dict],
    audits: list[dict],
    ledger: list[dict],
) -> dict:
    """Extract harness-design metrics from run artifacts.

    These measure how well the harness affordances serve the agent,
    not just what the agent produced.
    """
    # 1. Docs lookup before submit: did agent use lookup_docs/knowledge_tool?
    docs_lookups = 0
    total_submits = 0
    docs_before_submit = 0

    # 2. Time to first write (turn index of first write_file/apply_patch)
    first_write_turns: list[int] = []

    # 3. Search tool usage vs grep-via-exec
    search_tool_uses = 0
    grep_via_exec = 0

    for _stem, conv in conversations.items():
        turns = conv.get("turns") or []
        found_docs = False
        found_write = False

        for turn in turns:
            assistant = turn.get("assistant_message") or {}
            tool_calls = assistant.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function") or {}
                name = func.get("name", "")

                if name in ("lookup_docs", "knowledge_tool"):
                    docs_lookups += 1
                    found_docs = True

                if name == "search_files":
                    search_tool_uses += 1

                if name in ("exec", "execute"):
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    cmd = (args.get("command", "") if isinstance(args, dict) else "").strip()
                    if cmd.startswith("grep ") or cmd.startswith("rg "):
                        grep_via_exec += 1

                if name in ("write_file", "apply_patch", "create_file") and not found_write:
                    found_write = True
                    first_write_turns.append(turn.get("turn", 0))

                if name == "submit":
                    total_submits += 1
                    if found_docs:
                        docs_before_submit += 1

    # 4. Review audit metrics
    malformed_merge_commits = 0
    uninspected_nominations = 0
    no_dynamic_check = 0
    total_review_rounds = len(audits)

    for audit in audits:
        # Malformed merge_commits
        if audit.get("request_rework") is False and not audit.get("merge_ok", True):
            malformed_merge_commits += 1

        # Uninspected nominations
        uninspected = audit.get("uninspected_nominated_commits_by_role") or {}
        if any(commits for commits in uninspected.values()):
            uninspected_nominations += 1

        # No dynamic check
        if not audit.get("review_dynamic_checks_ran", True):
            no_dynamic_check += 1

    # 5. Rework salvage rate (from conversations)
    rework_total = 0
    rework_with_commit = 0
    for _stem, conv in conversations.items():
        phase = conv.get("phase", "")
        if "rework" not in phase:
            continue
        rework_total += 1
        usage = extract_tool_usage_from_conversation(conv)
        if usage["write_count"] > 0:
            rework_with_commit += 1

    # 6. Time to first passing validation (from ledger)
    first_pass_round: int | None = None
    for entry in ledger:
        if entry.get("validation_passed"):
            first_pass_round = entry.get("round_index", 0) + 1
            break

    return {
        "docs_lookup_count": docs_lookups,
        "docs_before_submit_rate": docs_before_submit / total_submits if total_submits else None,
        "search_tool_uses": search_tool_uses,
        "grep_via_exec": grep_via_exec,
        "search_vs_exec_ratio": search_tool_uses / (search_tool_uses + grep_via_exec) if (search_tool_uses + grep_via_exec) else None,
        "total_audit_rounds": total_review_rounds,
        "malformed_merge_commits": malformed_merge_commits,
        "malformed_merge_rate": malformed_merge_commits / total_review_rounds if total_review_rounds else None,
        "uninspected_nominations": uninspected_nominations,
        "uninspected_nomination_rate": uninspected_nominations / total_review_rounds if total_review_rounds else None,
        "no_dynamic_check": no_dynamic_check,
        "no_dynamic_check_rate": no_dynamic_check / total_review_rounds if total_review_rounds else None,
        "rework_salvage_rate": rework_with_commit / rework_total if rework_total else None,
        "rework_total": rework_total,
        "rework_with_writes": rework_with_commit,
        "mean_first_write_turn": sum(first_write_turns) / len(first_write_turns) if first_write_turns else None,
        "first_passing_validation_round": first_pass_round,
        "ambiguity_reports": 0,  # populated by caller if available
    }


# ---------------------------------------------------------------------------
# Tool name normalization
# ---------------------------------------------------------------------------

# Canonical tool names — maps driver-specific names to canonical form
_TOOL_NAME_ALIASES: dict[str, str] = {
    "execute": "exec",
}


def normalize_tool_name(name: str) -> str:
    """Normalize tool name to canonical form."""
    return _TOOL_NAME_ALIASES.get(name, name)


def normalize_tool_distribution(dist: dict[str, int]) -> dict[str, int]:
    """Normalize all tool names in a distribution dict."""
    normalized: dict[str, int] = {}
    for name, count in dist.items():
        canonical = normalize_tool_name(name)
        normalized[canonical] = normalized.get(canonical, 0) + count
    return normalized


def classify_anti_pattern(tool_name: str, args: dict) -> str | None:
    """Return anti-pattern label if exec/execute duplicates a dedicated tool.

    E.g. exec('find ...') -> 'find_via_exec', exec('cat ...') -> 'cat_via_exec'
    Returns None if not an anti-pattern.
    """
    if tool_name not in ("execute", "exec"):
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
