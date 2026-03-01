#!/usr/bin/env python3
"""
Per-run replay inspector.

Usage:
    python scripts/inspect_run.py <run_dir> [options]

Options:
    --phase PHASE       Filter to specific phase (bootstrap, implementation, rework, review, reflect, finalize)
    --role ROLE         Filter to specific role (coder_a, coder_b, planner_reviewer)
    --context           Show context details for matching phases
    --tools             Show tool usage details for matching phases
    --json              Output as JSON instead of human-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_data_utils import (
    extract_context_summary,
    extract_tool_usage_from_conversation,
    load_contexts,
    load_conversations,
    load_manifest,
    load_review_ledger,
    load_role_phases,
)


def _build_overview(run_dir: Path, manifest: dict | None, role_phases: list[dict], ledger: list[dict]) -> dict:
    """Build run overview section."""
    overview: dict = {"run_dir": str(run_dir)}

    if manifest:
        overview["task_id"] = manifest.get("task_id", "?")
        overview["run_id"] = manifest.get("run_id", "?")
        overview["roles"] = manifest.get("roles", [])
        overview["hidden_pass"] = manifest.get("hidden_pass")
        overview["public_pass"] = manifest.get("public_pass")
        overview["sandbox_backend"] = manifest.get("sandbox_backend")
        overview["started_at"] = manifest.get("started_at")
        overview["finished_at"] = manifest.get("finished_at")

        metrics = manifest.get("metrics") or {}
        overview["wall_clock_sec"] = metrics.get("wall_clock_sec")
        overview["review_iterations"] = metrics.get("review_iterations")
        overview["llm_total_tokens"] = metrics.get("llm_total_tokens")

    # Extract model from first role_phase
    models = set()
    for rp in role_phases:
        output = rp.get("output") or {}
        m = output.get("model")
        if m:
            models.add(m)
    overview["models"] = sorted(models)
    overview["review_rounds"] = len(ledger)

    return overview


def _build_phase_timeline(role_phases: list[dict]) -> list[dict]:
    """Build phase timeline from role_phase events."""
    timeline = []
    for rp in role_phases:
        output = rp.get("output") or {}
        usage = output.get("openrouter_usage") or {}
        entry = {
            "role": rp.get("role", "?"),
            "phase": rp.get("phase", "?"),
            "model": output.get("model"),
            "turns": output.get("openrouter_turn_count"),
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "files_changed": output.get("files_changed", []),
            "summary": (output.get("summary") or "")[:200],
            "ok": rp.get("ok"),
        }
        timeline.append(entry)
    return timeline


def _build_phase_details(
    conversations: dict[str, dict],
    contexts: dict[str, dict],
    phase_filter: str | None,
    role_filter: str | None,
    show_context: bool,
    show_tools: bool,
) -> list[dict]:
    """Build per-phase detail from conversation and context files."""
    details = []

    # Match conversations to contexts by stem prefix
    conv_stems = sorted(conversations.keys())

    for stem in conv_stems:
        conv = conversations[stem]
        role = conv.get("role", "?")
        phase = conv.get("phase", "?")

        if role_filter and role != role_filter:
            continue
        if phase_filter and phase_filter not in phase:
            continue

        detail: dict = {
            "file": stem,
            "role": role,
            "phase": phase,
            "model": conv.get("model"),
        }

        tool_usage = extract_tool_usage_from_conversation(conv)
        detail["turns"] = tool_usage["turns_total"]
        detail["tool_calls"] = tool_usage["tool_call_count"]
        detail["write_count"] = tool_usage["write_count"]
        detail["continued_from_prior"] = tool_usage["continued_from_prior"]
        detail["input_tokens"] = tool_usage["total_input_tokens"]
        detail["output_tokens"] = tool_usage["total_output_tokens"]

        if show_tools:
            detail["tool_distribution"] = tool_usage["tool_distribution"]
            detail["anti_patterns"] = tool_usage["anti_patterns"]
            detail["anti_pattern_count"] = len(tool_usage["anti_patterns"])

        # Find matching context
        ctx_stem = stem.replace("_conversation", "_context")
        ctx = contexts.get(ctx_stem)
        if ctx and show_context:
            detail["context"] = extract_context_summary(ctx)

        details.append(detail)

    return details


def _build_review_summary(ledger: list[dict]) -> list[dict]:
    """Build review round summary."""
    rounds = []
    for entry in ledger:
        rounds.append({
            "round": entry.get("round_index"),
            "decision": entry.get("decision"),
            "commits_merged": entry.get("commits_merged", {}),
            "open_issues": entry.get("open_issues", []),
            "validation_passed": entry.get("validation_passed"),
            "merge_ok": entry.get("merge_ok"),
            "summary": (entry.get("summary") or "")[:200],
        })
    return rounds


def _build_continuity_report(conversations: dict[str, dict]) -> dict:
    """Report session continuity across rework phases."""
    rework_convs = []
    for stem, conv in conversations.items():
        phase = conv.get("phase", "")
        if "rework" not in phase:
            continue
        tool_usage = extract_tool_usage_from_conversation(conv)
        rework_convs.append({
            "file": stem,
            "role": conv.get("role"),
            "continued_from_prior": tool_usage["continued_from_prior"],
            "turns": tool_usage["turns_total"],
            "write_count": tool_usage["write_count"],
            "tool_calls": tool_usage["tool_call_count"],
        })

    total = len(rework_convs)
    with_prior = sum(1 for r in rework_convs if r["continued_from_prior"])
    with_writes = sum(1 for r in rework_convs if r["write_count"] > 0)

    # Write rate with vs without prior conversation
    writes_with_prior = sum(1 for r in rework_convs if r["continued_from_prior"] and r["write_count"] > 0)
    writes_without_prior = sum(1 for r in rework_convs if not r["continued_from_prior"] and r["write_count"] > 0)
    n_with_prior = sum(1 for r in rework_convs if r["continued_from_prior"])
    n_without_prior = total - n_with_prior

    return {
        "rework_phases_total": total,
        "with_prior_conversation": with_prior,
        "with_writes": with_writes,
        "write_rate_with_prior": writes_with_prior / n_with_prior if n_with_prior else None,
        "write_rate_without_prior": writes_without_prior / n_without_prior if n_without_prior else None,
        "phases": rework_convs,
    }


def inspect(run_dir: Path, args: argparse.Namespace) -> dict:
    """Run full inspection and return structured result."""
    manifest = load_manifest(run_dir)
    role_phases = load_role_phases(run_dir)
    conversations = load_conversations(run_dir)
    contexts = load_contexts(run_dir)
    ledger = load_review_ledger(run_dir)

    result: dict = {}
    result["overview"] = _build_overview(run_dir, manifest, role_phases, ledger)
    result["phase_timeline"] = _build_phase_timeline(role_phases)

    show_detail = args.phase or args.role or args.context or args.tools
    if show_detail:
        result["phase_details"] = _build_phase_details(
            conversations, contexts,
            phase_filter=args.phase,
            role_filter=args.role,
            show_context=args.context,
            show_tools=args.tools,
        )

    result["review_rounds"] = _build_review_summary(ledger)
    result["session_continuity"] = _build_continuity_report(conversations)

    return result


def _print_human(result: dict) -> None:
    """Print human-readable output."""
    ov = result["overview"]
    print("=" * 72)
    print("RUN OVERVIEW")
    print("=" * 72)
    for k, v in ov.items():
        if k == "run_dir":
            continue
        print(f"  {k}: {v}")

    print()
    print("PHASE TIMELINE")
    print("-" * 72)
    fmt = "{:<20s} {:<16s} {:>6s} {:>10s} {:>10s} {:>3s}  {}"
    print(fmt.format("role", "phase", "turns", "in_tok", "out_tok", "ok", "summary"))
    print(fmt.format("----", "-----", "-----", "------", "-------", "--", "-------"))
    for entry in result["phase_timeline"]:
        turns = str(entry["turns"] or "?")
        in_tok = str(entry["input_tokens"] or "?")
        out_tok = str(entry["output_tokens"] or "?")
        ok_str = "Y" if entry["ok"] else ("N" if entry["ok"] is False else "?")
        summary = (entry["summary"] or "")[:40]
        print(fmt.format(entry["role"], entry["phase"], turns, in_tok, out_tok, ok_str, summary))

    details = result.get("phase_details")
    if details:
        print()
        print("PHASE DETAILS")
        print("-" * 72)
        for d in details:
            print(f"\n  [{d['role']}] {d['phase']} — {d['file']}")
            print(f"    model: {d.get('model')}")
            print(f"    turns: {d['turns']}, tool_calls: {d['tool_calls']}, writes: {d['write_count']}")
            print(f"    tokens: {d['input_tokens']} in / {d['output_tokens']} out")
            print(f"    continued_from_prior: {d['continued_from_prior']}")

            if "tool_distribution" in d:
                print(f"    tool distribution:")
                for tool, count in sorted(d["tool_distribution"].items(), key=lambda x: -x[1]):
                    print(f"      {tool}: {count}")
                if d.get("anti_patterns"):
                    print(f"    anti-patterns ({d['anti_pattern_count']}):")
                    for ap in d["anti_patterns"][:10]:
                        print(f"      {ap['label']}: {ap['command'][:80]}")

            if "context" in d:
                ctx = d["context"]
                print(f"    context:")
                print(f"      has_prior_conversation: {ctx['has_prior_conversation']} ({ctx['prior_message_count']} msgs)")
                print(f"      has_rework_feedback: {ctx['has_rework_feedback']}")
                print(f"      has_reflection_directive: {ctx['has_reflection_directive']}")
                print(f"      keys: {', '.join(ctx['context_keys'])}")

    rounds = result.get("review_rounds", [])
    if rounds:
        print()
        print("REVIEW ROUNDS")
        print("-" * 72)
        for r in rounds:
            merged = {k: len(v) for k, v in (r.get("commits_merged") or {}).items() if v}
            issues = len(r.get("open_issues") or [])
            print(f"  Round {r['round']}: {r['decision']}  merged={merged}  open_issues={issues}  val={r['validation_passed']}")
            if r.get("summary"):
                print(f"    {r['summary'][:100]}")

    cont = result.get("session_continuity", {})
    if cont.get("rework_phases_total", 0) > 0:
        print()
        print("SESSION CONTINUITY")
        print("-" * 72)
        print(f"  rework phases: {cont['rework_phases_total']}")
        print(f"  with prior conversation: {cont['with_prior_conversation']}")
        print(f"  with writes: {cont['with_writes']}")
        rate_with = cont.get("write_rate_with_prior")
        rate_without = cont.get("write_rate_without_prior")
        print(f"  write rate (with prior): {rate_with:.1%}" if rate_with is not None else "  write rate (with prior): n/a")
        print(f"  write rate (without prior): {rate_without:.1%}" if rate_without is not None else "  write rate (without prior): n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a single loopbench run")
    parser.add_argument("run_dir", type=str, help="Path to the run directory")
    parser.add_argument("--phase", type=str, default=None, help="Filter by phase name")
    parser.add_argument("--role", type=str, default=None, help="Filter by role name")
    parser.add_argument("--context", action="store_true", help="Show context details")
    parser.add_argument("--tools", action="store_true", help="Show tool usage details")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    result = inspect(run_dir, args)

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
