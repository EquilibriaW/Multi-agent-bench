#!/usr/bin/env python3
"""
Per-experiment aggregate usage report.

Usage:
    python scripts/harness_usage_report.py <experiment_dir_or_glob> [options]

Examples:
    python scripts/harness_usage_report.py experiments/my-experiment
    python scripts/harness_usage_report.py "runs/kimi-k25-condenser-30x1_*"

Options:
    --model-breakdown   Show per-model breakdown
    --json              Output as JSON instead of human-readable
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_data_utils import (
    extract_context_summary,
    extract_harness_design_metrics,
    extract_tool_usage_from_conversation,
    load_ambiguity_reports,
    load_contexts,
    load_conversations,
    load_manifest,
    load_review_audits,
    load_review_ledger,
    load_role_phases,
    normalize_tool_distribution,
)


def _resolve_run_dirs(target: str) -> list[Path]:
    """Resolve target to a list of run directories."""
    target_path = Path(target).resolve()

    # Case 1: experiment directory with results.jsonl
    results_file = target_path / "results.jsonl"
    if results_file.exists():
        run_dirs = []
        # Find runs root — typically sibling to experiments/
        project_root = target_path.parent.parent
        runs_root = project_root / "runs"
        with results_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                run_id = rec.get("run_id")
                if run_id:
                    rd = runs_root / run_id
                    if rd.exists():
                        run_dirs.append(rd)
        return run_dirs

    # Case 2: glob pattern
    matches = sorted(glob.glob(target))
    if matches:
        return [Path(m).resolve() for m in matches if Path(m).is_dir()]

    # Case 3: single run directory
    if target_path.is_dir() and (target_path / "events.jsonl").exists():
        return [target_path]

    return []


def _safe_mean(values: list) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.fmean(xs) if xs else None


def _safe_median(values: list) -> float | None:
    xs = [v for v in values if v is not None]
    return statistics.median(xs) if xs else None


def _ratio(a: int, b: int) -> float:
    return a / b if b > 0 else 0.0


def _collect_run_data(run_dir: Path) -> dict | None:
    """Extract all relevant data from a single run."""
    manifest = load_manifest(run_dir)
    if not manifest:
        return None

    role_phases = load_role_phases(run_dir)
    conversations = load_conversations(run_dir)
    contexts = load_contexts(run_dir)
    ledger = load_review_ledger(run_dir)

    # Determine model from role_phase outputs
    models_by_role: dict[str, str] = {}
    for rp in role_phases:
        role = rp.get("role", "unknown")
        output = rp.get("output") or {}
        model = output.get("model")
        if model and role not in models_by_role:
            models_by_role[role] = model

    # Primary model = coder model (they do most work)
    primary_model = (
        models_by_role.get("coder_a")
        or models_by_role.get("coder_b")
        or models_by_role.get("planner_reviewer")
        or "unknown"
    )

    # Aggregate conversation metrics
    conv_metrics: list[dict] = []
    for stem, conv in conversations.items():
        usage = extract_tool_usage_from_conversation(conv)
        usage["role"] = conv.get("role", "?")
        usage["phase"] = conv.get("phase", "?")
        usage["model"] = conv.get("model")
        usage["file"] = stem
        conv_metrics.append(usage)

    # Context metrics
    ctx_metrics: list[dict] = []
    for stem, ctx in contexts.items():
        summary = extract_context_summary(ctx)
        summary["role"] = ctx.get("role", "?")
        summary["phase"] = ctx.get("phase", "?")
        summary["file"] = stem
        ctx_metrics.append(summary)

    audits = load_review_audits(run_dir)
    ambiguity = load_ambiguity_reports(run_dir)

    design_metrics = extract_harness_design_metrics(
        conversations=conversations,
        contexts=contexts,
        audits=audits,
        ledger=ledger,
    )
    design_metrics["ambiguity_reports"] = len(ambiguity)

    return {
        "run_dir": str(run_dir),
        "task_id": manifest.get("task_id"),
        "hidden_pass": manifest.get("hidden_pass"),
        "public_pass": manifest.get("public_pass"),
        "primary_model": primary_model,
        "models_by_role": models_by_role,
        "metrics": manifest.get("metrics") or {},
        "role_phases": role_phases,
        "conv_metrics": conv_metrics,
        "ctx_metrics": ctx_metrics,
        "ledger": ledger,
        "design_metrics": design_metrics,
    }


def _build_model_summary(runs: list[dict]) -> dict:
    """Build per-model summary from collected run data."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_model[run["primary_model"]].append(run)

    summaries = {}
    for model, model_runs in sorted(by_model.items()):
        n = len(model_runs)
        n_hidden_pass = sum(1 for r in model_runs if r["hidden_pass"] is True)
        n_public_pass = sum(1 for r in model_runs if r["public_pass"] is True)
        n_complete = sum(1 for r in model_runs if r["metrics"].get("wall_clock_sec") is not None)

        # Token aggregation
        all_input = []
        all_output = []
        all_turns = []
        tool_dist_total: Counter = Counter()
        anti_pattern_total = 0
        exec_total = 0
        total_writes = 0

        # Rework metrics
        rework_phases = 0
        rework_with_writes = 0
        rework_files_changed: list[int] = []
        rework_with_prior = 0
        rework_with_prior_and_writes = 0
        rework_without_prior = 0
        rework_without_prior_and_writes = 0

        for run in model_runs:
            for cm in run["conv_metrics"]:
                all_input.append(cm["total_input_tokens"])
                all_output.append(cm["total_output_tokens"])
                all_turns.append(cm["turns_total"])
                for tool, count in cm["tool_distribution"].items():
                    tool_dist_total[tool] += count
                anti_pattern_total += len(cm["anti_patterns"])
                exec_total += cm["tool_distribution"].get("execute", 0) + cm["tool_distribution"].get("exec", 0)
                total_writes += cm["write_count"]

                # Rework tracking
                if "rework" in cm.get("phase", ""):
                    rework_phases += 1
                    if cm["write_count"] > 0:
                        rework_with_writes += 1
                    if cm["continued_from_prior"]:
                        rework_with_prior += 1
                        if cm["write_count"] > 0:
                            rework_with_prior_and_writes += 1
                    else:
                        rework_without_prior += 1
                        if cm["write_count"] > 0:
                            rework_without_prior_and_writes += 1

            # Files changed in rework from role_phases
            for rp in run["role_phases"]:
                if "rework" in rp.get("phase", ""):
                    output = rp.get("output") or {}
                    fc = output.get("files_changed")
                    if isinstance(fc, list):
                        rework_files_changed.append(len(fc))

        summaries[model] = {
            "n_runs": n,
            "pass_rate": _ratio(n_hidden_pass, n),
            "completion_rate": _ratio(n_complete, n),
            "mean_turns": _safe_mean(all_turns),
            "mean_input_tokens": _safe_mean(all_input),
            "mean_output_tokens": _safe_mean(all_output),
            "tool_distribution_top10": dict(tool_dist_total.most_common(10)),
            "anti_pattern_rate": _ratio(anti_pattern_total, exec_total) if exec_total else 0.0,
            "anti_pattern_count": anti_pattern_total,
            "total_writes": total_writes,
            "rework": {
                "phases_total": rework_phases,
                "write_rate": _ratio(rework_with_writes, rework_phases),
                "mean_files_changed": _safe_mean(rework_files_changed),
                "prior_conversation_rate": _ratio(rework_with_prior, rework_phases),
                "write_rate_with_prior": _ratio(rework_with_prior_and_writes, rework_with_prior) if rework_with_prior else None,
                "write_rate_without_prior": _ratio(rework_without_prior_and_writes, rework_without_prior) if rework_without_prior else None,
            },
        }

    return summaries


def _build_phase_summary(runs: list[dict]) -> dict:
    """Build per-phase-type summary."""
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        for cm in run["conv_metrics"]:
            by_phase[cm.get("phase", "unknown")].append(cm)

    summaries = {}
    for phase, metrics in sorted(by_phase.items()):
        tool_dist: Counter = Counter()
        anti_count = 0
        exec_count = 0
        tokens_in = []
        tokens_out = []

        for m in metrics:
            for tool, count in m["tool_distribution"].items():
                tool_dist[tool] += count
            anti_count += len(m["anti_patterns"])
            exec_count += m["tool_distribution"].get("execute", 0) + m["tool_distribution"].get("exec", 0)
            tokens_in.append(m["total_input_tokens"])
            tokens_out.append(m["total_output_tokens"])

        summaries[phase] = {
            "n_phases": len(metrics),
            "mean_input_tokens": _safe_mean(tokens_in),
            "mean_output_tokens": _safe_mean(tokens_out),
            "tool_distribution_top5": dict(tool_dist.most_common(5)),
            "anti_pattern_rate": _ratio(anti_count, exec_count) if exec_count else 0.0,
        }

    return summaries


def _build_review_summary(runs: list[dict]) -> dict:
    """Build review pipeline summary."""
    rounds_per_run = []
    round_decisions: dict[int, Counter] = defaultdict(Counter)
    round_merge_counts: dict[int, list[int]] = defaultdict(list)

    for run in runs:
        ledger = run["ledger"]
        rounds_per_run.append(len(ledger))
        for entry in ledger:
            ri = entry.get("round_index", 0)
            decision = entry.get("decision", "unknown")
            round_decisions[ri][decision] += 1
            merged = entry.get("commits_merged") or {}
            total_merged = sum(len(v) for v in merged.values() if isinstance(v, list))
            round_merge_counts[ri].append(total_merged)

    # Rounds to accept: for runs that eventually accepted
    accepted_runs = [
        run for run in runs
        if any(e.get("decision") in ("merge", "finalize") for e in run["ledger"])
    ]
    rounds_to_accept = []
    for run in accepted_runs:
        for entry in run["ledger"]:
            if entry.get("decision") in ("merge", "finalize"):
                rounds_to_accept.append(entry.get("round_index", 0) + 1)
                break

    per_round = {}
    for ri in sorted(round_decisions.keys()):
        decisions = dict(round_decisions[ri])
        total = sum(decisions.values())
        rework_count = decisions.get("rework", 0)
        per_round[f"round_{ri}"] = {
            "decisions": decisions,
            "rework_rate": _ratio(rework_count, total),
            "mean_commits_merged": _safe_mean(round_merge_counts.get(ri, [])),
        }

    return {
        "mean_review_rounds": _safe_mean(rounds_per_run),
        "median_review_rounds": _safe_median(rounds_per_run),
        "mean_rounds_to_accept": _safe_mean(rounds_to_accept),
        "per_round": per_round,
    }


def _build_failure_analysis(runs: list[dict]) -> dict:
    """Build failure analysis summary."""
    bucket_counter: Counter = Counter()
    for run in runs:
        if run["hidden_pass"] is True:
            continue
        metrics = run["metrics"]
        reason = metrics.get("hidden_failure_reason")
        infra = metrics.get("hidden_infra_error")
        if infra:
            bucket_counter["infra_error"] += 1
        elif reason:
            bucket_counter[reason[:80]] += 1
        else:
            bucket_counter["unknown"] += 1

    return {
        "total_failures": sum(bucket_counter.values()),
        "buckets": dict(bucket_counter.most_common(15)),
    }


def _build_design_metrics_summary(runs: list[dict]) -> dict:
    """Aggregate harness-design metrics across runs."""
    docs_lookups = 0
    docs_before_submit = 0
    total_submits = 0
    search_uses = 0
    grep_exec = 0
    malformed_merge = 0
    uninspected_nom = 0
    no_dyn_check = 0
    total_audit_rounds = 0
    rework_total = 0
    rework_salvaged = 0
    first_write_turns: list[float] = []
    first_pass_rounds: list[int] = []
    ambiguity_count = 0

    # Split by outcome for comparative analysis
    pass_runs = [r for r in runs if r["hidden_pass"] is True]
    fail_runs = [r for r in runs if r["hidden_pass"] is not True]

    for run in runs:
        dm = run.get("design_metrics") or {}
        docs_lookups += dm.get("docs_lookup_count", 0)
        search_uses += dm.get("search_tool_uses", 0)
        grep_exec += dm.get("grep_via_exec", 0)
        malformed_merge += dm.get("malformed_merge_commits", 0)
        uninspected_nom += dm.get("uninspected_nominations", 0)
        no_dyn_check += dm.get("no_dynamic_check", 0)
        total_audit_rounds += dm.get("total_audit_rounds", 0)
        rework_total += dm.get("rework_total", 0)
        rework_salvaged += dm.get("rework_with_writes", 0)
        ambiguity_count += dm.get("ambiguity_reports", 0)

        fwt = dm.get("mean_first_write_turn")
        if fwt is not None:
            first_write_turns.append(fwt)

        fpr = dm.get("first_passing_validation_round")
        if fpr is not None:
            first_pass_rounds.append(fpr)

    # Count submits and docs-before-submit from conversations directly
    for run in runs:
        for cm in run["conv_metrics"]:
            dist = cm["tool_distribution"]
            if "submit" in dist:
                total_submits += dist["submit"]

    # Compute docs-before-submit for pass vs fail
    docs_before_pass = sum(
        (r.get("design_metrics") or {}).get("docs_lookup_count", 0) > 0
        for r in pass_runs
    )
    docs_before_fail = sum(
        (r.get("design_metrics") or {}).get("docs_lookup_count", 0) > 0
        for r in fail_runs
    )

    return {
        "docs_lookup_total": docs_lookups,
        "docs_lookup_rate_pass": _ratio(docs_before_pass, len(pass_runs)) if pass_runs else None,
        "docs_lookup_rate_fail": _ratio(docs_before_fail, len(fail_runs)) if fail_runs else None,
        "search_tool_uses": search_uses,
        "grep_via_exec": grep_exec,
        "search_adoption": _ratio(search_uses, search_uses + grep_exec) if (search_uses + grep_exec) else None,
        "total_audit_rounds": total_audit_rounds,
        "malformed_merge_commits": malformed_merge,
        "malformed_merge_rate": _ratio(malformed_merge, total_audit_rounds),
        "uninspected_nominations": uninspected_nom,
        "uninspected_nomination_rate": _ratio(uninspected_nom, total_audit_rounds),
        "no_dynamic_check": no_dyn_check,
        "no_dynamic_check_rate": _ratio(no_dyn_check, total_audit_rounds),
        "rework_salvage_rate": _ratio(rework_salvaged, rework_total),
        "rework_total": rework_total,
        "rework_salvaged": rework_salvaged,
        "mean_first_write_turn": _safe_mean(first_write_turns),
        "mean_first_passing_round": _safe_mean(first_pass_rounds),
        "ambiguity_reports": ambiguity_count,
    }


def generate_report(run_dirs: list[Path], model_breakdown: bool = False) -> dict:
    """Generate full aggregate report."""
    runs = []
    for rd in run_dirs:
        data = _collect_run_data(rd)
        if data:
            runs.append(data)

    if not runs:
        return {"error": "No valid runs found"}

    report: dict = {
        "n_runs": len(runs),
        "n_pass": sum(1 for r in runs if r["hidden_pass"] is True),
        "pass_rate": _ratio(
            sum(1 for r in runs if r["hidden_pass"] is True),
            len(runs),
        ),
    }

    if model_breakdown:
        report["per_model"] = _build_model_summary(runs)

    report["per_phase"] = _build_phase_summary(runs)
    report["review_pipeline"] = _build_review_summary(runs)
    report["failure_analysis"] = _build_failure_analysis(runs)
    report["harness_design"] = _build_design_metrics_summary(runs)

    return report


def _print_human(report: dict) -> None:
    """Print human-readable report."""
    if "error" in report:
        print(f"Error: {report['error']}", file=sys.stderr)
        return
    print("=" * 72)
    print("HARNESS USAGE REPORT")
    print("=" * 72)
    print(f"  Runs: {report['n_runs']}")
    print(f"  Pass: {report['n_pass']} ({report['pass_rate']:.1%})")

    per_model = report.get("per_model")
    if per_model:
        print()
        print("PER-MODEL SUMMARY")
        print("-" * 72)
        for model, ms in per_model.items():
            print(f"\n  Model: {model}")
            print(f"    runs: {ms['n_runs']}, pass_rate: {ms['pass_rate']:.1%}, completion: {ms['completion_rate']:.1%}")
            mt = ms.get("mean_turns")
            mi = ms.get("mean_input_tokens")
            mo = ms.get("mean_output_tokens")
            print(f"    mean turns: {mt:.1f}" if mt else "    mean turns: n/a", end="")
            print(f", mean tokens: {mi:.0f} in / {mo:.0f} out" if mi and mo else ", mean tokens: n/a")
            print(f"    top tools: {ms['tool_distribution_top10']}")
            print(f"    anti-pattern rate: {ms['anti_pattern_rate']:.1%} ({ms['anti_pattern_count']} total)")
            rw = ms["rework"]
            print(f"    rework: {rw['phases_total']} phases, write_rate={rw['write_rate']:.1%}", end="")
            mfc = rw.get("mean_files_changed")
            print(f", mean_files_changed={mfc:.1f}" if mfc is not None else "")
            pcr = rw.get("prior_conversation_rate")
            wrp = rw.get("write_rate_with_prior")
            wrn = rw.get("write_rate_without_prior")
            print(f"    continuity: prior_rate={pcr:.1%}" if pcr else "    continuity: n/a", end="")
            print(f", write_with_prior={wrp:.1%}" if wrp is not None else "", end="")
            print(f", write_without_prior={wrn:.1%}" if wrn is not None else "")

    print()
    print("PER-PHASE SUMMARY")
    print("-" * 72)
    for phase, ps in report.get("per_phase", {}).items():
        mi = ps.get("mean_input_tokens")
        mo = ps.get("mean_output_tokens")
        tok_str = f"{mi:.0f} in / {mo:.0f} out" if mi and mo else "n/a"
        print(f"  {phase} (n={ps['n_phases']}): tokens={tok_str}, anti_pattern_rate={ps['anti_pattern_rate']:.1%}")
        print(f"    top tools: {ps['tool_distribution_top5']}")

    rv = report.get("review_pipeline", {})
    if rv:
        print()
        print("REVIEW PIPELINE")
        print("-" * 72)
        mr = rv.get("mean_review_rounds")
        mra = rv.get("mean_rounds_to_accept")
        print(f"  mean review rounds: {mr:.1f}" if mr is not None else "  mean review rounds: n/a")
        print(f"  mean rounds to accept: {mra:.1f}" if mra is not None else "  mean rounds to accept: n/a")
        for rname, rd in rv.get("per_round", {}).items():
            print(f"  {rname}: {rd['decisions']}, rework_rate={rd['rework_rate']:.1%}")

    fa = report.get("failure_analysis", {})
    if fa.get("total_failures"):
        print()
        print("FAILURE ANALYSIS")
        print("-" * 72)
        print(f"  total failures: {fa['total_failures']}")
        for bucket, count in fa.get("buckets", {}).items():
            print(f"    {bucket}: {count}")

    hd = report.get("harness_design", {})
    if hd:
        print()
        print("HARNESS DESIGN METRICS")
        print("-" * 72)
        # Progressive disclosure
        dl = hd.get("docs_lookup_total", 0)
        dlp = hd.get("docs_lookup_rate_pass")
        dlf = hd.get("docs_lookup_rate_fail")
        print(f"  docs lookups: {dl} total", end="")
        print(f", pass_runs={dlp:.1%}" if dlp is not None else "", end="")
        print(f", fail_runs={dlf:.1%}" if dlf is not None else "")
        # Search adoption
        su = hd.get("search_tool_uses", 0)
        ge = hd.get("grep_via_exec", 0)
        sa = hd.get("search_adoption")
        print(f"  search: {su} search_files, {ge} grep-via-exec", end="")
        print(f", adoption={sa:.1%}" if sa is not None else "")
        # Review quality
        mm = hd.get("malformed_merge_commits", 0)
        mmr = hd.get("malformed_merge_rate", 0)
        un = hd.get("uninspected_nominations", 0)
        unr = hd.get("uninspected_nomination_rate", 0)
        print(f"  review quality: malformed_merges={mm} ({mmr:.1%}), uninspected_noms={un} ({unr:.1%})")
        # Rework effectiveness
        rt = hd.get("rework_total", 0)
        rs = hd.get("rework_salvaged", 0)
        rsr = hd.get("rework_salvage_rate", 0)
        print(f"  rework salvage: {rs}/{rt} ({rsr:.1%})")
        # Timing
        fwt = hd.get("mean_first_write_turn")
        fpr = hd.get("mean_first_passing_round")
        print(f"  mean first write turn: {fwt:.1f}" if fwt is not None else "  mean first write turn: n/a")
        print(f"  mean first passing round: {fpr:.1f}" if fpr is not None else "  mean first passing round: n/a")
        # Ambiguity
        ar = hd.get("ambiguity_reports", 0)
        if ar:
            print(f"  ambiguity reports: {ar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate harness usage report")
    parser.add_argument("target", type=str, help="Experiment dir or glob pattern for run dirs")
    parser.add_argument("--model-breakdown", action="store_true", help="Show per-model breakdown")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    args = parser.parse_args()

    run_dirs = _resolve_run_dirs(args.target)
    if not run_dirs:
        print(f"Error: no run directories found for: {args.target}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(run_dirs)} run directories", file=sys.stderr)
    report = generate_report(run_dirs, model_breakdown=args.model_breakdown)

    if args.json_output:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
