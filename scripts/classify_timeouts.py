#!/usr/bin/env python3
"""
Classify the 87 "timeout" runs (runs without manifest.json) from the
full-eval-2model-224x1-20260223 kimi-k2.5 experiment.

Categories:
  model_timeout  - Ran for >3000s (50+ min), burned most of the 60-min budget.
                   The model was too slow / stuck in rework loops.
  model_fail     - The harness ran to completion (status.md says
                   "public_pass=False") but the solution failed validation.
                   No manifest because pass=False, NOT because of a timeout.
  harness_abort  - Ran for <300s (<5 min). The experiment was killed before
                   these runs had a chance to finish.
  infra_timeout  - Ran 300-3000s with evidence of infra failures (sandbox
                   not found, Docker build errors, peer connection drops).
  model_slow     - Ran 300-3000s with model activity but no infra failures.
                   The model was working but hadn't finished when killed.

Usage:
    python scripts/classify_timeouts.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path("/Users/alex/Multi-agent-bench/runs")
GLOB = "full-eval-2model-224x1-20260223_kimi-k2.5_*_r01"
BUDGET_S = 3600  # 60-minute wall-clock budget

# Thresholds
TIMEOUT_THRESHOLD = 3000   # >3000s => model burned the budget
ABORT_THRESHOLD   = 300    # <300s  => never really started

# Infra error patterns in event payloads / stderr
INFRA_PATTERNS = [
    re.compile(r"sandbox was not found", re.IGNORECASE),
    re.compile(r"sandbox timeout", re.IGNORECASE),
    re.compile(r"docker.*(?:error|fail|timeout)", re.IGNORECASE),
    re.compile(r"peer closed connection", re.IGNORECASE),
    re.compile(r"incomplete chunked read", re.IGNORECASE),
    re.compile(r"connection.*(?:reset|refused|timed out)", re.IGNORECASE),
]


def load_events(run_dir: Path):
    """Load all events from events.jsonl."""
    events = []
    p = run_dir / "events.jsonl"
    if not p.exists():
        return events
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def get_duration_s(events):
    """Return duration in seconds from first to last event."""
    if len(events) < 2:
        return 0.0
    first_ts = events[0].get("ts_ms", 0)
    last_ts  = events[-1].get("ts_ms", 0)
    return (last_ts - first_ts) / 1000.0


def has_infra_errors(events):
    """Check if events contain infra-related errors. Returns list of clues."""
    clues = []
    for e in events:
        payload = e.get("payload", {})
        payload_str = json.dumps(payload)

        for pat in INFRA_PATTERNS:
            if pat.search(payload_str):
                clues.append(pat.pattern)
                break  # one match per event is enough

        # Also check role_phase stderr for infra errors
        if e.get("type") == "role_phase" and not payload.get("ok", True):
            stderr = payload.get("stderr", "")
            for pat in INFRA_PATTERNS:
                if pat.search(stderr):
                    clues.append(f"stderr:{pat.pattern}")
                    break
    return clues


def check_status_completed(run_dir: Path):
    """Check if status.md indicates the run fully completed (public_pass=False)."""
    status_file = run_dir / "status.md"
    if not status_file.exists():
        return False
    text = status_file.read_text()
    return "public_pass=False" in text


def count_role_stdio_files(run_dir: Path):
    """Count non-empty files in role_stdio/."""
    stdio_dir = run_dir / "role_stdio"
    if not stdio_dir.exists():
        return 0
    count = 0
    for f in stdio_dir.iterdir():
        if f.is_file() and f.stat().st_size > 0:
            count += 1
    return count


def count_rework_rounds(run_dir: Path):
    """Count how many rework rounds are mentioned in status.md."""
    status_file = run_dir / "status.md"
    if not status_file.exists():
        return 0
    text = status_file.read_text()
    rounds = re.findall(r"rework round (\d+)", text, re.IGNORECASE)
    if rounds:
        return max(int(r) for r in rounds)
    return 0


def get_status_summary(run_dir: Path):
    """Get key lines from status.md for display."""
    status_file = run_dir / "status.md"
    if not status_file.exists():
        return "(no status.md)"
    lines = status_file.read_text().strip().split("\n")
    # Filter to substantive lines (skip header, skip blank)
    return [l.strip("- ").strip() for l in lines
            if l.strip() and not l.startswith("#") and l.strip() != "-"]


def get_last_phase_from_events(events):
    """Extract the last role_phase from events for insight into what was happening."""
    last_phase = None
    for e in events:
        if e.get("type") == "role_phase":
            p = e.get("payload", {})
            last_phase = f"{p.get('role', '?')}/{p.get('phase', '?')} ok={p.get('ok', '?')}"
    return last_phase


def classify_run(run_dir: Path):
    """Classify a single run. Returns (category, details_dict)."""
    events = load_events(run_dir)
    duration = get_duration_s(events)
    completed = check_status_completed(run_dir)
    stdio_files = count_role_stdio_files(run_dir)
    rework_rounds = count_rework_rounds(run_dir)
    infra_clues = has_infra_errors(events)
    last_phase = get_last_phase_from_events(events)

    details = {
        "run": run_dir.name,
        "duration_s": round(duration, 1),
        "event_count": len(events),
        "stdio_files": stdio_files,
        "rework_rounds": rework_rounds,
        "completed_public_pass_false": completed,
        "infra_error_count": len(infra_clues),
        "last_phase": last_phase,
    }

    # Count role_phase events as evidence of model activity
    phase_count = sum(1 for e in events if e.get("type") == "role_phase")
    details["phase_count"] = phase_count

    # Classification logic
    if completed:
        # The run actually finished. No manifest because the solution failed,
        # not because of a timeout. This is a model quality issue.
        category = "model_fail"
    elif duration > TIMEOUT_THRESHOLD:
        # Burned most of the 60-min budget
        category = "model_timeout"
    elif duration < ABORT_THRESHOLD:
        # Never really started -- the experiment was killed early
        category = "harness_abort"
    elif infra_clues:
        # Between 5-50 min, and we see infra errors
        category = "infra_timeout"
    elif stdio_files > 0 or phase_count >= 3:
        # Model was actively working (produced stdio output or completed
        # multiple role phases) but was killed before finishing
        category = "model_slow"
    else:
        # Unclear -- default to harness_abort if no activity
        category = "harness_abort"

    details["infra_clues_sample"] = list(set(infra_clues))[:5]
    return category, details


def sample_model_timeout_deep_dive(results):
    """Deep-dive into 3-5 model_timeout runs to understand what the model was doing."""
    model_timeout_runs = [r for cat, r in results if cat == "model_timeout"]
    # If there are none, try model_fail with high duration
    if not model_timeout_runs:
        model_timeout_runs = sorted(
            [r for cat, r in results if cat == "model_fail"],
            key=lambda r: r["duration_s"],
            reverse=True
        )[:5]

    # Also sample some model_fail runs that ran long
    model_fail_long = sorted(
        [r for cat, r in results if cat == "model_fail" and r["duration_s"] > 1500],
        key=lambda r: r["duration_s"],
        reverse=True
    )[:3]

    sample = model_timeout_runs[:3] + model_fail_long[:2]
    # De-duplicate
    seen = set()
    deduped = []
    for r in sample:
        if r["run"] not in seen:
            seen.add(r["run"])
            deduped.append(r)
    sample = deduped[:5]

    print("\n" + "=" * 80)
    print("DEEP DIVE: Model Timeout / Long-Running Failure Samples")
    print("=" * 80)

    for r in sample:
        run_dir = RUNS_DIR / r["run"]
        print(f"\n--- {r['run']} ---")
        print(f"  Duration: {r['duration_s']}s ({r['duration_s']/60:.1f} min)")
        print(f"  Rework rounds: {r['rework_rounds']}")
        print(f"  Completed (public_pass=False): {r['completed_public_pass_false']}")
        print(f"  Last phase: {r['last_phase']}")
        print(f"  Role stdio files: {r['stdio_files']}")

        # Show status.md summary
        status_lines = get_status_summary(run_dir)
        print(f"  Status progression ({len(status_lines)} steps):")
        for line in status_lines[-10:]:  # last 10 lines
            print(f"    > {line}")

        # Check last few events
        events = load_events(run_dir)
        print(f"  Last 3 events:")
        for e in events[-3:]:
            etype = e.get("type", "?")
            payload = e.get("payload", {})
            if etype == "role_phase":
                role = payload.get("role", "?")
                phase = payload.get("phase", "?")
                ok = payload.get("ok", "?")
                stderr_snip = payload.get("stderr", "")[:120]
                print(f"    [{etype}] {role}/{phase} ok={ok}")
                if stderr_snip:
                    print(f"      stderr: {stderr_snip}")
            elif etype == "tool_result":
                tool = payload.get("tool", "?")
                ok = payload.get("ok", "?")
                print(f"    [{etype}] tool={tool} ok={ok}")
            else:
                print(f"    [{etype}]")

        # Check role_stdio for rework loops
        stdio_dir = run_dir / "role_stdio"
        if stdio_dir.exists():
            rework_files = sorted([f.name for f in stdio_dir.iterdir()
                                   if "rework" in f.name and f.name.endswith(".stdout.log")])
            if rework_files:
                print(f"  Rework stdout files: {len(rework_files)}")
                for rf in rework_files[-2:]:  # show last 2
                    fp = stdio_dir / rf
                    size = fp.stat().st_size
                    print(f"    {rf} ({size} bytes)")


def main():
    # Find all runs without manifest.json
    run_dirs = sorted(RUNS_DIR.glob(GLOB))
    no_manifest = [d for d in run_dirs if not (d / "manifest.json").exists()]
    total_runs = len(run_dirs)

    print(f"Total matching runs: {total_runs}")
    print(f"Runs without manifest.json: {len(no_manifest)}")
    print(f"Runs with manifest.json (passed): {total_runs - len(no_manifest)}")
    print()

    # Classify each run
    results = []
    for run_dir in no_manifest:
        cat, details = classify_run(run_dir)
        results.append((cat, details))

    # Aggregate
    categories = defaultdict(list)
    for cat, details in results:
        categories[cat].append(details)

    # Summary
    print("=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print()

    category_descriptions = {
        "model_fail":     "Completed but failed validation (public_pass=False). "
                          "Model produced a solution, but it was wrong.",
        "model_timeout":  "Ran >50 min, burned most of the 60-min budget. "
                          "Model was too slow or stuck in loops.",
        "model_slow":     "Ran 5-50 min with active model work but no infra errors. "
                          "Model was working but killed before finishing.",
        "infra_timeout":  "Ran 5-50 min with infra errors (sandbox/Docker/connection). "
                          "Run was hampered or blocked by infrastructure issues.",
        "harness_abort":  "Ran <5 min or had no meaningful activity. "
                          "Experiment was killed before these runs could start.",
    }

    for cat in ["model_fail", "model_timeout", "model_slow", "infra_timeout", "harness_abort"]:
        runs = categories.get(cat, [])
        durations = [r["duration_s"] for r in runs]
        print(f"  {cat:20s}  {len(runs):3d} runs")
        if durations:
            print(f"    {'':20s}  duration range: {min(durations):.0f}s - {max(durations):.0f}s "
                  f"(median {sorted(durations)[len(durations)//2]:.0f}s)")
        print(f"    {category_descriptions.get(cat, '')}")
        print()

    total_classified = sum(len(v) for v in categories.values())
    print(f"  Total classified: {total_classified}")
    print()

    # Detailed breakdown
    print("=" * 80)
    print("DETAILED LISTING BY CATEGORY")
    print("=" * 80)

    for cat in ["model_fail", "model_timeout", "model_slow", "infra_timeout", "harness_abort"]:
        runs = sorted(categories.get(cat, []), key=lambda r: -r["duration_s"])
        if not runs:
            continue
        print(f"\n--- {cat} ({len(runs)} runs) ---")
        for r in runs:
            task_id = r["run"].replace("full-eval-2model-224x1-20260223_kimi-k2.5_", "").replace("_r01", "")
            dur_min = r["duration_s"] / 60
            extra = ""
            if r["rework_rounds"] > 0:
                extra += f" rework={r['rework_rounds']}"
            if r["infra_error_count"] > 0:
                extra += f" infra_errors={r['infra_error_count']}"
            if r["last_phase"]:
                extra += f" last={r['last_phase']}"
            print(f"  {task_id:70s} {dur_min:6.1f}m  events={r['event_count']:3d}  "
                  f"stdio={r['stdio_files']:2d}{extra}")

    # Attribution analysis
    print("\n" + "=" * 80)
    print("ATTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    n_model_fail  = len(categories.get("model_fail", []))
    n_model_to    = len(categories.get("model_timeout", []))
    n_model_slow  = len(categories.get("model_slow", []))
    n_infra       = len(categories.get("infra_timeout", []))
    n_abort       = len(categories.get("harness_abort", []))

    model_attributable = n_model_fail + n_model_to
    ambiguous          = n_model_slow + n_infra
    not_attributable   = n_abort

    print(f"  Clearly model-attributable failures:   {model_attributable:3d}  "
          f"(model_fail + model_timeout)")
    print(f"    - model_fail (completed, wrong):     {n_model_fail:3d}")
    print(f"    - model_timeout (burned budget):     {n_model_to:3d}")
    print()
    print(f"  Ambiguous / mixed cause:               {ambiguous:3d}  "
          f"(model_slow + infra_timeout)")
    print(f"    - model_slow (working, killed):      {n_model_slow:3d}")
    print(f"    - infra_timeout (infra issues):      {n_infra:3d}")
    print()
    print(f"  Not attributable (experiment killed):   {not_attributable:3d}  "
          f"(harness_abort)")
    print()

    # Pass rate impact
    n_passed = total_runs - len(no_manifest)
    if total_runs > 0:
        print(f"  Original pass rate:                    {n_passed}/{total_runs} = "
              f"{100*n_passed/total_runs:.1f}%")
    else:
        print(f"  Original pass rate:                    0/0 = N/A (no runs matched)")
    print()
    # If we exclude harness_abort from denominator
    valid_runs = total_runs - n_abort
    print(f"  If we exclude harness_abort ({n_abort}) from denominator:")
    if valid_runs > 0:
        print(f"    Adjusted pass rate:                  {n_passed}/{valid_runs} = "
              f"{100*n_passed/valid_runs:.1f}%")
    else:
        print(f"    Adjusted pass rate:                  {n_passed}/{valid_runs} = N/A")
    print()
    # If we also exclude infra_timeout
    valid_runs2 = total_runs - n_abort - n_infra
    print(f"  If we also exclude infra_timeout ({n_infra}):")
    if valid_runs2 > 0:
        print(f"    Adjusted pass rate:                  {n_passed}/{valid_runs2} = "
              f"{100*n_passed/valid_runs2:.1f}%")
    else:
        print(f"    Adjusted pass rate:                  {n_passed}/{valid_runs2} = N/A")

    # Deep dive into model_timeout/model_fail samples
    sample_model_timeout_deep_dive(results)


if __name__ == "__main__":
    main()
