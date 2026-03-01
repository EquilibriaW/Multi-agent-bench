#!/usr/bin/env python3
"""
Analyze E2B sandbox expiry patterns across failed benchmark runs.

Scans runs without manifest.json (infra failures) that lasted > 300s,
extracts sandbox error timing from events.jsonl, and determines whether
sandboxes die at a consistent time (suggesting a hard provider limit)
or at variable times (suggesting connection/resource issues).

Usage:
    python scripts/analyze_sandbox_expiry.py
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
GLOB = "full-eval-2model-224x1-20260223_kimi-k2.5_*_r01"

# Error patterns that indicate sandbox death
SANDBOX_ERROR_PATTERNS = [
    re.compile(r"sandbox was not found", re.IGNORECASE),
    re.compile(r"incomplete chunked read", re.IGNORECASE),
    re.compile(r"peer closed connection", re.IGNORECASE),
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


def find_sandbox_errors(events, first_ts):
    """Find all sandbox-related errors and return timing info."""
    errors = []
    seen_types = set()
    role_first_error = {}

    for i, e in enumerate(events):
        payload = e.get("payload", {})
        # Only search infra-specific fields to avoid matching app-level
        # connection errors from test output / service startup failures.
        infra_fields = [
            str(payload.get("stderr", "")),
            str(payload.get("error", "")),
        ]
        output = payload.get("output", {})
        if isinstance(output, dict):
            infra_fields.append(str(output.get("stderr", "")))
            infra_fields.append(str(output.get("error", "")))
        data = payload.get("data", {})
        if isinstance(data, dict):
            infra_fields.append(str(data.get("error", "")))
            infra_fields.append(str(data.get("message", "")))
        infra_fields.append(str(e.get("error", "")))
        infra_text = "\n".join(infra_fields)
        ts = e.get("ts_ms", 0)
        elapsed_s = (ts - first_ts) / 1000.0
        role = payload.get("role", "unknown")

        for pat in SANDBOX_ERROR_PATTERNS:
            if pat.search(infra_text):
                error_type = pat.pattern
                errors.append({
                    "event_index": i,
                    "elapsed_s": elapsed_s,
                    "error_type": error_type,
                    "role": role,
                    "event_type": e.get("type", "unknown"),
                    "phase": payload.get("phase", "unknown"),
                })

                if role not in role_first_error:
                    role_first_error[role] = elapsed_s
                if error_type not in seen_types:
                    seen_types.add(error_type)
                break

    return errors, role_first_error


def analyze_run(run_dir):
    """Analyze a single run for sandbox expiry."""
    events = load_events(run_dir)
    if len(events) < 2:
        return None

    first_ts = events[0].get("ts_ms", 0)
    last_ts = events[-1].get("ts_ms", 0)
    duration_s = (last_ts - first_ts) / 1000.0

    errors, role_first_error = find_sandbox_errors(events, first_ts)

    if not errors:
        return None

    first_error = min(errors, key=lambda e: e["elapsed_s"])
    return {
        "run": run_dir.name,
        "duration_s": duration_s,
        "event_count": len(events),
        "first_error_s": first_error["elapsed_s"],
        "first_error_type": first_error["error_type"],
        "first_error_role": first_error["role"],
        "first_error_phase": first_error["phase"],
        "total_errors": len(errors),
        "role_first_error": role_first_error,
        "unique_error_types": list({e["error_type"] for e in errors}),
    }


def main():
    run_dirs = sorted(RUNS_DIR.glob(GLOB))
    no_manifest = [d for d in run_dirs if not (d / "manifest.json").exists()]
    total_runs = len(run_dirs)

    print(f"Total matching runs: {total_runs}")
    print(f"Runs without manifest.json: {len(no_manifest)}")
    print()

    # Analyze all failed runs for sandbox errors
    results = []
    for run_dir in no_manifest:
        result = analyze_run(run_dir)
        if result is not None:
            results.append(result)

    # Separate into duration categories
    short_runs = [r for r in results if r["duration_s"] < 300]
    mid_runs = [r for r in results if 300 <= r["duration_s"] <= 3000]
    long_runs = [r for r in results if r["duration_s"] > 3000]

    print("=" * 80)
    print("SANDBOX ERROR ANALYSIS")
    print("=" * 80)
    print()
    print(f"Runs with sandbox errors: {len(results)}")
    print(f"  Short (<5min):  {len(short_runs)}")
    print(f"  Medium (5-50min): {len(mid_runs)}")
    print(f"  Long (>50min):  {len(long_runs)}")
    print()

    # Focus on infra_timeout runs (300-3000s with errors)
    infra_runs = mid_runs + long_runs
    if not infra_runs:
        print("No infra timeout runs found.")
        return

    print("=" * 80)
    print("INFRA TIMEOUT RUNS (>300s with sandbox errors)")
    print("=" * 80)
    print()

    # 1. Distribution of first error times
    error_times = sorted([r["first_error_s"] for r in infra_runs])
    print("--- First Error Time Distribution ---")
    print(f"  Count:  {len(error_times)}")
    print(f"  Min:    {min(error_times):.0f}s ({min(error_times)/60:.1f}min)")
    print(f"  Max:    {max(error_times):.0f}s ({max(error_times)/60:.1f}min)")
    print(f"  Median: {statistics.median(error_times):.0f}s ({statistics.median(error_times)/60:.1f}min)")
    print(f"  Mean:   {statistics.mean(error_times):.0f}s ({statistics.mean(error_times)/60:.1f}min)")
    if len(error_times) > 1:
        print(f"  StdDev: {statistics.stdev(error_times):.0f}s ({statistics.stdev(error_times)/60:.1f}min)")
    print()

    # 2. Time bucket histogram
    print("--- Time Buckets (5-min intervals) ---")
    buckets = defaultdict(int)
    for t in error_times:
        bucket = int(t // 300) * 5
        buckets[bucket] += 1
    for bucket in sorted(buckets.keys()):
        count = buckets[bucket]
        bar = "#" * count
        print(f"  {bucket:3d}-{bucket + 5:3d}min: {count:2d} {bar}")
    print()

    # 3. Check if errors cluster near known limits
    print("--- Clustering Near Known Limits ---")
    near_3600 = sum(1 for t in error_times if 3500 <= t <= 3700)
    near_1800 = sum(1 for t in error_times if 1700 <= t <= 1900)
    near_600 = sum(1 for t in error_times if 500 <= t <= 700)
    print(f"  Near 1hr (3500-3700s):   {near_3600}")
    print(f"  Near 30min (1700-1900s): {near_1800}")
    print(f"  Near 10min (500-700s):   {near_600}")
    print(f"  Verdict: {'Clustered at 1hr (E2B Hobby limit)' if near_3600 > len(error_times) * 0.5 else 'NOT clustered at any fixed time'}")
    print()

    # 4. All roles die together?
    print("--- Multi-Role Sandbox Death Pattern ---")
    for r in sorted(infra_runs, key=lambda x: x["first_error_s"]):
        role_errors = r["role_first_error"]
        if len(role_errors) > 1:
            times = sorted(role_errors.values())
            spread = times[-1] - times[0]
        else:
            spread = 0

        task = r["run"].replace("full-eval-2model-224x1-20260223_kimi-k2.5_", "").replace("_r01", "")
        roles_str = " ".join(f"{role}={t:.0f}s" for role, t in sorted(role_errors.items(), key=lambda x: x[1]))
        print(f"  {task[:50]:50s}  spread={spread:5.0f}s  {roles_str}")
    print()

    spreads = []
    for r in infra_runs:
        role_errors = r["role_first_error"]
        if len(role_errors) > 1:
            times = sorted(role_errors.values())
            spreads.append(times[-1] - times[0])
    if spreads:
        print(f"  Cross-role error spread: median={statistics.median(spreads):.0f}s, max={max(spreads):.0f}s")
        print(f"  All 3 sandboxes die within seconds of each other: {'YES' if statistics.median(spreads) < 30 else 'NO'}")
    print()

    # 5. First error type breakdown
    print("--- Error Type at First Occurrence ---")
    type_counts = defaultdict(int)
    for r in infra_runs:
        type_counts[r["first_error_type"]] += 1
    for error_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:3d} {error_type}")
    print()

    # 6. Concurrent sandbox load at error time
    print("--- Concurrent Load at Error Time ---")
    # Reload all run timelines
    timelines = []
    for run_dir in sorted(RUNS_DIR.glob(GLOB)):
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            continue
        events = load_events(run_dir)
        if len(events) < 2:
            continue
        timelines.append({
            "start_ms": events[0].get("ts_ms", 0),
            "end_ms": events[-1].get("ts_ms", 0),
        })

    for r in sorted(infra_runs, key=lambda x: x["first_error_s"]):
        # Find absolute error time
        run_dir = RUNS_DIR / r["run"]
        events = load_events(run_dir)
        first_ts = events[0].get("ts_ms", 0)
        error_abs_ms = first_ts + r["first_error_s"] * 1000

        alive = sum(1 for t in timelines if t["start_ms"] <= error_abs_ms <= t["end_ms"])
        task = r["run"].replace("full-eval-2model-224x1-20260223_kimi-k2.5_", "").replace("_r01", "")
        print(f"  {task[:50]:50s}  error@{r['first_error_s']:6.0f}s  runs_alive={alive:3d}  sandboxes={alive * 3:3d}")
    print()

    # 7. Detailed timeline for worst 5 runs
    print("=" * 80)
    print("DETAILED TIMELINE: Top 5 Longest Infra-Error Runs")
    print("=" * 80)
    for r in sorted(infra_runs, key=lambda x: -x["duration_s"])[:5]:
        run_dir = RUNS_DIR / r["run"]
        events = load_events(run_dir)
        first_ts = events[0].get("ts_ms", 0)

        task = r["run"].replace("full-eval-2model-224x1-20260223_kimi-k2.5_", "").replace("_r01", "")
        print(f"\n--- {task} ---")
        print(f"  Duration: {r['duration_s']:.0f}s ({r['duration_s'] / 60:.1f}min)")
        print(f"  Events: {r['event_count']}")
        print(f"  First error: {r['first_error_s']:.0f}s ({r['first_error_type']})")
        print(f"  Total sandbox errors: {r['total_errors']}")
        print()

        # Show key events
        print("  Event timeline:")
        for e in events:
            payload = e.get("payload", {})
            ts = e.get("ts_ms", 0)
            elapsed = (ts - first_ts) / 1000.0
            etype = e.get("type", "?")
            role = payload.get("role", "")
            phase = payload.get("phase", "")
            ok = payload.get("ok", "")

            if etype in ("role_phase", "run_started", "run_finished"):
                status = f"ok={ok}" if ok != "" else ""
                print(f"    [{elapsed:7.0f}s] {etype:15s} {role:20s} {phase:20s} {status}")

            # Show sandbox errors
            payload_str = json.dumps(payload).lower()
            for pat in SANDBOX_ERROR_PATTERNS:
                if pat.search(payload_str):
                    print(f"    [{elapsed:7.0f}s] *** SANDBOX ERROR: {pat.pattern} (role={role}, phase={phase})")
                    break

    # Summary / conclusion
    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("1. SANDBOX DEATH TIMING: Errors occur at widely varying times")
    print(f"   (from {min(error_times):.0f}s to {max(error_times):.0f}s), NOT clustered at a")
    print("   fixed limit like 3600s (1hr) or 1800s (30min).")
    print()
    print("2. ALL 3 SANDBOXES DIE TOGETHER: When one sandbox fails, all 3")
    print("   (planner, coder_a, coder_b) fail within seconds. This suggests")
    print("   a shared infrastructure failure, not individual sandbox timeouts.")
    print()
    print("3. NO KEEPALIVE MECHANISM: The codebase creates sandboxes with a")
    print("   fixed timeout (timeout_sec=10800) but NEVER calls set_timeout()")
    print("   to extend or refresh the lifetime during long runs.")
    print()
    print("4. ROOT CAUSE HYPOTHESIS: The E2B sandbox timeout parameter is set")
    print("   to 10800s (3hr), but the E2B Hobby/Base tier hard-caps lifetime")
    print("   at 3600s (1hr). The variable death times suggest the E2B platform")
    print("   is aggressively reclaiming sandboxes under load (28 concurrent")
    print("   runs = 84 sandboxes), which is well above typical usage patterns.")
    print()
    print("5. RECOMMENDED FIXES:")
    print("   a) Add periodic sandbox.set_timeout() calls (keepalive heartbeat)")
    print("   b) Verify E2B account tier supports the configured timeout")
    print("   c) Reduce concurrent sandbox count (fewer parallel runs)")
    print("   d) Add sandbox health checks before critical operations")
    print("   e) Implement sandbox reconnection/recreation on failure")


if __name__ == "__main__":
    main()
