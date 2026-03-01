#!/usr/bin/env python3
"""
analyze_sandbox_deaths.py

Scan all full-eval runs to find sandbox death patterns.
For each run, find the FIRST event mentioning E2B sandbox errors,
compute actual sandbox lifetime, and produce a histogram.

IMPORTANT: Distinguishes actual E2B sandbox infrastructure errors from
application-level exceptions (like UserNotFoundException in code content).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Optional

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

# Patterns that indicate ACTUAL E2B sandbox death (not code-level exceptions).
# These must appear in stderr or error fields, not in plan/code content.
E2B_ERROR_PATTERNS = [
    # E2B sandbox gone
    "Sandbox .* not found",
    "sandbox was not found",
    # HTTP transport failures indicating sandbox died
    "incomplete chunked read",
    "peer closed connection without sending complete message body",
    "RemoteProtocolError",
    "Server disconnected without sending a response",
    # E2B sync failures (our wrapper detects these)
    "e2b sync upload-extract failed",
    "e2b sync download-pack failed",
    "e2b command failed",
    # Timeout on sandbox operations
    "e2b sync exceeded",
    # Connection-level errors to sandbox
    "ConnectionResetError",
    "ConnectionRefusedError",
]

# Compiled regex for speed
E2B_ERROR_RE = re.compile("|".join(E2B_ERROR_PATTERNS), re.IGNORECASE)

# Patterns to EXCLUDE -- these are code-content false positives
FALSE_POSITIVE_PATTERNS = [
    "UserNotFoundException",
    "ItemNotFoundException",
    "RecordNotFoundException",
    "ResourceNotFoundException",
    "EntityNotFoundException",
    "ObjectNotFoundException",
    "FileNotFoundException",
    "PageNotFoundException",
    "NotFoundError",
    "NotFoundException.*Could not find",
    "NotFoundException.*not found in database",
]
FALSE_POSITIVE_RE = re.compile("|".join(FALSE_POSITIVE_PATTERNS), re.IGNORECASE)


def is_sandbox_error(line: str) -> Optional[str]:
    """
    Returns the matched error pattern if this line contains a genuine
    E2B sandbox infrastructure error. Returns None for false positives
    or no match.
    """
    # Quick pre-filter: does it contain any error-ish keyword at all?
    line_lower = line.lower()
    if not any(kw in line_lower for kw in [
        "incomplete chunked",
        "remoteprotocol",
        "peer closed",
        "e2b sync",
        "e2b command",
        "connectionreset",
        "connectionrefused",
        "server disconnected",
    ]):
        # The only other pattern is "Sandbox X not found" from E2B API.
        # We need to check stderr fields specifically, not full line content.
        try:
            evt = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return None

        payload = evt.get("payload", {})
        stderr = str(payload.get("stderr", ""))
        # Check tool_result stderr for sandbox not found
        if "sandbox" in stderr.lower() and "not found" in stderr.lower():
            return "Sandbox not found (stderr)"
        # Check for error in data field
        data = payload.get("data", {})
        if isinstance(data, dict):
            for v in data.values():
                sv = str(v)
                if "sandbox" in sv.lower() and "not found" in sv.lower():
                    return "Sandbox not found (data)"
        # Check output stderr
        output = payload.get("output", {})
        if isinstance(output, dict):
            out_stderr = str(output.get("stderr", ""))
            if ("e2b" in out_stderr.lower() or "sandbox" in out_stderr.lower()) and (
                "not found" in out_stderr.lower()
                or "incomplete chunked" in out_stderr.lower()
                or "peer closed" in out_stderr.lower()
            ):
                return "Sandbox error (output.stderr)"
        return None

    # For transport-level errors, restrict matching to infra-specific fields
    # to avoid misclassifying app-level connection failures (e.g. service-not-listening
    # test errors) as sandbox deaths.
    try:
        evt = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None

    payload = evt.get("payload", {})
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
    # Also check top-level event error field
    infra_fields.append(str(evt.get("error", "")))

    infra_text = "\n".join(infra_fields)
    m = E2B_ERROR_RE.search(infra_text)
    if m:
        return m.group(0)
    return None


def scan_run(events_path: Path):
    """Scan a run's events.jsonl and return analysis info."""
    first_ts = None
    first_error_ts = None
    first_error_msg = None
    total_events = 0
    last_ts = None
    sandbox_ids_seen = set()

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_events += 1
            ts = evt.get("ts_ms")
            if ts is None:
                payload = evt.get("payload", {})
                ts = payload.get("ts_ms")

            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            # Track sandbox IDs
            payload = evt.get("payload", {})
            data = payload.get("data", {})
            if isinstance(data, dict) and "sandbox_id" in data:
                sandbox_ids_seen.add(data["sandbox_id"])

            # Check for errors (only first one)
            if first_error_ts is not None:
                continue

            err = is_sandbox_error(line)
            if err:
                first_error_ts = ts
                first_error_msg = err

    if first_ts is None:
        return None

    run_id = events_path.parent.name
    return {
        "run_id": run_id,
        "first_ts_ms": first_ts,
        "last_ts_ms": last_ts,
        "first_error_ts_ms": first_error_ts,
        "first_error_msg": first_error_msg,
        "total_events": total_events,
        "sandbox_ids": sandbox_ids_seen,
        "total_duration_sec": (last_ts - first_ts) / 1000.0 if last_ts and first_ts else None,
        "time_to_first_error_sec": (first_error_ts - first_ts) / 1000.0 if first_error_ts and first_ts else None,
    }


def text_histogram(values, bin_width=60, label="seconds"):
    """Print a text-based histogram."""
    if not values:
        print("  (no data)")
        return

    bins = Counter()
    for v in values:
        bucket = int(v // bin_width) * bin_width
        bins[bucket] += 1

    all_buckets = sorted(bins.keys())
    max_count = max(bins.values()) if bins else 1
    bar_max = 60

    for bucket in all_buckets:
        count = bins[bucket]
        bar_len = max(1, int((count / max_count) * bar_max))
        lo = bucket
        hi = bucket + bin_width
        bar = "#" * bar_len
        print(f"  {lo:6.0f}-{hi:6.0f}s | {bar} ({count})")


def main():
    pattern = "full-eval-2model-224x1-20260223_kimi-k2.5_*"
    run_dirs = sorted(RUNS_DIR.glob(pattern))

    print(f"Scanning {len(run_dirs)} run directories matching '{pattern}'...")
    print()

    results = []
    errored = []
    clean = []

    for rd in run_dirs:
        events_path = rd / "events.jsonl"
        if not events_path.exists():
            continue
        info = scan_run(events_path)
        if info is None:
            continue
        results.append(info)
        if info["first_error_ts_ms"] is not None:
            errored.append(info)
        else:
            clean.append(info)

    print(f"Total runs scanned: {len(results)}")
    print(f"Runs with GENUINE E2B sandbox errors: {len(errored)}")
    print(f"Runs without E2B sandbox errors: {len(clean)}")
    print()

    # --- Error breakdown ---
    if errored:
        print("=" * 70)
        print("ERROR TYPE BREAKDOWN")
        print("=" * 70)
        error_types = Counter(r["first_error_msg"] for r in errored)
        for err, cnt in error_types.most_common():
            print(f"  {cnt:4d}  {err}")
        print()

    # --- Lifetime histogram for errored runs ---
    if errored:
        lifetimes = [r["time_to_first_error_sec"] for r in errored if r["time_to_first_error_sec"] is not None]
        print("=" * 70)
        print(f"TIME TO FIRST E2B ERROR (sandbox lifetime) - {len(lifetimes)} runs")
        print("=" * 70)
        if lifetimes:
            print(f"  min:    {min(lifetimes):7.1f}s  ({min(lifetimes)/60:.1f} min)")
            print(f"  max:    {max(lifetimes):7.1f}s  ({max(lifetimes)/60:.1f} min)")
            print(f"  mean:   {sum(lifetimes)/len(lifetimes):7.1f}s  ({sum(lifetimes)/len(lifetimes)/60:.1f} min)")
            sorted_lt = sorted(lifetimes)
            median = sorted_lt[len(sorted_lt) // 2]
            print(f"  median: {median:7.1f}s  ({median/60:.1f} min)")
            print()
            print("  Histogram (60s bins):")
            text_histogram(lifetimes, bin_width=60)
            print()
            print("  Histogram (300s = 5min bins):")
            text_histogram(lifetimes, bin_width=300)
        print()

    # --- Total duration histogram for ALL runs ---
    print("=" * 70)
    print(f"TOTAL RUN DURATION (all {len(results)} runs)")
    print("=" * 70)
    durations = [r["total_duration_sec"] for r in results if r["total_duration_sec"] is not None]
    if durations:
        print(f"  min:    {min(durations):7.1f}s  ({min(durations)/60:.1f} min)")
        print(f"  max:    {max(durations):7.1f}s  ({max(durations)/60:.1f} min)")
        print(f"  mean:   {sum(durations)/len(durations):7.1f}s  ({sum(durations)/len(durations)/60:.1f} min)")
        sorted_d = sorted(durations)
        median_d = sorted_d[len(sorted_d) // 2]
        print(f"  median: {median_d:7.1f}s  ({median_d/60:.1f} min)")
        print()
        print("  Histogram (300s = 5min bins):")
        text_histogram(durations, bin_width=300)
    print()

    # --- List errored runs ---
    if errored:
        print("=" * 70)
        print(f"ALL {len(errored)} ERRORED RUNS (sorted by time-to-first-error)")
        print("=" * 70)
        by_lifetime = sorted(
            [r for r in errored if r["time_to_first_error_sec"] is not None],
            key=lambda r: r["time_to_first_error_sec"]
        )
        for r in by_lifetime:
            lt = r["time_to_first_error_sec"]
            task = r["run_id"].replace("full-eval-2model-224x1-20260223_kimi-k2.5_", "")
            print(f"  {lt:7.1f}s ({lt/60:5.1f} min)  {r['first_error_msg']:<55s} {task}")
        print()

    # --- Clustering analysis ---
    if errored:
        lifetimes_filtered = [r["time_to_first_error_sec"] for r in errored if r["time_to_first_error_sec"] is not None]
        near_300 = sum(1 for lt in lifetimes_filtered if 280 <= lt <= 320)
        near_600 = sum(1 for lt in lifetimes_filtered if 580 <= lt <= 620)
        near_3600 = sum(1 for lt in lifetimes_filtered if 3580 <= lt <= 3620)
        near_10800 = sum(1 for lt in lifetimes_filtered if 10780 <= lt <= 10820)
        print("=" * 70)
        print("CLUSTERING ANALYSIS (potential timeout boundaries)")
        print("=" * 70)
        print(f"  Near 300s  (5 min,  default SDK timeout):  {near_300} runs")
        print(f"  Near 600s  (10 min):                       {near_600} runs")
        print(f"  Near 3600s (60 min, config default):       {near_3600} runs")
        print(f"  Near 10800s (180 min, longrun config):     {near_10800} runs")
        print()

        near_10 = sum(1 for lt in lifetimes_filtered if 8 <= lt <= 15)
        near_108 = sum(1 for lt in lifetimes_filtered if 100 <= lt <= 115)
        print(f"  Near 10-15s  (10.8s if ms/s mismatch):    {near_10} runs")
        print(f"  Near 100-115s (if some other mismatch):    {near_108} runs")
        print()

    # --- Summary diagnosis ---
    print("=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print()
    print("SDK Parameter Analysis:")
    print("  - e2b_code_interpreter.Sandbox inherits from e2b.Sandbox")
    print("  - Sandbox.create(timeout=N) accepts timeout in SECONDS")
    print("  - SDK docstring: 'Timeout for the sandbox in **seconds**'")
    print("  - SDK default_sandbox_timeout = 300 (seconds)")
    print("  - API model NewSandbox.timeout default = 15 (seconds)")
    print("  - PostSandboxesSandboxIDTimeoutBody.timeout is 'Timeout in seconds'")
    print()
    print("Our code:")
    print("  - config.E2BBackendConfig.timeout_sec = 10800 (for longrun)")
    print("  - e2b_sandbox.py line 64: create_kwargs['timeout'] = options.timeout_sec")
    print("  - This passes timeout=10800 to Sandbox.create(timeout=10800)")
    print("  - SDK correctly interprets this as 10800 seconds = 3 hours")
    print()
    print("=> NO UNITS MISMATCH. The SDK uses seconds throughout.")
    print("   The timeout value 10800 is correctly passed as seconds.")
    print()
    if errored:
        lifetimes_filtered = [r["time_to_first_error_sec"] for r in errored if r["time_to_first_error_sec"] is not None]
        print(f"Actual error pattern ({len(lifetimes_filtered)} errored runs):")
        if lifetimes_filtered:
            print(f"  Errors occur between {min(lifetimes_filtered)/60:.1f} and {max(lifetimes_filtered)/60:.1f} minutes")
            print(f"  This is well BEFORE the 3-hour timeout expires")
            print()
            print("Likely root causes (NOT a units mismatch):")
            print("  1. E2B platform instability / infrastructure issues")
            print("     - 'incomplete chunked read' = HTTP connection dropped mid-transfer")
            print("     - 'peer closed connection' = sandbox VM or proxy crashed")
            print("  2. Resource exhaustion in the sandbox VM (OOM, disk full)")
            print("  3. E2B platform rate limiting or capacity constraints")
            print("     (running 30 parallel sandboxes = 90+ concurrent VMs)")
            print("  4. Network instability between client and E2B API")
            print()
            print("Recommendation:")
            print("  - Add sandbox.set_timeout() keep-alive calls periodically")
            print("  - Add retry logic for sandbox operations (reconnect/recreate)")
            print("  - Monitor E2B sandbox metrics (CPU/memory) for resource exhaustion")
            print("  - Check if reducing max_parallel_rollouts reduces error rate")


if __name__ == "__main__":
    main()
