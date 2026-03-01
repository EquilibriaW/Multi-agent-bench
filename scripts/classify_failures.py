#!/usr/bin/env python3
"""
Classify failed kimi-k2.5 runs from the full-eval-2model-224x1-20260223 experiment.

Categories:
  env_build_fail  - Docker build error, dependency failure, container won't start
  merge_blocked   - Code produced but never merged successfully
  model_logic     - Merged & deployed, but hidden tests failed (wrong logic)
  timeout         - Run didn't finish (no manifest.json) — interrupted or timed out
  other           - Anything else

Usage:
    python3 scripts/classify_failures.py
"""

import json
import os
import glob
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

RUNS_DIR = "/Users/alex/Multi-agent-bench/runs"
PATTERN = "full-eval-2model-224x1-20260223_kimi-k2.5_*_r01"

# ---------------------------------------------------------------------------
# Build-error indicators in hidden_validate/result.log
# ---------------------------------------------------------------------------
BUILD_ERROR_PATTERNS = [
    re.compile(r"failed to build", re.IGNORECASE),
    re.compile(r"failed to solve", re.IGNORECASE),
    re.compile(r"ERROR: process .* did not complete successfully", re.IGNORECASE),
    re.compile(r"COMPILATION ERROR", re.IGNORECASE),
    re.compile(r"BUILD FAILED", re.IGNORECASE),
    re.compile(r"Could not resolve", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"npm ERR!", re.IGNORECASE),
    re.compile(r"npm error", re.IGNORECASE),
    re.compile(r"bundle install.*exit code", re.IGNORECASE),
    re.compile(r"An error occurred while installing", re.IGNORECASE),
    re.compile(r"no such file or directory.*Dockerfile", re.IGNORECASE),
    re.compile(r"open Dockerfile: no such file", re.IGNORECASE),
    re.compile(r"Dockerfile.*not found", re.IGNORECASE),
    re.compile(r"cannot open.*for reading", re.IGNORECASE),
    re.compile(r"Unable to find image", re.IGNORECASE),
]

CONTAINER_START_FAIL_PATTERNS = [
    re.compile(r"container .* does not exist", re.IGNORECASE),
    re.compile(r"container .* has exited", re.IGNORECASE),
    re.compile(r"Startup check failed", re.IGNORECASE),
    re.compile(r"Unable to start due to configuration error", re.IGNORECASE),
    re.compile(r"Error response from daemon", re.IGNORECASE),
]

TEST_RAN_PATTERNS = [
    re.compile(r"test session starts"),
    re.compile(r"PASSED"),
    re.compile(r"FAILED test_"),
    re.compile(r"ERROR test_"),
    re.compile(r"passed.*failed", re.IGNORECASE),
]


def read_result_log(run_dir):
    """Read hidden_validate/result.log if it exists."""
    path = os.path.join(run_dir, "hidden_validate", "result.log")
    if os.path.isfile(path):
        with open(path, "r", errors="replace") as f:
            return f.read()
    return ""


def read_manifest(run_dir):
    """Read manifest.json if it exists."""
    path = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def read_workflow_summary(run_dir):
    """Read artifacts/workflow_summary.json if it exists."""
    path = os.path.join(run_dir, "artifacts", "workflow_summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def get_elapsed_from_events(run_dir):
    """Get elapsed time from events.jsonl."""
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.isfile(path):
        return None
    first_ts = None
    last_ts = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                ts = e.get("ts_ms")
                if ts is not None:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
            except (json.JSONDecodeError, KeyError):
                pass
    if first_ts is not None and last_ts is not None:
        return (last_ts - first_ts) / 1000.0
    return None


def has_any_merged_commits(ws):
    """Check if any coder had commits merged."""
    if not ws:
        return False
    mc = ws.get("merged_commits", {})
    for role, commits in mc.items():
        if commits:
            return True
    return False


def all_coders_blocked(ws):
    """Check if no coder had any commits merged at all."""
    if not ws:
        return True
    mc = ws.get("merged_commits", {})
    if not mc:
        return True
    return all(len(commits) == 0 for commits in mc.values())


def classify_run(run_dir):
    """
    Classify a single run.
    Returns (category, reason_detail).
    """
    manifest = read_manifest(run_dir)
    ws = read_workflow_summary(run_dir)
    result_log = read_result_log(run_dir)
    task_name = os.path.basename(run_dir)

    # ── No manifest → the harness never finished writing results ─────────
    if manifest is None:
        elapsed = get_elapsed_from_events(run_dir)
        elapsed_str = f"{elapsed:.0f}s" if elapsed is not None else "unknown"
        # Sub-classify: very short runs (< 60s) are likely harness crashes /
        # experiment killed early, not genuine timeouts
        if elapsed is not None and elapsed < 60:
            return "timeout", f"no manifest, early abort (elapsed {elapsed_str})"
        return "timeout", f"no manifest (elapsed {elapsed_str})"

    metrics = manifest.get("metrics", {})
    hidden_pass = manifest.get("hidden_pass")

    # Should only be called for failed runs, but guard anyway
    if hidden_pass is True:
        return "passed", "hidden_pass=true"

    failure_reason = metrics.get("hidden_failure_reason", "")
    infra_error = metrics.get("hidden_infra_error", False)
    wall_clock = metrics.get("wall_clock_sec", 0)

    # ── 1. Check for env/build failures ──────────────────────────────────
    # Docker build errors in result.log
    for pat in BUILD_ERROR_PATTERNS:
        if pat.search(result_log):
            match_text = pat.pattern[:60]
            return "env_build_fail", f"build error: {match_text}"

    # Container failed to start (built OK but crashed on startup)
    for pat in CONTAINER_START_FAIL_PATTERNS:
        if pat.search(result_log):
            match_text = pat.pattern[:60]
            return "env_build_fail", f"container start fail: {match_text}"

    # hidden_failure_reason starts with Docker build preamble but we didn't
    # match any specific error — could be a startup/config issue. Check if
    # tests never ran.
    if failure_reason.startswith("#") and not any(
        p.search(result_log) for p in TEST_RAN_PATTERNS
    ):
        return "env_build_fail", f"docker issue, no tests ran: {failure_reason[:80]}"

    # ── 2. Check for merge_blocked ───────────────────────────────────────
    if ws and all_coders_blocked(ws):
        mc = ws.get("merge_conflicts", 0)
        return "merge_blocked", f"no commits merged (merge_conflicts={mc})"

    # Also: if merge_conflicts > 0 and only partial merges, and hidden tests
    # show ConnectionError (service never started because code was incomplete)
    if ws and ws.get("merge_conflicts", 0) > 0:
        # If the failure is a connection error, the incomplete merge likely
        # caused missing functionality
        if "ConnectionError" in result_log or "ConnectionRefusedError" in result_log:
            # But only if not all coders merged
            mc = ws.get("merged_commits", {})
            merged_roles = [r for r, c in mc.items() if c]
            total_roles = len(mc)
            if len(merged_roles) < total_roles:
                return "merge_blocked", (
                    f"partial merge ({len(merged_roles)}/{total_roles} roles), "
                    f"merge_conflicts={ws['merge_conflicts']}, service unreachable"
                )

    # ── 3. Check for model_logic (tests ran but failed) ──────────────────
    tests_ran = any(p.search(result_log) for p in TEST_RAN_PATTERNS)
    if tests_ran:
        # Extract short failure info from hidden_failure_reason
        short_reason = failure_reason[:120] if failure_reason else "tests failed"
        return "model_logic", short_reason

    # ── 4. Check for timeout ─────────────────────────────────────────────
    # manifest exists but wall_clock is very high relative to budget
    remaining = metrics.get("remaining_budget", {})
    budget_wall = remaining.get("wall_clock_sec", 3600)
    if wall_clock > 3500:  # close to typical 3600s budget
        return "timeout", f"wall_clock={wall_clock:.0f}s"

    # ── 5. Check for empty/missing hidden validation output ─────────────
    if not result_log.strip():
        return "other", f"empty hidden_validate output, wall={wall_clock:.0f}s"

    # ── 6. Fallback ──────────────────────────────────────────────────────
    return "other", f"reason={failure_reason[:100]}, wall={wall_clock:.0f}s"


def extract_task_id(run_dir_name):
    """Extract a short human-readable task id from the full directory name."""
    # Pattern: full-eval-2model-224x1-20260223_kimi-k2.5_<task_id>_r01
    prefix = "full-eval-2model-224x1-20260223_kimi-k2.5_"
    suffix = "_r01"
    name = run_dir_name
    if name.startswith(prefix):
        name = name[len(prefix):]
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name


def main():
    all_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, PATTERN)))
    print(f"Total kimi-k2.5 run directories: {len(all_dirs)}")
    print()

    # Separate passed / failed / no-manifest
    results = {}  # category -> list of (task_id, detail)
    total_passed = 0
    total_failed = 0
    total_no_manifest = 0

    for d in all_dirs:
        manifest = read_manifest(d)
        task_id = extract_task_id(os.path.basename(d))

        if manifest is None:
            total_no_manifest += 1
            cat, detail = classify_run(d)
            results.setdefault(cat, []).append((task_id, detail))
            total_failed += 1
            continue

        if manifest.get("hidden_pass") is True:
            total_passed += 1
            continue

        total_failed += 1
        cat, detail = classify_run(d)
        results.setdefault(cat, []).append((task_id, detail))

    print(f"Passed (hidden_pass=true):   {total_passed}")
    print(f"Failed (to classify):        {total_failed}")
    print(f"  of which no manifest.json: {total_no_manifest}")
    print()

    # ── Summary table ────────────────────────────────────────────────────
    CATEGORY_ORDER = [
        "env_build_fail",
        "merge_blocked",
        "model_logic",
        "timeout",
        "other",
    ]

    print("=" * 72)
    print(f"{'Category':<20s}  {'Count':>5s}  {'% of failed':>10s}")
    print("-" * 72)
    for cat in CATEGORY_ORDER:
        items = results.get(cat, [])
        pct = 100.0 * len(items) / total_failed if total_failed else 0
        print(f"{cat:<20s}  {len(items):>5d}  {pct:>9.1f}%")
    # Anything unexpected
    for cat in sorted(results.keys()):
        if cat not in CATEGORY_ORDER:
            items = results[cat]
            pct = 100.0 * len(items) / total_failed if total_failed else 0
            print(f"{cat:<20s}  {len(items):>5d}  {pct:>9.1f}%")
    print("-" * 72)
    print(f"{'TOTAL':<20s}  {total_failed:>5d}  {'100.0%':>10s}")
    print("=" * 72)
    print()

    # ── Per-category detail ──────────────────────────────────────────────
    for cat in CATEGORY_ORDER:
        items = results.get(cat, [])
        if not items:
            continue
        print(f"\n{'─' * 72}")
        print(f"  {cat.upper()} ({len(items)} runs)")
        print(f"{'─' * 72}")

        # Group by detail pattern for readability
        detail_groups = defaultdict(list)
        for task_id, detail in items:
            # Normalize detail for grouping
            detail_key = detail[:80]
            detail_groups[detail_key].append(task_id)

        for detail_key in sorted(detail_groups.keys()):
            tasks = detail_groups[detail_key]
            print(f"\n  [{len(tasks)}] {detail_key}")
            for t in sorted(tasks):
                print(f"       - {t}")

    # ── Timeout sub-analysis ─────────────────────────────────────────────
    timeout_items = results.get("timeout", [])
    early_aborts = [t for t, d in timeout_items if "early abort" in d]
    long_timeouts = [t for t, d in timeout_items if "early abort" not in d]

    print(f"\n{'=' * 72}")
    print("TIMEOUT SUB-ANALYSIS")
    print(f"{'=' * 72}")
    print(f"  Early aborts (<60s, harness killed):  {len(early_aborts):>4d}")
    print(f"  Genuine long runs (>=60s):            {len(long_timeouts):>4d}")

    # ── Model logic sub-analysis ─────────────────────────────────────────
    # Re-read result.log for each model_logic run to sub-classify
    model_items = results.get("model_logic", [])
    conn_error_tasks = []
    assertion_fail_tasks = []
    status_code_tasks = []
    other_test_tasks = []

    prefix = "full-eval-2model-224x1-20260223_kimi-k2.5_"
    suffix = "_r01"
    for task_id, detail in model_items:
        run_name = prefix + task_id + suffix
        rlog = read_result_log(os.path.join(RUNS_DIR, run_name))
        # Check what kind of failure it is
        has_conn = bool(re.search(
            r"ConnectionError|ConnectionRefused|ConnectTimeout|ReadTimeout|"
            r"requests\.exceptions\.Connection|Connection refused",
            rlog
        ))
        has_assert = bool(re.search(r"assert\s|AssertionError|AssertionError", rlog))
        has_status = bool(re.search(r"assert\s+\d+\s*==\s*\d+", rlog))

        if has_conn and not has_assert:
            conn_error_tasks.append(task_id)
        elif has_status or has_assert:
            if has_conn:
                # Mixed: some endpoints connect, some don't — still counts
                # as model logic since tests ran
                assertion_fail_tasks.append(task_id)
            else:
                assertion_fail_tasks.append(task_id)
        else:
            other_test_tasks.append(task_id)

    print(f"\n{'=' * 72}")
    print("MODEL LOGIC SUB-ANALYSIS")
    print(f"{'=' * 72}")
    print(f"  Connection errors only (service not reachable):  {len(conn_error_tasks):>4d}")
    print(f"  Assertion / status code failures:                {len(assertion_fail_tasks):>4d}")
    print(f"  Other test failures:                             {len(other_test_tasks):>4d}")
    if conn_error_tasks:
        print(f"    Connection error tasks:")
        for t in sorted(conn_error_tasks):
            print(f"      - {t}")
    if other_test_tasks:
        print(f"    Other test failure tasks:")
        for t in sorted(other_test_tasks):
            print(f"      - {t}")

    # ── Genuine model failures vs infra issues ───────────────────────────
    print(f"\n{'=' * 72}")
    print("OVERALL SUMMARY: Genuine model failures vs infrastructure/harness issues")
    print(f"{'=' * 72}")
    model_fail_count = len(results.get("model_logic", []))
    infra_count = (
        len(results.get("env_build_fail", []))
        + len(results.get("timeout", []))
    )
    merge_count = len(results.get("merge_blocked", []))
    other_count = len(results.get("other", []))

    print(f"  Model logic failures:       {model_fail_count:>4d}  (genuine — model produced wrong code)")
    print(f"  Env / build failures:       {len(results.get('env_build_fail', [])):>4d}  (infra — app wouldn't build/start)")
    print(f"  Merge blocked:              {merge_count:>4d}  (model — code never merged)")
    print(f"  Timeout / incomplete:       {len(results.get('timeout', [])):>4d}  (infra/harness — run didn't finish)")
    print(f"  Other:                      {other_count:>4d}")
    print()
    print(f"  Genuine model failures:     {model_fail_count + merge_count:>4d}  (model_logic + merge_blocked)")
    print(f"  Infrastructure issues:      {infra_count:>4d}  (env_build_fail + timeout)")
    print(f"  Ambiguous/other:            {other_count:>4d}")
    print()

    # ── Adjusted pass rate ───────────────────────────────────────────────
    total_with_manifest = total_passed + (total_failed - total_no_manifest)
    infra_with_manifest = len(results.get("env_build_fail", []))
    evaluable = total_with_manifest - infra_with_manifest
    print(f"{'=' * 72}")
    print("ADJUSTED PASS RATE (excluding infra failures)")
    print(f"{'=' * 72}")
    print(f"  Total runs with manifest:          {total_with_manifest}")
    print(f"  Env/build failures (excluded):     {infra_with_manifest}")
    print(f"  Evaluable runs:                    {evaluable}")
    print(f"  Passed:                            {total_passed}")
    if evaluable > 0:
        print(f"  Adjusted pass rate:                {100.0 * total_passed / evaluable:.1f}%")
    if total_with_manifest > 0:
        print(f"  Raw pass rate (all with manifest): {100.0 * total_passed / total_with_manifest:.1f}%")
    else:
        print(f"  Raw pass rate (all with manifest): N/A (no runs with manifest)")


if __name__ == "__main__":
    main()
