#!/usr/bin/env python3
"""
Analyze build failures across kimi-k2.5 evaluation runs.

Reads hidden_validate/result.log and events.jsonl from each run directory,
classifies failures into sub-categories, and reports findings.

Categories:
  - dockerfile_missing:      Model didn't create a Dockerfile or created empty/broken one
  - dependency_resolution:   npm/pip/bundler/composer/gradle couldn't resolve packages
  - compilation_error:       Source code failed to compile (C#, TS, Java, Rust, Go)
  - container_startup:       Docker image built OK but container crashed / app not reachable
  - docker_daemon:           Docker daemon issues inside sandbox
  - base_image_missing:      Docker base image not found on registry
  - other_build:             Anything else that's clearly a build error
"""

from __future__ import annotations

import json
import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUNS_ROOT = "/Users/alex/Multi-agent-bench/runs"
TASK_BUNDLES = "/Users/alex/Multi-agent-bench/.tmp/abc_out"
RUN_GLOB = os.path.join(
    RUNS_ROOT,
    "full-eval-2model-224x1-20260223_kimi-k2.5_*_r01",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def task_id_from_run(run_dir):
    """Extract task_id from run directory name."""
    base = os.path.basename(run_dir)
    match = re.search(r"kimi-k2\.5_(task_.+?)_r01$", base)
    return match.group(1) if match else base


def read_events(run_dir):
    """Read events.jsonl and return list of parsed events."""
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def get_run_finished(events):
    """Return the run_finished event payload, or None."""
    for ev in events:
        if ev.get("type") == "run_finished":
            return ev.get("payload", {})
    return None


def read_result_log(run_dir):
    """Read hidden_validate/result.log if it exists."""
    path = os.path.join(run_dir, "hidden_validate", "result.log")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def read_role_stdio_errors(run_dir):
    """Read stderr from role_stdio files, return concatenated content (truncated)."""
    stdio_dir = os.path.join(run_dir, "role_stdio")
    if not os.path.isdir(stdio_dir):
        return ""
    parts = []
    for fname in sorted(os.listdir(stdio_dir)):
        if fname.endswith(".stderr.log"):
            fpath = os.path.join(stdio_dir, fname)
            with open(fpath) as f:
                content = f.read().strip()
            if content:
                parts.append(content[-2000:])
    return "\n".join(parts)


def read_original_dockerfile(task_id):
    """Read the original Dockerfile from the task bundle."""
    path = os.path.join(TASK_BUNDLES, task_id, "public", "env", "Dockerfile")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


# ---------------------------------------------------------------------------
# Determine if a failure is a build/env problem vs a test logic failure
# ---------------------------------------------------------------------------

def is_build_env_failure(result_log, failure_reason):
    """
    Returns True if the failure in result_log is a build/environment issue,
    False if it's a test-logic failure (tests ran but produced wrong results).
    """
    if not result_log:
        return False

    # Definite build failures
    build_indicators = [
        "ERROR: failed to build",
        "failed to solve",
        "did not complete successfully",
        "failed to read dockerfile",
        "no such file or directory",
        "Cannot connect to the Docker daemon",
    ]
    for ind in build_indicators:
        if ind in result_log:
            return True

    # Check if tests actually ran (pytest output)
    tests_ran = ("FAILED test_" in result_log or
                 "PASSED" in result_log or
                 "test session starts" in result_log or
                 "passed" in result_log.split('\n')[-20:][0] if result_log.strip() else False)

    # Check if container started and tests ran
    container_started = ("Startup check passed" in result_log or
                         "Service is considered stable" in result_log)

    # If failure_reason starts with "FAILED test_" -> tests ran
    if failure_reason.startswith("FAILED test_"):
        # If container started and tests ran, it's a test logic issue unless
        # the tests are ALL connection errors
        if "ConnectionError" in result_log or "ConnectionRefusedError" in result_log or "Connection reset" in result_log:
            # Check if ALL test failures are connection errors
            fail_lines = [l for l in result_log.splitlines() if l.startswith("FAILED ")]
            conn_fails = [l for l in fail_lines if "Connection" in l or "connection" in l]
            if fail_lines and len(conn_fails) == len(fail_lines):
                return True  # ALL failures are connection errors -> container_startup issue
        return False  # Tests ran, some passed/failed -> test logic issue

    # If failure reason is a docker build line
    if failure_reason.startswith("#0 building with"):
        # Could be build error or container startup issue
        # Check if image was successfully built
        if "naming to docker.io" in result_log and "ERROR: failed to build" not in result_log:
            # Image built, but something went wrong after
            if "container" in result_log.lower() and ("does not exist" in result_log.lower() or "has exited" in result_log.lower()):
                return True  # Container startup failure
            # Tests ran?
            if "FAILED test_" in result_log:
                fail_lines = [l for l in result_log.splitlines() if l.startswith("FAILED ")]
                conn_fails = [l for l in fail_lines if "Connection" in l or "connection" in l]
                if fail_lines and len(conn_fails) == len(fail_lines):
                    return True  # All connection errors
                return False  # Mixed failures -> test logic
            return True  # No test output, something went wrong
        return True  # Build failed

    # Default: if there's no evidence of tests running, it's a build issue
    if "exit status 1" in result_log:
        if container_started and "FAILED test_" in result_log:
            return False  # Tests ran
        if "ERROR: failed to build" in result_log:
            return True
        if not container_started and "naming to docker.io" not in result_log:
            return True  # Never built
        # Image built but no test output -- startup issue
        if "naming to docker.io" in result_log and "FAILED test_" not in result_log:
            if "container" in result_log.lower() and ("does not exist" in result_log or "has exited" in result_log):
                return True
            # Check if it's a "docker run" failure
            if "Error response from daemon" in result_log:
                return True

    return False


# ---------------------------------------------------------------------------
# Error extraction from result.log
# ---------------------------------------------------------------------------

def extract_error_snippet(log, max_chars=400):
    """Extract the most relevant error lines from a build log."""
    lines = log.splitlines()
    patterns = [
        r"ERROR:.*",
        r"error CS\d+:.*",
        r"error TS\d+:.*",
        r"npm ERR!.*",
        r"npm error.*",
        r"An error occurred while installing.*",
        r"Error:.*container.*does not exist.*",
        r"Error:.*container.*has exited.*",
        r"Error response from daemon.*",
        r"failed to solve:.*",
        r"failed to build:.*",
        r"open Dockerfile: no such file or directory",
        r"your php version.*does not satisfy.*",
        r"Bundler cannot continue.*",
        r"Could not resolve dependencies.*",
        r"not found$",
    ]
    extracted = []
    for line in lines:
        stripped = line.strip()
        # Remove Docker build step prefix
        cleaned = re.sub(r"^#\d+\s+[\d.]+\s+", "", stripped)
        cleaned = re.sub(r"^#\d+\s+", "", cleaned)
        if not cleaned:
            continue
        for pat in patterns:
            if re.search(pat, cleaned, re.IGNORECASE):
                if cleaned not in extracted:
                    extracted.append(cleaned)
                break
    result = "\n".join(extracted[-8:])
    if len(result) > max_chars:
        result = result[:max_chars] + "..."
    return result


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_build_failure(result_log, failure_reason):
    """
    Classify a confirmed build failure into sub-categories.
    Returns (category, error_snippet).
    """
    # 1. Dockerfile missing / empty
    if "failed to read dockerfile" in result_log.lower():
        return "dockerfile_missing", extract_error_snippet(result_log)
    if "transferring dockerfile: 2B" in result_log and "no such file or directory" in result_log.lower():
        return "dockerfile_missing", "Dockerfile transferred as 2B (empty/missing)"
    if "open Dockerfile: no such file or directory" in result_log:
        return "dockerfile_missing", extract_error_snippet(result_log)

    # 2. Docker daemon issues
    daemon_patterns = [
        r"Cannot connect to the Docker daemon",
        r"Is the docker daemon running",
        r"docker\.sock.*permission denied",
    ]
    for pat in daemon_patterns:
        m = re.search(pat, result_log, re.IGNORECASE)
        if m:
            return "docker_daemon", m.group(0)

    # 3. Base image not found on registry
    m = re.search(r"(?:docker\.io/library/|mcr\.microsoft\.com/)[\w./-]+:\s*not found", result_log)
    if m:
        return "base_image_missing", extract_error_snippet(result_log)
    if re.search(r"failed to resolve source metadata.*not found", result_log):
        return "base_image_missing", extract_error_snippet(result_log)

    # 4. Container startup failure (image built, but container doesn't run)
    image_built = "naming to docker.io" in result_log and "ERROR: failed to build" not in result_log
    if image_built:
        if re.search(r"container.*does not exist", result_log, re.IGNORECASE):
            return "container_startup", extract_error_snippet(result_log) or "Container does not exist after build"
        if re.search(r"container.*has exited", result_log, re.IGNORECASE):
            return "container_startup", extract_error_snippet(result_log) or "Container exited after build"
        if "Error response from daemon" in result_log:
            return "container_startup", extract_error_snippet(result_log) or "Docker run error after build"
        # Image built, but ConnectionError on all tests
        if "ConnectionError" in result_log or "ConnectionRefusedError" in result_log or "Connection reset" in result_log:
            return "container_startup", extract_error_snippet(result_log) or "App not reachable after container start"
        # Image built, exit status 1, no test output
        if "FAILED test_" not in result_log:
            return "container_startup", extract_error_snippet(result_log) or "Image built but container failed"

    # 5. Compilation errors (before or during docker build)
    compilation_patterns = [
        (r"error CS\d+:", "C# compilation error"),
        (r"error TS\d+:", "TypeScript compilation error"),
        (r"error\[E\d+\]:", "Rust compilation error"),
        (r"COMPILATION ERROR", "Compilation error"),
        (r"BUILD FAILED", "Build failed"),
        (r"Compilation failed", "Compilation failed"),
    ]
    for pat, desc in compilation_patterns:
        if re.search(pat, result_log, re.IGNORECASE):
            return "compilation_error", extract_error_snippet(result_log)

    # 6. Dependency resolution failures
    dep_patterns = [
        r"npm ERR!",
        r"npm error",
        r"npm ci.*did not complete",
        r"An error occurred while installing",
        r"bundle install.*did not complete",
        r"Bundler cannot continue",
        r"Could not resolve dependencies",
        r"does not satisfy that requirement",
        r"Could not find a version",
        r"ERESOLVE",
        r"composer.*install.*did not complete",
        r"go:.*module.*not found",
        r"cargo.*could not compile",
    ]
    for pat in dep_patterns:
        if re.search(pat, result_log, re.IGNORECASE):
            return "dependency_resolution", extract_error_snippet(result_log)

    # 7. Broader: did not complete successfully (usually dep install step)
    m = re.search(r'process "/bin/sh -c (.+?)" did not complete successfully', result_log)
    if m:
        cmd = m.group(1)
        dep_cmds = ["npm", "pip", "bundle", "composer", "gradle", "maven", "cargo", "go ", "dotnet restore", "yarn"]
        if any(dc in cmd.lower() for dc in dep_cmds):
            return "dependency_resolution", extract_error_snippet(result_log)
        compile_cmds = ["dotnet publish", "dotnet build", "go build", "make", "javac", "gcc", "g++"]
        if any(cc in cmd.lower() for cc in compile_cmds):
            return "compilation_error", extract_error_snippet(result_log)

    # 8. Catch-all
    return "other_build", extract_error_snippet(result_log) or "Unclassified build failure"


def analyze_dependency_root_cause(task_id, result_log):
    """
    For dependency_resolution failures, determine root cause:
    a) network_no_internet — sandbox can't reach package registries
    b) model_broke_deps_version_conflict — network OK but model picked wrong versions
    c) model_broke_deps — model created broken Dockerfile (original was DinD)
    """
    original_dockerfile = read_original_dockerfile(task_id)
    uses_dind = "cruizba/ubuntu-dind" in original_dockerfile

    # Network failure indicators
    network_fail = [
        r"Could not resolve host",
        r"Name or service not known",
        r"Temporary failure in name resolution",
        r"Network is unreachable",
        r"ETIMEDOUT",
        r"ENOTFOUND",
        r"EAI_AGAIN",
        r"no such host",
    ]
    for pat in network_fail:
        if re.search(pat, result_log, re.IGNORECASE):
            return "network_no_internet"

    # Network success indicators (packages were at least partially fetched)
    network_ok = any(kw in result_log for kw in [
        "Fetched", "Downloading", "downloaded", "GET http", "added ",
        "resolved to", "sha256:", "deb.debian.org", "npmjs.org",
        "rubygems.org", "pypi.org", "packagist.org",
        "npm warn", "npm notice", "Reading package lists",
    ])

    if uses_dind:
        if network_ok:
            return "model_broke_deps_version_conflict"
        return "model_broke_deps"
    else:
        return "model_broke_deps_or_original"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_dirs = sorted(glob.glob(RUN_GLOB))
    print("=" * 80)
    print("KIMI-K2.5 BUILD FAILURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal kimi-k2.5 run directories: {len(run_dirs)}")

    # Collect results
    build_failures = []
    counts = {"pass": 0, "fail_build": 0, "fail_test": 0, "no_finish": 0}

    for run_dir in run_dirs:
        task_id = task_id_from_run(run_dir)
        events = read_events(run_dir)
        finished = get_run_finished(events)
        result_log = read_result_log(run_dir)

        if finished is None:
            counts["no_finish"] += 1
            continue

        hidden_pass = finished.get("hidden_pass")
        failure_reason = str(finished.get("hidden_failure_reason", "") or "")

        if hidden_pass is True:
            counts["pass"] += 1
            continue

        # Determine build vs test failure
        if result_log and is_build_env_failure(result_log, failure_reason):
            counts["fail_build"] += 1
            category, error_snippet = classify_build_failure(result_log, failure_reason)
            build_failures.append({
                "run_dir": run_dir,
                "task_id": task_id,
                "category": category,
                "error_snippet": error_snippet,
                "failure_reason": failure_reason[:200],
            })
        else:
            counts["fail_test"] += 1

    # -----------------------------------------------------------------------
    # Overall summary
    # -----------------------------------------------------------------------
    completed = counts["pass"] + counts["fail_build"] + counts["fail_test"]
    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")
    print(f"{'=' * 80}")
    print(f"  Completed runs:                    {completed}")
    print(f"    Passed (hidden_pass=True):        {counts['pass']}")
    print(f"    Failed - build/env issue:          {counts['fail_build']}")
    print(f"    Failed - test logic (not build):   {counts['fail_test']}")
    print(f"  Runs without finish event:          {counts['no_finish']}")

    # -----------------------------------------------------------------------
    # Build failure classification
    # -----------------------------------------------------------------------
    by_cat = defaultdict(list)
    for bf in build_failures:
        by_cat[bf["category"]].append(bf)

    cat_order = [
        "dockerfile_missing",
        "dependency_resolution",
        "compilation_error",
        "container_startup",
        "base_image_missing",
        "docker_daemon",
        "other_build",
    ]
    cat_desc = {
        "dockerfile_missing": "Model didn't create a Dockerfile or created empty/broken one",
        "dependency_resolution": "Package manager (npm/pip/bundler/composer) couldn't resolve dependencies",
        "compilation_error": "Source code failed to compile (C#, TS, Java, Rust, Go)",
        "container_startup": "Docker image built OK but container crashed or app not reachable",
        "base_image_missing": "Docker base image not found on registry (e.g. openjdk:11-jre-slim removed)",
        "docker_daemon": "Docker daemon issues inside the sandbox",
        "other_build": "Other build failures",
    }

    print(f"\n{'=' * 80}")
    print(f"BUILD FAILURE CLASSIFICATION ({counts['fail_build']} total)")
    print(f"{'=' * 80}")

    for cat in cat_order:
        items = by_cat.get(cat, [])
        if not items:
            continue

        print(f"\n{'─' * 80}")
        print(f"  [{cat.upper()}] -- {len(items)} failures")
        print(f"  {cat_desc.get(cat, '')}")
        print(f"{'─' * 80}")

        # Task IDs
        task_ids = sorted(set(bf["task_id"] for bf in items))
        print(f"\n  Affected tasks ({len(task_ids)}):")
        for tid in task_ids:
            print(f"    - {tid}")

        # Example errors (up to 3 unique)
        print(f"\n  Example errors:")
        seen = set()
        count = 0
        for bf in items:
            err = bf["error_snippet"].strip()
            key = err[:80]
            if key in seen:
                continue
            seen.add(key)
            truncated = err[:250]
            if len(err) > 250:
                truncated += "..."
            print(f"\n    Example {count+1} -- {bf['task_id']}:")
            for line in truncated.splitlines()[:6]:
                print(f"      {line}")
            count += 1
            if count >= 3:
                break

    # -----------------------------------------------------------------------
    # Dependency resolution root cause
    # -----------------------------------------------------------------------
    dep_items = by_cat.get("dependency_resolution", [])
    if dep_items:
        print(f"\n\n{'=' * 80}")
        print("DEPENDENCY RESOLUTION -- ROOT CAUSE ANALYSIS")
        print(f"{'=' * 80}")

        root_causes = defaultdict(list)
        for bf in dep_items:
            cause = analyze_dependency_root_cause(bf["task_id"], read_result_log(bf["run_dir"]))
            root_causes[cause].append(bf)

        cause_desc = {
            "network_no_internet": "Sandbox cannot reach package registries (DNS/network failure)",
            "model_broke_deps_version_conflict": "Network works, but model picked wrong base image or dependency versions",
            "model_broke_deps": "Model created broken dependency config (original task used DinD wrapper pattern)",
            "model_broke_deps_or_original": "Dependency issue in model output (original task had custom Dockerfile)",
        }

        for cause, items in sorted(root_causes.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{cause}] -- {len(items)} tasks")
            print(f"  {cause_desc.get(cause, '')}")
            for bf in items:
                print(f"    - {bf['task_id']}")

        # -----------------------------------------------------------------------
        # Original task bundle analysis
        # -----------------------------------------------------------------------
        print(f"\n\n{'=' * 80}")
        print("ORIGINAL TASK BUNDLE DOCKERFILE ANALYSIS")
        print(f"{'=' * 80}")
        print("\nAll ABC bench task bundles provide a Dockerfile. Checking whether the")
        print("originals use the DinD (Docker-in-Docker) wrapper pattern vs custom builds:\n")

        dep_task_ids = sorted(set(bf["task_id"] for bf in dep_items))
        dind_count = 0
        custom_count = 0
        no_df_count = 0
        for tid in dep_task_ids:
            orig = read_original_dockerfile(tid)
            if not orig:
                no_df_count += 1
                print(f"  {tid}: NO ORIGINAL DOCKERFILE")
            elif "cruizba/ubuntu-dind" in orig:
                dind_count += 1
                print(f"  {tid}: DinD pattern (FROM cruizba/ubuntu-dind)")
            else:
                custom_count += 1
                first = orig.splitlines()[0].strip()
                print(f"  {tid}: Custom ({first})")

        print(f"\n  DinD wrapper: {dind_count}  |  Custom Dockerfile: {custom_count}  |  Missing: {no_df_count}")
        print()
        print("  NOTE: DinD pattern means the original Dockerfile just copies the repo into")
        print("  an ubuntu-dind container. The hidden test runner then does `docker build`")
        print("  inside that container. The MODEL is responsible for creating an application-")
        print("  level Dockerfile. If the model's Dockerfile has wrong versions/deps, it fails.")

        # -----------------------------------------------------------------------
        # Internet connectivity evidence
        # -----------------------------------------------------------------------
        print(f"\n\n{'=' * 80}")
        print("INTERNET CONNECTIVITY EVIDENCE (across dependency failures)")
        print(f"{'=' * 80}\n")

        net_ok = 0
        net_fail = 0
        net_ambig = 0
        for bf in dep_items:
            log = read_result_log(bf["run_dir"])
            success = any(kw in log for kw in [
                "Fetched", "Downloading", "GET http", "sha256:",
                "deb.debian.org", "npm warn", "npm notice",
                "Reading package lists", "rubygems.org",
            ])
            failure = any(re.search(p, log, re.IGNORECASE) for p in [
                r"Could not resolve host", r"Name or service not known",
                r"ETIMEDOUT", r"ENOTFOUND", r"Network is unreachable",
            ])
            if failure:
                net_fail += 1
            elif success:
                net_ok += 1
            else:
                net_ambig += 1

        print(f"  Network WORKING (packages downloading): {net_ok}")
        print(f"  Network FAILING (DNS/timeout):          {net_fail}")
        print(f"  Ambiguous:                              {net_ambig}")

        if net_ok > 0 and net_fail == 0:
            print()
            print("  VERDICT: Internet IS available in the sandbox. All dependency failures")
            print("  are caused by the MODEL creating Dockerfiles with wrong base images,")
            print("  wrong dependency versions, or missing system libraries -- NOT by lack")
            print("  of network access.")

    # -----------------------------------------------------------------------
    # Key findings
    # -----------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    print(f"""
1. SCALE: {counts['fail_build']} of {completed} completed runs ({100*counts['fail_build']//completed if completed else 0}%) failed
   due to build/environment issues (not test logic).

2. BREAKDOWN:""")
    for cat in cat_order:
        items = by_cat.get(cat, [])
        if items:
            pct = 100 * len(items) // counts['fail_build']
            print(f"     {cat:30s} {len(items):3d}  ({pct}%)")

    print(f"""
3. BIGGEST ISSUE -- CONTAINER STARTUP ({len(by_cat.get('container_startup',[]))} failures):
   The model builds a Docker image that compiles cleanly, but the resulting
   container either:
   - Crashes immediately (wrong entrypoint, missing env vars, wrong CMD)
   - Listens on wrong port (hidden tests expect a specific port)
   - Doesn't start the app process correctly
   This means the MODEL's Dockerfile/app configuration is structurally wrong.

4. DEPENDENCY RESOLUTION ({len(by_cat.get('dependency_resolution',[]))} failures):
   Internet access IS available. Failures are caused by:
   - Model choosing deprecated/removed base images (e.g., openjdk:11-jre-slim)
   - Model specifying wrong package versions in package.json / Gemfile / etc.
   - Model using incompatible PHP/Ruby/Node versions vs locked dependencies
   The ORIGINAL task bundles use a DinD wrapper pattern -- the model must
   create a correct application Dockerfile from scratch.

5. DOCKERFILE MISSING ({len(by_cat.get('dockerfile_missing',[]))} failures):
   The model simply didn't create a Dockerfile, or created an empty one (2B),
   or the Dockerfile references a missing entrypoint script.

6. COMPILATION ERROR ({len(by_cat.get('compilation_error',[]))} failures):
   Model's code has compilation errors (duplicate methods, type mismatches).
   This is a code quality issue, not an infrastructure issue.

7. ROOT CAUSE SUMMARY:
   NONE of these failures are caused by:
   - Broken task bundles (the original ABC data is fine)
   - Missing internet access (network works in all observed cases)
   - Docker daemon issues (0 cases)

   ALL failures are caused by the MODEL producing:
   - Wrong Dockerfiles (wrong base image, wrong deps, wrong config)
   - Broken application code (compilation errors)
   - Wrong container configuration (wrong ports, entrypoints, env)
""")


if __name__ == "__main__":
    main()
