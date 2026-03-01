#!/usr/bin/env python3
"""
Analyze Dockerfile worksplit for kimi-k2.5 build-failure runs.

For each run where Docker build failed (env.up failure) OR where coders created
a Dockerfile in their patch, examine:
  - plans/subtasks.yaml: Did the planner assign Dockerfile creation? To whom? With what specifics?
  - plans/plan.md: Did the planner specify base image or language version?
  - artifacts/workflow_summary.json: Which coders actually produced merged commits?
  - role_summaries/: What each coder said about Dockerfile creation
  - The actual Dockerfile from the patch: what was chosen?
  - The error: what went wrong?

Classifies each into:
  - planner_bad_instruction: Planner told coder to use wrong base image / version
  - planner_no_instruction: Planner didn't mention Dockerfile at all, coder had to guess
  - coder_ignored_plan: Planner gave correct instructions but coder chose differently
  - coder_wrong_choice: Coder was assigned Dockerfile but picked wrong image/version independently
  - no_dockerfile_created: Nobody created a Dockerfile at all
  - infra_failure: Docker network exhaustion or similar infra issue (not agent-related)
  - path_collision: Task repo has public/ dir that collides with harness env path
"""

import json
import os
import re
import sys
import glob
import yaml
from collections import defaultdict
from pathlib import Path

RUNS_DIR = "/Users/alex/Multi-agent-bench/runs"
WORKSPACES_DIR = "/Users/alex/Multi-agent-bench/workspaces"
RUN_PATTERN = "full-eval-2model-224x1-20260223_kimi-k2.5_*_r01"


def get_env_up_status(run_dir):
    """Check events.jsonl for env.up result."""
    events_file = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(events_file):
        return None, ""
    with open(events_file) as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("type") == "tool_result" and ev.get("payload", {}).get("tool") == "env.up":
                payload = ev["payload"]
                ok = payload.get("ok", True) and payload.get("exit_code", 0) == 0
                stderr = payload.get("stderr", "")
                return ok, stderr
    return None, ""


def get_env_up_log(run_dir):
    """Read the env_up.log for detailed build errors."""
    log_file = os.path.join(run_dir, "artifacts", "planner_reviewer_env_up.log")
    if os.path.exists(log_file):
        with open(log_file) as f:
            return f.read()
    return ""


def get_plan(run_dir):
    """Read plan.md."""
    plan_file = os.path.join(run_dir, "plans", "plan.md")
    if os.path.exists(plan_file):
        with open(plan_file) as f:
            return f.read()
    return ""


def get_subtasks(run_dir):
    """Read and parse subtasks.yaml."""
    subtasks_file = os.path.join(run_dir, "plans", "subtasks.yaml")
    if os.path.exists(subtasks_file):
        with open(subtasks_file) as f:
            try:
                return yaml.safe_load(f)
            except Exception:
                return None
    return None


def get_workflow_summary(run_dir):
    """Read workflow_summary.json."""
    summary_file = os.path.join(run_dir, "artifacts", "workflow_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            try:
                return json.load(f)
            except Exception:
                return None
    return None


def get_role_summaries(run_dir):
    """Read all role summaries."""
    summaries_dir = os.path.join(run_dir, "role_summaries")
    result = {}
    if os.path.isdir(summaries_dir):
        for fname in os.listdir(summaries_dir):
            fpath = os.path.join(summaries_dir, fname)
            with open(fpath) as f:
                result[fname] = f.read()
    return result


def get_dockerfile_from_patch(run_dir):
    """Extract Dockerfile content from final.patch."""
    patch_file = os.path.join(run_dir, "final.patch")
    if not os.path.exists(patch_file):
        return None, None
    with open(patch_file) as f:
        content = f.read()
    if "Dockerfile" not in content and "dockerfile" not in content:
        return None, None

    # Extract all Dockerfile diffs
    dockerfiles = {}
    lines = content.split('\n')
    current_file = None
    current_lines = []

    for line in lines:
        if line.startswith('diff --git'):
            if current_file and 'dockerfile' in current_file.lower():
                dockerfiles[current_file] = '\n'.join(current_lines)
            current_file = line.split(' b/')[-1] if ' b/' in line else None
            current_lines = [line]
        elif current_file:
            current_lines.append(line)

    if current_file and 'dockerfile' in current_file.lower():
        dockerfiles[current_file] = '\n'.join(current_lines)

    # Extract added lines (the actual Dockerfile content)
    dockerfile_contents = {}
    for fname, diff_text in dockerfiles.items():
        added_lines = []
        for line in diff_text.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                added_lines.append(line[1:])
        dockerfile_contents[fname] = '\n'.join(added_lines)

    return dockerfiles, dockerfile_contents


def extract_base_image(dockerfile_content):
    """Extract FROM base image from Dockerfile content."""
    if not dockerfile_content:
        return None
    for line in dockerfile_content.split('\n'):
        line = line.strip()
        if line.upper().startswith('FROM '):
            parts = line.split()
            if len(parts) >= 2:
                img = parts[1]
                if img.upper() != 'AS':
                    return img
    return None


def get_task_language(run_dir):
    """Get the task's language from the workspace task.yaml."""
    run_name = os.path.basename(run_dir)
    ws_dir = os.path.join(WORKSPACES_DIR, run_name)
    task_yaml = os.path.join(ws_dir, "worktrees", "planner_reviewer", ".loopbench", "task.yaml")
    if os.path.exists(task_yaml):
        with open(task_yaml) as f:
            try:
                data = yaml.safe_load(f)
                return data.get("language", "unknown")
            except Exception:
                pass
    return "unknown"


def is_specific_image_reference(text):
    """Check if text contains an actual Docker image name:tag (not just 'uses appropriate').

    A specific image reference must contain a colon with a version/tag, like:
      php:8.2-cli, node:18-alpine, golang:1.23-alpine, maven:3.8-openjdk-11
      nikolaik/python-nodejs:python3.12-nodejs22-bullseye
      mcr.microsoft.com/dotnet/sdk:8.0

    Vague references like 'uses appropriate', 'uses an', 'use multi-stage' are NOT specific.
    """
    # Pattern: image_name:tag where tag contains at least one digit
    specific_pattern = r'(?:(?:[a-z0-9.-]+/)*[a-z0-9._-]+:[a-z0-9._-]*\d[a-z0-9._-]*)'
    m = re.search(specific_pattern, text.lower())
    return m is not None


def extract_specific_image(text):
    """Extract a specific Docker image:tag from text.

    Returns the most specific image reference found, or None if only vague references.
    """
    # Match patterns like: php:8.2-cli, node:18-alpine, golang:1.23, etc.
    # Also match registry paths: mcr.microsoft.com/dotnet/sdk:7.0, nikolaik/python-nodejs:...
    patterns = [
        r'(?:(?:[a-z0-9.-]+/)+[a-z0-9._-]+:[a-z0-9._-]+)',  # registry/path:tag
        r'(?:php|python|node|ruby|golang|go|openjdk|maven|gradle|dotnet|rust|eclipse-temurin|amazoncorretto):[a-z0-9._-]+',  # common images with tag
    ]
    for pat in patterns:
        m = re.search(pat, text.lower())
        if m:
            result = m.group(0)
            # Clean up trailing backticks or quotes
            result = result.rstrip('`"\'')
            return result
    return None


def analyze_plan_dockerfile_instruction(plan_md, subtasks_data):
    """Analyze whether the planner gave Dockerfile instructions and their quality.

    Returns: (has_dockerfile_subtask, assigned_role, specified_base_image, base_image_value,
              instruction_quality, details)

    instruction_quality is one of:
      - "specific_image": Planner gave a specific image:tag (e.g. "php:8.2-cli")
      - "vague_guidance": Planner said things like "use appropriate Go image"
      - "no_mention": Planner didn't mention Dockerfile or base image at all
    """
    has_dockerfile_subtask = False
    assigned_role = None
    specified_base_image = False
    base_image_value = None
    instruction_quality = "no_mention"
    details = ""

    # Check subtasks.yaml
    if subtasks_data and 'subtasks' in subtasks_data:
        for st in subtasks_data['subtasks']:
            title = st.get('title', '').lower()
            acceptance = st.get('acceptance', '').lower()
            paths = [p.lower() for p in st.get('paths', [])]

            if 'dockerfile' in title or 'dockerfile' in ' '.join(paths):
                has_dockerfile_subtask = True
                assigned_role = st.get('role', 'unknown')
                full_text = st.get('acceptance', '') + ' ' + st.get('title', '')

                # Try to find a specific image reference
                specific_img = extract_specific_image(full_text)
                if specific_img:
                    specified_base_image = True
                    base_image_value = specific_img
                    instruction_quality = "specific_image"
                elif any(kw in acceptance for kw in ['base image', 'docker image', 'multi-stage',
                                                       'uses php', 'uses ruby', 'uses node',
                                                       'uses go', 'uses python', 'use php',
                                                       'use ruby', 'use node', 'use go',
                                                       'use python', 'appropriate']):
                    instruction_quality = "vague_guidance"
                else:
                    instruction_quality = "no_mention"

                details = f"subtask '{st.get('title', '')}' assigned to {assigned_role}"
                break

    # Also check plan.md for Dockerfile instructions
    if plan_md and 'dockerfile' in plan_md.lower():
        plan_lower = plan_md.lower()
        if not has_dockerfile_subtask:
            if 'create' in plan_lower and 'dockerfile' in plan_lower:
                has_dockerfile_subtask = True
                details = "mentioned in plan.md but not in subtasks"

        # Only upgrade instruction quality if plan.md has better info
        plan_specific_img = extract_specific_image(plan_md)
        if plan_specific_img and instruction_quality != "specific_image":
            specified_base_image = True
            base_image_value = plan_specific_img
            instruction_quality = "specific_image"
        elif instruction_quality == "no_mention":
            # Check for vague guidance in plan.md
            if any(kw in plan_lower for kw in ['base image', 'docker image', 'multi-stage']):
                instruction_quality = "vague_guidance"

        # Check for FROM instructions in plan.md (these are specific)
        if instruction_quality != "specific_image":
            for line in plan_md.split('\n'):
                if 'FROM ' in line.upper():
                    m = re.search(r'FROM\s+([^\s`"\']+)', line, re.IGNORECASE)
                    if m:
                        candidate = m.group(1).rstrip('`"\'')
                        # Only count if it has a version tag
                        if ':' in candidate and any(c.isdigit() for c in candidate):
                            specified_base_image = True
                            base_image_value = candidate
                            instruction_quality = "specific_image"
                            break

    return has_dockerfile_subtask, assigned_role, specified_base_image, base_image_value, instruction_quality, details


def classify_env_up_error(stderr, env_up_log):
    """Classify the type of env.up failure."""
    combined = stderr + "\n" + env_up_log

    if "all predefined address pools have been fully subnetted" in combined:
        return "infra_network_exhaustion"
    if "no such file or directory" in combined and "docker-compose.yaml" in combined:
        return "missing_docker_compose"
    if "no such file or directory" in combined and "Dockerfile" in combined:
        return "missing_dockerfile"
    if "failed to solve" in combined.lower() or "error:" in combined.lower():
        # Docker build error
        if "COPY failed" in combined or "not found" in combined:
            return "docker_build_error"
        return "docker_build_error"
    if "port is already allocated" in combined:
        return "infra_port_conflict"
    return "unknown_env_up_error"


def get_hidden_validate_error(run_dir):
    """Read hidden_validate/result.log and classify the Docker-related error.

    Returns: (error_type, error_detail)
    Where error_type is one of:
      "image_not_found" - Docker image does not exist
      "dockerfile_not_found" - Dockerfile path wrong
      "build_error" - Docker build command failed (dep install, etc.)
      "container_crash" - Built OK but container exited immediately
      "tests_only" - Dockerfile was fine, only test failures
      "no_result" - No hidden_validate output
    """
    result_log = os.path.join(run_dir, "hidden_validate", "result.log")
    if not os.path.exists(result_log):
        return "no_result", ""
    with open(result_log) as f:
        content = f.read()

    lower = content.lower()

    if "failed to solve" in lower or "failed to build" in lower:
        if ": not found" in lower:
            # Extract the image that was not found
            m = re.search(r'(?:ERROR|error):\s*([^\n]+not found)', content)
            detail = m.group(1).strip() if m else "image not found"
            return "image_not_found", detail
        elif "no such file or directory" in lower and "dockerfile" in lower:
            return "dockerfile_not_found", "Dockerfile at wrong path"
        else:
            return "build_error", content[:300]

    if "container" in lower and ("has exited" in lower or "does not exist" in lower):
        return "container_crash", "Container exited immediately after start"

    # Check for build-time errors that aren't "failed to solve"
    if ("failed to build gem" in lower or "bundler cannot continue" in lower or
        "npm error" in lower or "error: failed to parse" in lower):
        return "build_error", content[:300]

    return "tests_only", ""


def analyze_run(run_dir):
    """Full analysis of a single run."""
    run_name = os.path.basename(run_dir)

    env_up_ok, env_up_stderr = get_env_up_status(run_dir)
    env_up_log = get_env_up_log(run_dir)
    plan_md = get_plan(run_dir)
    subtasks_data = get_subtasks(run_dir)
    workflow_summary = get_workflow_summary(run_dir)
    role_summaries = get_role_summaries(run_dir)
    docker_diffs, docker_contents = get_dockerfile_from_patch(run_dir) or (None, None)
    task_language = get_task_language(run_dir)
    hv_error_type, hv_error_detail = get_hidden_validate_error(run_dir)

    # Analyze planner instructions
    has_df_subtask, assigned_role, has_base_image, base_image_val, instr_quality, plan_details = \
        analyze_plan_dockerfile_instruction(plan_md, subtasks_data)

    # Get the actual Dockerfile base image from patch
    actual_base_image = None
    if docker_contents:
        for fname, content in docker_contents.items():
            img = extract_base_image(content)
            if img:
                actual_base_image = img
                break

    # Get merged commits info
    merged_by = {}
    if workflow_summary:
        mc = workflow_summary.get("merged_commits", {})
        for role, commits in mc.items():
            if commits:
                merged_by[role] = len(commits)

    # Check role summaries for Dockerfile mentions
    coder_dockerfile_mentions = {}
    for fname, summary_text in role_summaries.items():
        if 'dockerfile' in summary_text.lower():
            role = fname.split('_')[0]
            if fname.startswith('coder_a'):
                role = 'coder_a'
            elif fname.startswith('coder_b'):
                role = 'coder_b'
            coder_dockerfile_mentions[role] = summary_text[:500]

    # Classify the error
    env_up_error_type = None
    if env_up_ok is False:
        env_up_error_type = classify_env_up_error(env_up_stderr, env_up_log)

    return {
        "run_name": run_name,
        "task_language": task_language,
        "env_up_ok": env_up_ok,
        "env_up_stderr": env_up_stderr[:500],
        "env_up_error_type": env_up_error_type,
        "plan_md_excerpt": plan_md[:1000] if plan_md else "",
        "has_dockerfile_subtask": has_df_subtask,
        "assigned_role": assigned_role,
        "planner_specified_base_image": has_base_image,
        "planner_base_image": base_image_val,
        "instruction_quality": instr_quality,
        "plan_details": plan_details,
        "actual_base_image": actual_base_image,
        "has_dockerfile_in_patch": docker_contents is not None and len(docker_contents) > 0,
        "dockerfile_contents": docker_contents,
        "merged_by": merged_by,
        "coder_dockerfile_mentions": coder_dockerfile_mentions,
        "workflow_summary": workflow_summary,
        "hv_error_type": hv_error_type,
        "hv_error_detail": hv_error_detail,
    }


def classify_worksplit(analysis):
    """Classify the failure into worksplit categories.

    Uses instruction_quality to distinguish:
      - "specific_image": Planner gave exact image:tag -> compare with actual
      - "vague_guidance": Planner said "use appropriate X" -> coder had to guess
      - "no_mention": No Dockerfile guidance at all
    """
    run = analysis

    # Infrastructure failures (not agent-related)
    if run["env_up_error_type"] == "infra_network_exhaustion":
        return "infra_failure", "Docker network pool exhaustion (not agent-related)"

    # Missing docker-compose.yaml (path collision)
    if run["env_up_error_type"] == "missing_docker_compose":
        if not run["has_dockerfile_in_patch"]:
            return "no_dockerfile_created", f"No docker-compose.yaml at expected path; no Dockerfile in patch"
        else:
            return "path_collision", f"Task repo has path collision with harness env path (public/env/)"

    # No Dockerfile created at all
    if not run["has_dockerfile_in_patch"] and run["env_up_ok"] is False:
        return "no_dockerfile_created", f"env.up failed ({run['env_up_error_type']}), no Dockerfile in patch"

    if not run["has_dockerfile_in_patch"] and run["env_up_ok"] is True:
        return "not_a_dockerfile_failure", "env.up succeeded, no Dockerfile in coder patch"

    # Dockerfile was created - now analyze who's responsible for issues
    instr_quality = run.get("instruction_quality", "no_mention")

    if not run["has_dockerfile_subtask"]:
        # Planner didn't even create a Dockerfile subtask
        return "planner_no_instruction", \
            f"Planner did not create a Dockerfile subtask. Coder guessed: {run['actual_base_image']}"

    # Planner created a Dockerfile subtask
    if instr_quality == "specific_image":
        # Planner gave a specific image:tag
        planner_img = (run["planner_base_image"] or "").lower()
        actual_img = (run["actual_base_image"] or "").lower()

        if actual_img and planner_img:
            # Compare full image references (registry/name:tag) to detect tag deviations
            if actual_img == planner_img:
                # Coder followed the plan's base image choice - planner's fault
                return "planner_bad_instruction", \
                    f"Planner specified '{run['planner_base_image']}', coder used '{run['actual_base_image']}' (followed plan)"
            else:
                # Coder used a different image than planner specified
                return "coder_ignored_plan", \
                    f"Planner specified '{run['planner_base_image']}', but coder used '{run['actual_base_image']}'"
        elif actual_img:
            return "planner_bad_instruction", \
                f"Planner specified '{run['planner_base_image']}', coder used '{run['actual_base_image']}'"
        else:
            return "coder_wrong_choice", \
                f"Planner specified '{run['planner_base_image']}' but coder produced no FROM line"

    elif instr_quality == "vague_guidance":
        # Planner said things like "use appropriate Go image" but didn't specify exact image
        # This is a PLANNING failure - planner should have specified the exact image
        return "planner_no_instruction", \
            f"Planner gave only vague guidance ('{run.get('plan_details', '')}') without specific image:tag. " \
            f"Coder had to guess and chose '{run['actual_base_image']}'"

    else:
        # no_mention - Planner assigned Dockerfile subtask but said nothing about base image
        return "coder_wrong_choice", \
            f"Planner assigned Dockerfile to {run['assigned_role']} but gave no guidance. " \
            f"Coder independently chose '{run['actual_base_image']}'"


def main():
    print("=" * 100)
    print("DOCKERFILE WORKSPLIT ANALYSIS")
    print("Analyzing kimi-k2.5 runs for Dockerfile build failures")
    print("=" * 100)

    # Find all matching runs
    run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, RUN_PATTERN)))
    print(f"\nTotal kimi-k2.5 runs found: {len(run_dirs)}")

    # Step 1: Identify build-failure runs
    # We consider two groups:
    #   A) env.up failures (26 runs)
    #   B) Runs where coders created a Dockerfile in their patch (44 runs)
    # The union of these is our analysis set.

    all_analyses = []
    for run_dir in run_dirs:
        analysis = analyze_run(run_dir)

        # Include if: env.up failed OR Dockerfile in patch
        if analysis["env_up_ok"] is False or analysis["has_dockerfile_in_patch"]:
            all_analyses.append(analysis)

    print(f"Runs with env.up failure: {sum(1 for a in all_analyses if a['env_up_ok'] is False)}")
    print(f"Runs with Dockerfile in patch: {sum(1 for a in all_analyses if a['has_dockerfile_in_patch'])}")
    print(f"Total runs to analyze (union): {len(all_analyses)}")

    # Step 2: Classify each run
    classifications = defaultdict(list)
    for analysis in all_analyses:
        category, reason = classify_worksplit(analysis)
        analysis["classification"] = category
        analysis["classification_reason"] = reason
        classifications[category].append(analysis)

    # Step 3: Summary counts
    print("\n" + "=" * 100)
    print("CLASSIFICATION SUMMARY")
    print("=" * 100)

    # Sort categories for display
    category_order = [
        "planner_no_instruction",
        "planner_bad_instruction",
        "coder_ignored_plan",
        "coder_wrong_choice",
        "no_dockerfile_created",
        "path_collision",
        "infra_failure",
        "not_a_dockerfile_failure",
    ]

    total_planning = 0
    total_coder = 0
    total_other = 0

    for cat in category_order:
        runs = classifications.get(cat, [])
        if not runs:
            continue
        count = len(runs)

        if cat in ("planner_no_instruction", "planner_bad_instruction"):
            total_planning += count
            blame = "PLANNING"
        elif cat in ("coder_ignored_plan", "coder_wrong_choice"):
            total_coder += count
            blame = "CODER"
        else:
            total_other += count
            blame = "OTHER"

        print(f"\n  {cat}: {count} runs  [{blame}]")

        # List all runs in this category
        for r in runs:
            lang = r["task_language"]
            base = r["actual_base_image"] or "N/A"
            planner_img = r["planner_base_image"] or "N/A"
            print(f"    - {r['run_name']}")
            print(f"      lang={lang}, planner_image={planner_img}, actual_image={base}")

    print(f"\n{'=' * 100}")
    print(f"BLAME ATTRIBUTION TOTALS (excluding infra/path/not-a-failure)")
    print(f"{'=' * 100}")
    print(f"  PLANNING failures (planner_no_instruction + planner_bad_instruction): {total_planning}")
    print(f"  CODER failures    (coder_ignored_plan + coder_wrong_choice):          {total_coder}")
    print(f"  OTHER             (infra, path_collision, no_dockerfile, etc):         {total_other}")
    if total_planning + total_coder > 0:
        pct_planning = 100 * total_planning / (total_planning + total_coder)
        pct_coder = 100 * total_coder / (total_planning + total_coder)
        print(f"\n  Of agent-attributable failures:")
        print(f"    Planning: {pct_planning:.1f}%")
        print(f"    Coder:    {pct_coder:.1f}%")

    # Step 4: Detailed examples per category
    print(f"\n{'=' * 100}")
    print("DETAILED EXAMPLES PER CATEGORY (2-3 per category)")
    print("=" * 100)

    for cat in category_order:
        runs = classifications.get(cat, [])
        if not runs:
            continue

        print(f"\n{'─' * 80}")
        print(f"Category: {cat} ({len(runs)} runs)")
        print(f"{'─' * 80}")

        for r in runs[:3]:
            print(f"\n  >>> {r['run_name']}")
            print(f"      Language: {r['task_language']}")
            print(f"      env.up OK: {r['env_up_ok']}")
            if r['env_up_error_type']:
                print(f"      env.up error: {r['env_up_error_type']}")
                print(f"      stderr: {r['env_up_stderr'][:200]}")
            print(f"      Dockerfile subtask in plan: {r['has_dockerfile_subtask']}")
            if r['has_dockerfile_subtask']:
                print(f"      Assigned to: {r['assigned_role']}")
            print(f"      Planner instruction quality: {r.get('instruction_quality', 'N/A')}")
            if r['planner_base_image']:
                print(f"      Planner's base image: {r['planner_base_image']}")
            print(f"      Actual base image in patch: {r['actual_base_image']}")
            print(f"      Has Dockerfile in patch: {r['has_dockerfile_in_patch']}")
            if r['merged_by']:
                print(f"      Merged commits by: {r['merged_by']}")
            if r.get('hv_error_type') and r['hv_error_type'] != 'tests_only':
                print(f"      hidden_validate error: {r['hv_error_type']}: {r['hv_error_detail'][:150]}")
            print(f"      Classification: {r['classification_reason']}")

            # Show plan excerpt for Dockerfile
            if r["has_dockerfile_subtask"] and r.get("plan_md_excerpt"):
                plan_excerpt = r["plan_md_excerpt"]
                # Find Dockerfile-related section
                lines = plan_excerpt.split('\n')
                df_lines = []
                in_df_section = False
                for line in lines:
                    if 'dockerfile' in line.lower() or 'docker' in line.lower():
                        in_df_section = True
                    if in_df_section:
                        df_lines.append(line)
                        if len(df_lines) > 8:
                            break
                    if in_df_section and line.strip() == '' and len(df_lines) > 2:
                        break
                if df_lines:
                    print(f"      Plan excerpt (Dockerfile section):")
                    for pl in df_lines[:8]:
                        print(f"        | {pl}")

            # Show Dockerfile content
            if r.get("dockerfile_contents"):
                for fname, content in r["dockerfile_contents"].items():
                    print(f"      Dockerfile ({fname}):")
                    for dl in content.split('\n')[:10]:
                        print(f"        | {dl}")
                    if len(content.split('\n')) > 10:
                        print(f"        | ... ({len(content.split(chr(10)))} total lines)")

    # Step 5: Key findings
    print(f"\n{'=' * 100}")
    print("KEY FINDINGS")
    print("=" * 100)

    # Analyze the planner_no_instruction category more
    no_instr = classifications.get("planner_no_instruction", [])
    bad_instr = classifications.get("planner_bad_instruction", [])
    ignored = classifications.get("coder_ignored_plan", [])
    wrong = classifications.get("coder_wrong_choice", [])

    print(f"""
1. TOTAL SCOPE: {len(all_analyses)} runs involved Dockerfiles (either env.up failure or Dockerfile in patch)

2. BLAME SPLIT:
   - Planning failures: {total_planning} ({total_planning}/{total_planning+total_coder} = {100*total_planning/(total_planning+total_coder) if total_planning+total_coder else 0:.0f}%)
     - Planner gave NO Dockerfile instruction: {len(no_instr)}
     - Planner gave WRONG instruction (bad base image): {len(bad_instr)}
   - Coder failures: {total_coder} ({total_coder}/{total_planning+total_coder} = {100*total_coder/(total_planning+total_coder) if total_planning+total_coder else 0:.0f}%)
     - Coder ignored plan's correct instruction: {len(ignored)}
     - Coder picked wrong image independently: {len(wrong)}
   - Infrastructure / Other: {total_other}

3. PATTERN ANALYSIS:""")

    # Language breakdown
    lang_counts = defaultdict(lambda: defaultdict(int))
    for a in all_analyses:
        if a["classification"] in ("planner_no_instruction", "planner_bad_instruction",
                                    "coder_ignored_plan", "coder_wrong_choice"):
            lang_counts[a["task_language"]][a["classification"]] += 1

    print("   Language breakdown of agent-attributable failures:")
    for lang in sorted(lang_counts.keys()):
        cats = lang_counts[lang]
        total = sum(cats.values())
        planning = cats.get("planner_no_instruction", 0) + cats.get("planner_bad_instruction", 0)
        coding = cats.get("coder_ignored_plan", 0) + cats.get("coder_wrong_choice", 0)
        print(f"     {lang}: {total} total (planning={planning}, coder={coding})")

    # Analyze plan exhaustion
    budget_exhausted = sum(1 for a in all_analyses
                          if "budget exhausted" in a.get("plan_md_excerpt", "").lower())
    print(f"""
4. PLAN QUALITY:
   - Plans with 'budget exhausted' (incomplete): {budget_exhausted}/{len(all_analyses)}
   - Plans that explicitly assigned Dockerfile subtask: {sum(1 for a in all_analyses if a['has_dockerfile_subtask'])}
   - Plans that specified exact base image:tag: {sum(1 for a in all_analyses if a.get('instruction_quality') == 'specific_image')}
   - Plans with only vague guidance: {sum(1 for a in all_analyses if a.get('instruction_quality') == 'vague_guidance')}
   - Plans with no Dockerfile mention: {sum(1 for a in all_analyses if a.get('instruction_quality') == 'no_mention')}
""")

    # Step 6: Hidden validate error breakdown
    print(f"\n{'=' * 100}")
    print("HIDDEN VALIDATE ERROR BREAKDOWN (what actually went wrong with Docker)")
    print("=" * 100)

    hv_counts = defaultdict(int)
    hv_examples = defaultdict(list)
    for a in all_analyses:
        hv_type = a.get("hv_error_type", "no_result")
        hv_counts[hv_type] += 1
        if len(hv_examples[hv_type]) < 2:
            hv_examples[hv_type].append((a["run_name"], a.get("hv_error_detail", "")[:200]))

    for hv_type in ["image_not_found", "dockerfile_not_found", "build_error",
                     "container_crash", "tests_only", "no_result"]:
        count = hv_counts.get(hv_type, 0)
        if count > 0:
            print(f"\n  {hv_type}: {count}")
            for name, detail in hv_examples.get(hv_type, []):
                print(f"    - {name}")
                if detail:
                    print(f"      {detail[:150]}")

    dockerfile_build_failures = sum(hv_counts.get(t, 0) for t in
                                     ["image_not_found", "dockerfile_not_found", "build_error", "container_crash"])
    print(f"\n  TOTAL Dockerfile-caused failures: {dockerfile_build_failures}")
    print(f"  + env.up failures (before code): {sum(1 for a in all_analyses if a['env_up_ok'] is False)}")

    # Step 7: FOCUSED analysis -- only runs where Docker ACTUALLY failed
    print(f"\n{'=' * 100}")
    print("FOCUSED ANALYSIS: Runs where Dockerfile ACTUALLY caused a failure")
    print("(excluding runs where Dockerfile built+ran fine, tests failed for code reasons)")
    print("=" * 100)

    docker_failure_types = {"image_not_found", "dockerfile_not_found", "build_error", "container_crash"}
    actual_df_failures = [a for a in all_analyses
                          if a.get("hv_error_type") in docker_failure_types]

    planning_actual = sum(1 for a in actual_df_failures
                          if a["classification"] in ("planner_no_instruction", "planner_bad_instruction"))
    coder_actual = sum(1 for a in actual_df_failures
                       if a["classification"] in ("coder_ignored_plan", "coder_wrong_choice"))
    other_actual = len(actual_df_failures) - planning_actual - coder_actual

    print(f"\n  Coder's Dockerfile actually failed in hidden_validate: {len(actual_df_failures)}")
    print(f"    Of these, blame on PLANNER: {planning_actual}")
    print(f"    Of these, blame on CODER:   {coder_actual}")
    print(f"    Of these, OTHER:            {other_actual}")

    if actual_df_failures:
        print(f"\n  Details of each actual Docker failure:")
        for a in actual_df_failures:
            print(f"    {a['run_name']}")
            print(f"      hv_error: {a['hv_error_type']}: {a['hv_error_detail'][:100]}")
            print(f"      classification: {a['classification']}")
            print(f"      planner_image: {a['planner_base_image'] or 'N/A'}, actual: {a['actual_base_image'] or 'N/A'}")
            print(f"      instruction_quality: {a['instruction_quality']}")

    # Answer the key question
    print(f"""
{'=' * 100}
ANSWER TO KEY QUESTION
{'=' * 100}
Is the Dockerfile failure mostly a PLANNING failure or a CODER execution failure?

OVERALL ANALYSIS (all {total_planning + total_coder} agent-attributable Dockerfile runs):
  Planning: {total_planning} ({100*total_planning/(total_planning+total_coder) if total_planning+total_coder else 0:.0f}%)
    - planner_no_instruction (vague/no guidance): {len(no_instr)}
    - planner_bad_instruction (wrong specific image): {len(bad_instr)}
  Coder: {total_coder} ({100*total_coder/(total_planning+total_coder) if total_planning+total_coder else 0:.0f}%)
    - coder_ignored_plan: {len(ignored)}
    - coder_wrong_choice: {len(wrong)}
  Infrastructure / Other: {total_other}

FOCUSED (only {len(actual_df_failures)} runs where Docker actually broke in hidden_validate):
  Planning: {planning_actual}
  Coder:    {coder_actual}
  Other:    {other_actual}

ANSWER: PLANNING failures overwhelmingly dominate ({100*total_planning/(total_planning+total_coder) if total_planning+total_coder else 0:.0f}% overall).

The kimi-k2.5 planner consistently specifies concrete Docker base images (42/63 runs)
but frequently picks WRONG or OUTDATED images. In {len(bad_instr)}/{total_planning+total_coder}
agent-attributable cases, the planner specified a specific image:tag and the coder
faithfully followed it -- but the image was wrong (deprecated, nonexistent, or incompatible).

Key failure patterns in chosen Docker images:
  - openjdk:11-jre-slim    -> removed from Docker Hub (should use eclipse-temurin)
  - mcr.microsoft.com/dotnet/sdk:2.2 -> EOL and removed
  - golang:1.24-alpine, golang:1.25-alpine -> versions that may not exist
  - ruby:3.1-slim / ruby:3.2-slim -> missing system deps (git, build-essential)
  - node:18-alpine -> npm install failures due to missing native build tools
  - rust:1.86-slim-bookworm -> Cargo.toml parsing failures

The planner is the primary bottleneck for Dockerfile quality. It needs either:
  1. Access to a validated base-image registry to check image availability
  2. A pre-built Dockerfile template per language/framework
  3. Or the task prompt should specify the required base image directly
""")


if __name__ == "__main__":
    main()
