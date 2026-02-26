"""
loopbench.review_diff_context

Review diff artifact preparation helpers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .shell import run_command
from .time_utils import now_ms


def build_candidate_merge_commits(
    *,
    role_paths: Dict[str, Path],
    coders: Iterable[str],
    coder_commits_by_role: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}

    for coder in coders:
        commits = coder_commits_by_role.get(coder, [])
        if not commits:
            continue
        role_path = role_paths[coder]
        entries: List[Dict[str, Any]] = []
        for sha in commits[-8:]:
            # Single git call for subject + file list.  Full patches are
            # extracted on-demand by review_diff_tool.py when the planner
            # actually requests them, avoiding eager materialization of
            # potentially large diffs that may never be read.
            result = run_command(
                ["git", "-C", str(role_path), "show", "--format=%s", "--name-only", "--no-color", sha],
                timeout_sec=20,
            )
            subject = ""
            files_changed: List[str] = []
            if result.ok:
                lines = result.stdout.splitlines()
                subject = lines[0].strip() if lines else ""
                # git outputs: subject, blank line, then file names
                files_changed = [line.strip() for line in lines[1:] if line.strip()]
            entries.append(
                {
                    "sha": sha,
                    "subject": subject,
                    "files_changed": files_changed[:30],
                }
            )
        if entries:
            out[coder] = entries
    return out


def build_review_diff_tool_context(
    *,
    role_paths: Dict[str, Path],
    planner: str,
    round_index: int,
    candidate_merge_commits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    planner_path = role_paths[planner]
    tool_root = planner_path / ".loopbench" / "artifacts" / "review_diffs"
    tool_root.mkdir(parents=True, exist_ok=True)
    script_path = tool_root / "review_diff_tool.py"
    if not script_path.exists():
        template_path = Path(__file__).with_name("review_diff_tool.py")
        script_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")

    round_dir = tool_root / f"round_{round_index}"
    round_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = round_dir / "manifest.json"
    rows: List[Dict[str, Any]] = []
    for coder, entries in candidate_merge_commits.items():
        coder_path = role_paths.get(coder)
        for entry in entries:
            sha = entry.get("sha", "")
            patch_rel = None
            # Pre-materialize patch from the coder worktree so the
            # review_diff_tool can serve it even when the reviewer's
            # worktree doesn't have the commit objects (e.g. E2B sandboxes).
            if coder_path and sha:
                patch_result = run_command(
                    ["git", "-C", str(coder_path), "show",
                     "--pretty=format:", "--no-color", "--unified=3", sha],
                    timeout_sec=25,
                )
                if patch_result.ok and patch_result.stdout.strip():
                    patch_file = round_dir / f"{coder}_{sha[:12]}.patch"
                    patch_file.write_text(patch_result.stdout, encoding="utf-8")
                    patch_rel = patch_file.name
            rows.append(
                {
                    "coder": coder,
                    "sha": sha,
                    "subject": entry.get("subject"),
                    "files_changed": entry.get("files_changed"),
                    "patch_path": patch_rel,
                }
            )
    manifest = {
        "round_index": round_index,
        "generated_at_ms": now_ms(),
        "commits": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest_rel = f".loopbench/artifacts/review_diffs/round_{round_index}/manifest.json"
    command_prefix = f"python .loopbench/artifacts/review_diffs/review_diff_tool.py --manifest {manifest_rel}"
    return {
        "command_prefix": command_prefix,
        "manifest_path": manifest_rel,
        "commands": {
            "list": f"{command_prefix} list",
            "show": f"{command_prefix} show --coder <coder_a|coder_b> --sha <commit_sha>",
            "files": f"{command_prefix} files --coder <coder_a|coder_b> --sha <commit_sha>",
        },
    }


def extract_inline_diffs(
    *,
    role_paths: Dict[str, Path],
    coders: Iterable[str],
    candidate_merge_commits: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract full diff content for each coder commit for inlining in prompt.

    Returns: {coder: [{sha, subject, files_changed, diff_content}]}
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for coder in coders:
        entries = candidate_merge_commits.get(coder, [])
        if not entries:
            continue
        coder_path = role_paths.get(coder)
        if not coder_path:
            continue
        inline_entries: List[Dict[str, Any]] = []
        for entry in entries:
            sha = entry.get("sha", "")
            if not sha:
                continue
            diff_result = run_command(
                ["git", "-C", str(coder_path), "show",
                 "--pretty=format:", "--no-color", "--unified=3", sha],
                timeout_sec=25,
            )
            diff_content = diff_result.stdout.strip() if diff_result.ok else ""
            inline_entries.append({
                "sha": sha,
                "subject": entry.get("subject", ""),
                "files_changed": entry.get("files_changed", []),
                "diff_content": diff_content,
            })
        if inline_entries:
            out[coder] = inline_entries
    return out