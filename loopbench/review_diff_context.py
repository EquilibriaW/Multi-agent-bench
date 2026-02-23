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
        for entry in entries:
            rows.append(
                {
                    "coder": coder,
                    "sha": entry.get("sha"),
                    "subject": entry.get("subject"),
                    "files_changed": entry.get("files_changed"),
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
