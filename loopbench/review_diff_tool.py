#!/usr/bin/env python3
"""
Lightweight diff inspector for planner review rounds.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def main() -> int:
    parser = argparse.ArgumentParser(description="LoopBench reviewer diff inspector")
    parser.add_argument("--manifest", required=True, help="Path to round manifest JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List candidate commits")
    for cmd in ("files", "show"):
        p = sub.add_parser(cmd, help=f"{cmd.title()} details for a commit")
        p.add_argument("--coder", required=True)
        p.add_argument("--sha", required=True)

    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    commits = data.get("commits", [])

    if args.cmd == "list":
        for row in _iter_rows(commits):
            files_changed = row.get("files_changed") or []
            file_count = len(files_changed) if isinstance(files_changed, list) else 0
            subject = str(row.get("subject") or "").replace("\n", " ").strip()
            print(f"{row['coder']}\t{row['sha']}\tfiles={file_count}\t{subject}")
        return 0

    row = _find_commit(commits, coder=args.coder, sha=args.sha)
    if row is None:
        print(f"commit not found for coder={args.coder} sha={args.sha}", file=sys.stderr)
        return 2

    if args.cmd == "files":
        for path in row.get("files_changed") or []:
            print(path)
        return 0

    sha = str(row.get("sha") or "").strip()
    # Extract patch on-demand via git — avoids pre-materializing large
    # diffs for every candidate commit up-front.
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
            cwd=str(manifest_path.parent),
        )
        if top.returncode == 0:
            worktree_root = top.stdout.strip()
            patch_proc = subprocess.run(
                ["git", "-C", worktree_root, "show",
                 "--pretty=format:", "--no-color", "--unified=1", sha],
                capture_output=True, text=True, timeout=25,
            )
            if patch_proc.returncode == 0 and patch_proc.stdout.strip():
                sys.stdout.write(patch_proc.stdout)
                return 0
    except Exception:  # noqa: BLE001
        pass

    # Fall back to pre-materialized patch file (legacy path).
    patch_rel = row.get("patch_path")
    if not isinstance(patch_rel, str) or not patch_rel.strip():
        print(f"could not extract patch for {sha}", file=sys.stderr)
        return 2
    patch_path = manifest_path.parent / patch_rel
    if not patch_path.exists():
        print(f"patch file missing: {patch_path}", file=sys.stderr)
        return 2
    sys.stdout.write(patch_path.read_text(encoding="utf-8", errors="replace"))
    return 0


def _iter_rows(commits: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    for row in commits:
        if not isinstance(row, dict):
            continue
        coder = str(row.get("coder") or "").strip()
        sha = str(row.get("sha") or "").strip()
        if not coder or not sha:
            continue
        yield row


def _find_commit(commits: Iterable[Any], *, coder: str, sha: str) -> Optional[Dict[str, Any]]:
    coder = str(coder or "").strip()
    sha = str(sha or "").strip()
    for row in _iter_rows(commits):
        row_coder = str(row.get("coder") or "").strip()
        row_sha = str(row.get("sha") or "").strip()
        if row_coder != coder:
            continue
        if row_sha == sha or row_sha.startswith(sha):
            return row
    return None


if __name__ == "__main__":
    raise SystemExit(main())
