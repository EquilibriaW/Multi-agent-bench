#!/usr/bin/env python3
"""Extract generated source files from loopbench runs into browsable file trees.

Applies each run's final.patch to a temp clone of the task's repo bundle,
then copies out changed files into a flat directory per model+task.

Usage:
    python scripts/extract_code.py runs/<run_dir>
    python scripts/extract_code.py runs/<run1> runs/<run2> ...
    python scripts/extract_code.py runs/experiment-*          # glob

Output:
    extracted/<model_slug>/<task_id>/
        _meta.json          # run metadata
        <source files>      # the actual generated code
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: extract_code.py <run_dir> [<run_dir> ...]", file=sys.stderr)
        return 1

    run_dirs = [Path(d) for d in sys.argv[1:]]
    output_root = Path("extracted")
    results: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        if not (run_dir / "manifest.json").is_file():
            print(f"skip: {run_dir.name} (no manifest.json)", file=sys.stderr)
            continue
        if not (run_dir / "final.patch").is_file():
            print(f"skip: {run_dir.name} (no final.patch)", file=sys.stderr)
            continue

        try:
            result = extract_run(run_dir, output_root)
            results.append(result)
        except Exception as exc:
            print(f"error: {run_dir.name}: {exc}", file=sys.stderr)

    if not results:
        print("no runs extracted", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"Extracted {len(results)} run(s) into {output_root}/\n")
    for r in results:
        status = []
        if r.get("hidden_pass"):
            status.append("hidden:PASS")
        elif r.get("hidden_pass") is False:
            status.append("hidden:FAIL")
        if r.get("public_pass"):
            status.append("public:PASS")
        elif r.get("public_pass") is False:
            status.append("public:FAIL")
        status_str = f"  [{', '.join(status)}]" if status else ""
        print(f"  {r['model_slug']}/{r['task_id']}/  ({len(r['files'])} files){status_str}")
        for f in r["files"]:
            print(f"    {f}")

    return 0


def extract_run(run_dir: Path, output_root: Path) -> Dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    task_id = manifest["task_id"]
    run_id = manifest["run_id"]

    model = _detect_model(run_dir)
    model_slug = _slugify(model)

    patch_text = (run_dir / "final.patch").read_text(encoding="utf-8")
    if not patch_text.strip():
        raise RuntimeError("final.patch is empty")

    # Find the task's repo bundle
    snapshot = _read_json(run_dir / "inputs" / "task_snapshot.json")
    bundle_path = _find_repo_bundle(snapshot, run_dir)
    base_commit = snapshot.get("base_commit") or manifest.get("base_commit")

    # Apply patch in a temp clone and extract files
    changed_files = _changed_files_from_patch(patch_text)
    deleted_files = _deleted_files_from_patch(patch_text)
    if not changed_files and not deleted_files:
        raise RuntimeError("no changed files in final.patch")

    out_dir = output_root / model_slug / task_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bundle_path and bundle_path.is_file():
        _extract_via_git(bundle_path, base_commit, patch_text, changed_files, out_dir)
    else:
        print(f"  warn: no repo bundle found, extracting from patch only", file=sys.stderr)
        _extract_from_patch_only(patch_text, changed_files, out_dir)

    # Write metadata
    all_affected = sorted(set(changed_files) | set(deleted_files))
    meta = {
        "run_id": run_id,
        "task_id": task_id,
        "model": model,
        "model_slug": model_slug,
        "public_pass": manifest.get("public_pass"),
        "hidden_pass": manifest.get("hidden_pass"),
        "files": sorted(changed_files),
        "deleted_files": sorted(deleted_files),
        "run_dir": str(run_dir),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"  ok: {model_slug}/{task_id} ({len(changed_files)} files)")
    return meta


def _extract_via_git(
    bundle_path: Path,
    base_commit: Optional[str],
    patch_text: str,
    changed_files: List[str],
    out_dir: Path,
) -> None:
    """Clone from bundle, apply patch, copy out changed files."""
    with tempfile.TemporaryDirectory(prefix="lb_extract_") as tmpdir:
        repo = Path(tmpdir) / "repo"
        # Clone from bundle
        subprocess.run(
            ["git", "clone", str(bundle_path), str(repo)],
            capture_output=True, check=True, timeout=30,
        )
        # Checkout base commit if specified
        if base_commit:
            subprocess.run(
                ["git", "-C", str(repo), "checkout", base_commit],
                capture_output=True, check=True, timeout=15,
            )
        # Apply the patch
        proc = subprocess.run(
            ["git", "-C", str(repo), "apply", "--allow-empty", "-"],
            input=patch_text, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            # Try with --3way as fallback
            proc = subprocess.run(
                ["git", "-C", str(repo), "apply", "--3way", "-"],
                input=patch_text, capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"git apply failed: {proc.stderr.strip()}")

        # Copy out changed files (including symlinks)
        for rel_path in changed_files:
            src = repo / rel_path
            dst = out_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_symlink():
                os.symlink(os.readlink(src), dst)
            elif src.is_file():
                shutil.copy2(src, dst)


def _extract_from_patch_only(
    patch_text: str,
    changed_files: List[str],
    out_dir: Path,
) -> None:
    """Fallback: reconstruct file contents from the unified diff.

    For new files (from /dev/null): full content.
    For modified files: the "after" lines from each hunk, gaps marked with «...».
    """
    deleted = set(_deleted_files_from_patch(patch_text))
    file_diffs = _split_patch_by_file(patch_text)
    for rel_path, diff_text in file_diffs.items():
        if rel_path in deleted:
            continue
        content = _reconstruct_after(diff_text)
        if not content and _is_rename_only(diff_text):
            continue  # rename-only without hunks; no content to reconstruct
        dst = out_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git a/.+ b/(.+)$")
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _changed_files_from_patch(patch_text: str) -> List[str]:
    """Return files that are added or modified (excludes deletions)."""
    files: List[str] = []
    for line in patch_text.splitlines():
        m = _DIFF_HEADER.match(line)
        if m:
            files.append(m.group(1))
    deleted = _deleted_files_from_patch(patch_text)
    deleted_set = set(deleted)
    return [f for f in files if f not in deleted_set]


def _deleted_files_from_patch(patch_text: str) -> List[str]:
    """Return files that are deleted by the patch."""
    deleted: List[str] = []
    lines = patch_text.splitlines()
    for i, line in enumerate(lines):
        if line == "+++ /dev/null" and i > 0 and lines[i - 1].startswith("--- a/"):
            deleted.append(lines[i - 1][6:])
    return deleted


_RENAME_RE = re.compile(r"^rename (from|to) ")


def _is_rename_only(diff_text: str) -> bool:
    """True if the diff is a rename with no hunks."""
    has_rename = False
    for line in diff_text.splitlines():
        if _RENAME_RE.match(line):
            has_rename = True
        if _HUNK_HEADER.match(line):
            return False  # has actual content changes
    return has_rename


def _split_patch_by_file(patch_text: str) -> Dict[str, str]:
    """Split a multi-file patch into per-file chunks."""
    result: Dict[str, str] = {}
    current_file: Optional[str] = None
    current_lines: List[str] = []

    for line in patch_text.splitlines():
        m = _DIFF_HEADER.match(line)
        if m:
            if current_file is not None:
                result[current_file] = "\n".join(current_lines)
            current_file = m.group(1)
            current_lines = [line]
        elif current_file is not None:
            current_lines.append(line)

    if current_file is not None:
        result[current_file] = "\n".join(current_lines)
    return result


def _reconstruct_after(diff_text: str) -> str:
    """Reconstruct the 'after' content from a single-file unified diff.

    New files get full reconstruction. Modified files get hunks
    with '// ... (unchanged) ...' markers between non-contiguous hunks.
    """
    lines = diff_text.splitlines()
    is_new_file = any(l.startswith("--- /dev/null") for l in lines)

    hunks: List[tuple] = []  # (start_line_1indexed, after_lines)
    current_start = 0
    current_after: List[str] = []
    in_hunk = False

    for line in lines:
        hm = _HUNK_HEADER.match(line)
        if hm:
            if in_hunk and current_after:
                hunks.append((current_start, current_after))
            current_start = int(hm.group(1))
            current_after = []
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            continue  # removed line, not in "after"
        if line.startswith("+"):
            current_after.append(line[1:])
        elif line.startswith("\\"):
            continue  # "\ No newline at end of file"
        else:
            # Context line (space prefix or empty)
            current_after.append(line[1:] if line.startswith(" ") else line)

    if in_hunk and current_after:
        hunks.append((current_start, current_after))

    if not hunks:
        return ""

    if is_new_file:
        # Full file content
        all_lines: List[str] = []
        for _, after_lines in hunks:
            all_lines.extend(after_lines)
        return "\n".join(all_lines) + "\n"

    # Modified file: join hunks with gap markers
    result_lines: List[str] = []
    prev_end = 0
    for start, after_lines in hunks:
        if prev_end > 0 and start > prev_end + 1:
            result_lines.append("// ... (unchanged) ...")
        result_lines.extend(after_lines)
        prev_end = start + len(after_lines) - 1

    if hunks[0][0] > 1:
        result_lines.insert(0, "// ... (unchanged) ...")
    result_lines.append("// ... (unchanged) ...")

    return "\n".join(result_lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model(run_dir: Path) -> str:
    agents_path = run_dir / "inputs" / "agents_config.json"
    if agents_path.is_file():
        agents = _read_json(agents_path)
        roles = agents.get("resolved", {}).get("roles", [])
        models = set()
        for role in roles:
            m = role.get("model", "")
            if m:
                models.add(m)
        if len(models) == 1:
            return models.pop()
        # Multiple models — return a combined string
        if models:
            return "+".join(sorted(models))
    # Fallback: try to parse from run_id
    return "unknown-model"


def _slugify(text: str) -> str:
    """Turn 'moonshotai/kimi-k2.5' into 'kimi-k2.5'."""
    # Strip provider prefix (moonshotai/, openai/, anthropic/, etc.)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text)


def _find_repo_bundle(snapshot: Dict[str, Any], run_dir: Path) -> Optional[Path]:
    """Locate the repo bundle from the task snapshot."""
    bundle = snapshot.get("workspace_base_repo_path")
    if bundle:
        p = Path(bundle)
        if p.is_file():
            return p
    # Try relative to task_root
    task_root = snapshot.get("task_root")
    if task_root:
        p = Path(task_root) / "repo.bundle"
        if p.is_file():
            return p
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
