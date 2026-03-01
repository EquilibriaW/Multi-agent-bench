#!/usr/bin/env python3
"""
loopbench role driver — DSPy + Pydantic.

Harness contract: reads LB_CONTEXT_JSON → runs agent → writes LB_OUTPUT_JSON

Design principles:
  - Coder = ReAct (local search + edits in bounded workspace)
  - Reviewer = Pipeline scaffold (map-reduce DAG, not repo exploration)
  - RLM pattern: llm_query() for symbolic sub-calls
  - LLM is the judge, Git/CI is the fact witness
  - Structured artifacts (PR packets, review findings) over prose
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _ensure_deps():
    """Idempotent: install missing deps if running in a sandbox."""
    import importlib
    missing = []
    for mod in ("dspy", "pydantic", "litellm"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if not missing:
        return
    import subprocess as _sp
    pkg_map = {"dspy": "dspy-ai", "pydantic": "pydantic", "litellm": "litellm"}
    pkgs = [pkg_map.get(m, m) for m in missing]
    print(f"[bootstrap] Installing: {', '.join(pkgs)}", file=sys.stderr)
    _sp.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"] + pkgs,
        timeout=120,
    )


_ensure_deps()

import dspy
import litellm
from pydantic import BaseModel, Field

# ── Globals (set in main) ────────────────────────────────────

WORKTREE = Path(".")
KNOWLEDGE_DIR: Path | None = None
_inspected_commits: dict[str, set[str]] = {}
_failed_commands: list[dict[str, Any]] = []

MAX_ITERS = {"bootstrap": 15, "reflect": 8, "finalize": 10}
DEFAULT_MAX_ITERS = 25

def _command_timeout_cap() -> int:
    """Max per-command timeout.  Mirrors openrouter_role_driver._get_command_policy()."""
    backend = os.environ.get("LOOPBENCH_SANDBOX_BACKEND", "").strip().lower()
    default = 1200 if backend == "e2b_firecracker" else 180
    raw = os.environ.get("LOOPBENCH_COMMAND_TIMEOUT_SEC")
    if raw is None:
        return default
    try:
        return max(30, min(int(raw), 3600))
    except ValueError:
        return default

COMMAND_TIMEOUT_CAP: int = _command_timeout_cap()


# ── Pydantic Models ──────────────────────────────────────────


class Subtask(BaseModel):
    id: str
    role: str
    title: str
    paths: list[str] = []
    acceptance: str = ""


class ReviewFinding(BaseModel):
    severity: str = "info"  # critical / major / minor / info
    file: str = ""
    description: str = ""
    suggested_fix: str = ""


class FeedbackItem(BaseModel):
    required_changes: list[str] = []
    suggested_changes: list[str] = []
    locations: list[str] = []
    acceptance_test: str = ""


# ── Tool Implementations ─────────────────────────────────────


def list_files(path: str = ".") -> str:
    """List directory contents with file sizes."""
    target = WORKTREE / path
    if not target.is_dir():
        return f"Error: {path} is not a directory"
    entries = []
    for p in sorted(target.iterdir()):
        if p.name.startswith("."):
            continue
        size = p.stat().st_size if p.is_file() else 0
        kind = "d" if p.is_dir() else "f"
        entries.append(f"{kind} {size:>8} {p.relative_to(WORKTREE)}")
    return "\n".join(entries[:200]) or "(empty)"


def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read file content. Output truncated to 10 KB."""
    target = WORKTREE / path
    if not target.exists():
        return f"Error: {path} not found"
    if not target.is_file():
        return f"Error: {path} is not a file"
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if offset > 0:
        lines = lines[offset:]
    if limit > 0:
        lines = lines[:limit]
    result = "".join(lines)
    return result[:10_000] + "\n… [truncated]" if len(result) > 10_000 else (result or "(empty file)")


def write_file(path: str, content: str) -> str:
    """Write full content to a file (creates or overwrites)."""
    target = WORKTREE / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"


def patch_file(path: str, old: str, new: str) -> str:
    """Replace exact text in a file.  old must appear exactly once."""
    target = WORKTREE / path
    if not target.exists():
        return f"Error: {path} not found"
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        return f"Error: old text not found in {path}"
    if count > 1:
        return f"Error: old text appears {count} times — must be unique"
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"Patched {path}"


def exec_command(command: str, timeout_sec: int = 60) -> str:
    """Run a shell command in the repository working directory."""
    try:
        r = subprocess.run(
            ["bash", "-c", command],
            cwd=str(WORKTREE),
            capture_output=True, text=True,
            timeout=min(timeout_sec, COMMAND_TIMEOUT_CAP),
        )
        if r.returncode != 0:
            _failed_commands.append({
                "command": command,
                "exit_code": r.returncode,
                "stderr_tail": (r.stderr or "")[-500:],
            })
        out = r.stdout[-5000:] if r.stdout else ""
        err = f"\n[stderr]\n{r.stderr[-2000:]}" if r.stderr else ""
        return f"exit={r.returncode}\n{out}{err}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout_sec}s"
    except Exception as e:
        return f"Error: {e}"


def git_status() -> str:
    """Show git working tree status."""
    return exec_command("git status --short -- . ':(exclude).loopbench/' ':(exclude)__pycache__/' ':(exclude)*.pyc'")


def git_diff(ref: str = "HEAD") -> str:
    """Show unified diff versus HEAD or a specific ref."""
    return exec_command(f"git diff {ref} -- . ':(exclude).loopbench/' ':(exclude)__pycache__/' ':(exclude)*.pyc'")


def git_add(paths: str = ".") -> str:
    """Stage files for commit."""
    return exec_command(f"git add -- {paths} ':(exclude).loopbench/' ':(exclude)__pycache__/' ':(exclude)*.pyc'")


def git_commit(message: str) -> str:
    """Create a git commit with staged changes."""
    escaped = message.replace("'", "'\\''")
    return exec_command(f"git commit -m '{escaped}'")


def git_show(commit: str, max_lines: int = 300) -> str:
    """Show a commit's diff. Use to inspect coder commits before merging."""
    result = exec_command(f"git show --stat --patch {commit} | head -{max_lines}")
    if "exit=0" in result.split("\n", 1)[0]:
        _inspected_commits.setdefault("_all", set()).add(commit.strip()[:40])
    return result


def git_diff_files(commit: str) -> str:
    """List files changed in a specific commit."""
    result = exec_command(f"git diff-tree --no-commit-id --name-status -r {commit}")
    if "exit=0" in result.split("\n", 1)[0]:
        _inspected_commits.setdefault("_all", set()).add(commit.strip()[:40])
    return result


def read_knowledge(surface: str) -> str:
    """Read a knowledge surface. Available: directive, task_understanding, failure_patterns, workflow_insights."""
    if KNOWLEDGE_DIR is None:
        return "Knowledge surfaces not available."
    p = KNOWLEDGE_DIR / f"{surface}.md"
    if p.exists():
        content = p.read_text(encoding="utf-8")
        return content[:5000] if content else "(empty)"
    return f"Surface '{surface}' not found."


def llm_query(question: str, context: str = "") -> str:
    """Ask a focused sub-question about provided text.
    Use this to analyze diffs or code without loading it into your main reasoning."""
    qa = dspy.Predict("context, question -> answer")
    return qa(context=context[:8000], question=question).answer


# ── DSPy Signatures ───────────────────────────────────────────

# -- Bootstrap (planner) --

class Bootstrap(dspy.Signature):
    """You are a technical planner for a 3-agent coding team.
    Read the task, explore the repo, decompose into subtasks for your coders.
    Split work across different files — two coders editing the same file causes merge conflicts."""

    task_description: str = dspy.InputField(desc="Task requirements (README.task.md)")
    coders: str = dspy.InputField(desc="Coder role names — use EXACTLY these in subtask role fields")
    repo_tree: str = dspy.InputField(desc="Repository file listing")

    plan_markdown: str = dspy.OutputField(desc="Full implementation plan in markdown")
    subtasks_json: str = dspy.OutputField(
        desc='JSON array: [{"id":"S1","role":"coder_a","title":"...","paths":["file.py"],"acceptance":"..."}]'
    )
    summary: str = dspy.OutputField(desc="One-line plan summary")


# -- Coder (implementation + rework) --

class Implement(dspy.Signature):
    """You are a coder in a 3-agent team. Implement the assigned changes.
    Read relevant files before editing. Run tests to verify.
    When finishing, report a structured PR packet: what you changed, why,
    what you tested, and any risks."""

    task_description: str = dspy.InputField(desc="Task requirements")
    assignment: str = dspy.InputField(desc="Your subtask: title, acceptance, assigned paths")
    plan: str = dspy.InputField(desc="Implementation plan from the tech lead")
    feedback: str = dspy.InputField(desc="Reviewer feedback to address (empty on first pass)")
    current_changes: str = dspy.InputField(desc="Your current git diff (empty on first pass)")

    # PR packet outputs — structured artifact for the reviewer
    summary: str = dspy.OutputField(desc="What you implemented (1-3 sentences)")
    commit_message: str = dspy.OutputField(desc="Git commit message")
    files_changed: str = dspy.OutputField(desc="Comma-separated list of files you modified")
    key_decisions: str = dspy.OutputField(desc="Key design decisions, one per line")
    tests_run: str = dspy.OutputField(desc="Test commands you ran and results (pass/fail)")
    known_risks: str = dspy.OutputField(desc="Potential risks or issues, one per line (or 'none')")


# -- Review pipeline sub-signatures --

class AnalyzeDiff(dspy.Signature):
    """Analyze a single coder's changes against requirements.
    Focus on correctness, completeness, and risk."""

    requirements: str = dspy.InputField(desc="Task requirements")
    pr_packet: str = dspy.InputField(desc="Coder's structured PR packet")
    diff_hunks: str = dspy.InputField(desc="Key diff hunks to inspect closely")
    already_merged_hunks: str = dspy.InputField(
        desc="Diffs from prior rounds already integrated into the main branch. "
             "This is the baseline — do NOT flag these as missing."
    )
    feedback_digest: str = dspy.InputField(
        desc="Prior round feedback history (what was requested, what was merged)"
    )

    requirements_met: str = dspy.OutputField(desc="Which requirements are addressed")
    requirements_gaps: str = dspy.OutputField(desc="Which requirements are missing or wrong")
    issues_json: str = dspy.OutputField(
        desc='JSON: [{"severity":"major","file":"x.py","description":"...","suggested_fix":"..."}]'
    )
    risk_score: int = dspy.OutputField(desc="0=safe to 5=critical risk")


class CheckInteraction(dspy.Signature):
    """Check for semantic conflicts between two coders' changes to shared files."""

    summary_a: str = dspy.InputField(desc="Summary of coder A's changes")
    summary_b: str = dspy.InputField(desc="Summary of coder B's changes")
    shared_files: str = dspy.InputField(desc="Files modified by both coders")

    has_conflict: bool = dspy.OutputField(desc="Whether there's a semantic conflict")
    description: str = dspy.OutputField(desc="Conflict description (or 'no conflict')")
    resolution: str = dspy.OutputField(desc="Suggested resolution if conflicting")


class MergeDecide(dspy.Signature):
    """Decide which commits to merge based on per-coder findings and interaction analysis.

    Your job: select which available commits to merge into the integrated branch.
    Include every commit that is safe to merge (scratch ok=true, no semantic conflicts).
    Omit commits that would break the codebase."""

    requirements: str = dspy.InputField(desc="Task requirements and acceptance criteria")
    feedback_digest: str = dspy.InputField(
        desc="Concise history of prior review rounds: what was requested, what was merged. "
             "Consult review_ledger for full details on any round."
    )
    all_findings: str = dspy.InputField(desc="Per-coder analysis results")
    interactions: str = dspy.InputField(desc="Cross-coder interaction findings")
    available_commits: str = dspy.InputField(desc="Commit SHAs available per coder")
    harness_issues: str = dspy.InputField(desc="Issues detected by harness")
    review_ledger: str = dspy.InputField(desc="Prior decisions with causal reasoning (why rework was needed)")
    scratch_merge_results: str = dspy.InputField(desc="Deterministic merge feasibility per commit")

    merge_commits_json: str = dspy.OutputField(
        desc='JSON: {"coder_a": ["sha"], "coder_b": []} — commits to merge'
    )
    reasoning: str = dspy.OutputField(desc="Explanation of the decision")


class WriteFeedback(dspy.Signature):
    """Write actionable feedback for a coder who needs rework.
    Be specific: exact files, exact changes, exact acceptance test.

    IMPORTANT: If prior_feedback is non-empty, these issues were flagged in
    earlier rounds. You MUST:
    1. Acknowledge what the coder fixed since last round
    2. Identify what remains unfixed or newly broken
    3. Never repeat prior feedback verbatim — reference it and explain what changed"""

    coder_findings: str = dspy.InputField(desc="This coder's review findings")
    decision_reasoning: str = dspy.InputField(desc="Why rework was requested")
    requirements: str = dspy.InputField(desc="Task requirements")
    prior_feedback: str = dspy.InputField(
        desc="Feedback given to this coder in prior rounds (empty if first round). "
             "Format: [Round N] feedback text. Reference this to avoid repeating yourself."
    )

    required_changes: str = dspy.OutputField(desc="Must-do changes, one per line")
    locations: str = dspy.OutputField(desc="File paths and line ranges to change")
    acceptance_test: str = dspy.OutputField(desc="How we'll verify it's fixed")


# -- Reflect + Finalize (simple) --

class Reflect(dspy.Signature):
    """Analyze what happened this round and produce knowledge for the next round."""

    round_summary: str = dspy.InputField(desc="What happened: merges, validation, feedback")
    current_knowledge: str = dspy.InputField(desc="Current knowledge surfaces")

    directive: str = dspy.OutputField(desc="Concise directive for next round (200-600 chars)")
    task_understanding: str = dspy.OutputField(desc="Updated task understanding")
    failure_patterns: str = dspy.OutputField(desc="Patterns in what went wrong")
    workflow_insights: str = dspy.OutputField(desc="Process improvements")


class AdversarialReflect(dspy.Signature):
    """You are a skeptical senior engineer reviewing what happened this round.
    Challenge assumptions. Identify what was tried and FAILED. Call out
    rationalization. If the same pattern repeats, say so bluntly.
    Focus on what the TEAM should do differently, not what looks good."""

    round_summary: str = dspy.InputField(desc="What happened: merges, validation, feedback")
    current_knowledge: str = dspy.InputField(desc="Current knowledge surfaces")
    prior_causes: str = dspy.InputField(desc="Causal chain from review ledger — why each prior round was reworked")

    directive: str = dspy.OutputField(desc="Contrarian directive for next round (200-600 chars)")
    task_understanding: str = dspy.OutputField(desc="What the team is STILL getting wrong about the task")
    failure_patterns: str = dspy.OutputField(desc="Recurring failure patterns — be blunt")
    workflow_insights: str = dspy.OutputField(desc="Process changes needed")


class Finalize(dspy.Signature):
    """Final integration check. Verify the code works end-to-end, run tests,
    fix any remaining issues."""

    task_description: str = dspy.InputField(desc="Task requirements")
    status: str = dspy.InputField(desc="Current merge status and metrics")

    summary: str = dspy.OutputField(desc="Final assessment")


# ── Orchestrator Review Pipeline ──────────────────────────────


def _build_feedback_digest(review_ledger: dict) -> str:
    """Build a concise structured digest of prior review rounds."""
    if not review_ledger:
        return ""
    entries = review_ledger.get("entries", [])
    if not entries:
        return ""
    lines = []
    for entry in entries:
        rd = entry.get("round_index", "?")
        decision = entry.get("decision", "?")
        merged = entry.get("commits_merged", {})
        merged_summary = ", ".join(
            f"{coder}: {len(shas)}" for coder, shas in merged.items() if shas
        ) or "none"
        lines.append(f"Round {rd}: {decision} (merged: {merged_summary})")
        # Per-coder feedback summary — first 150 chars of each
        fb_map = entry.get("coder_feedback_map", {})
        for coder, fb in fb_map.items():
            if fb:
                lines.append(f"  {coder}: {fb[:150]}...")
    return "\n".join(lines)


class OrchestratorReview(dspy.Module):
    """Pipeline scaffold for the reviewer / tech lead.

    Stages:
      1. Per-coder analysis (LLM, bounded — map)
      2. Interaction graph (deterministic — filter)
      3. Semantic interaction checks (LLM, only on overlapping pairs)
      4. Merge decision (LLM — reduce)
      5. Feedback generation (LLM, per-coder needing rework)

    The LLM is the judge; Git/CI is the fact witness.
    """

    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict(AnalyzeDiff)
        self.check_interaction = dspy.Predict(CheckInteraction)
        self.decide = dspy.Predict(MergeDecide)
        self.write_feedback = dspy.Predict(WriteFeedback)

    def forward(
        self,
        requirements: str,
        coder_packets: dict[str, dict],
        coder_commits: dict[str, list[str]],
        harness_issues: list[str],
        review_ledger: dict | None = None,
        scratch_merge_results: dict | None = None,
        inline_diffs: dict[str, list[dict]] | None = None,
        already_merged_diffs: dict[str, list[dict]] | None = None,
    ) -> dspy.Prediction:
        # ── Stage 1: Per-coder analysis (map) ──
        # Use inline_diffs from the harness (pre-materialized from coder
        # worktrees) instead of running git show locally — the reviewer's
        # worktree does not contain coder commit objects.
        findings: dict[str, Any] = {}
        for coder, packet in coder_packets.items():
            shas = coder_commits.get(coder, [])
            hunks = self._inspect_commits(shas, coder, inline_diffs)
            # _mark_inspected called inside _inspect_commits

            # Build already-merged context with separate budget so it
            # doesn't crowd out the new diff hunks.
            merged_context = ""
            if already_merged_diffs and coder in already_merged_diffs:
                parts = []
                for entry in already_merged_diffs[coder]:
                    diff = entry.get("diff_content", "")
                    if diff:
                        parts.append(f"[merged: {entry.get('sha', '?')[:8]}]\n{diff[:3000]}")
                merged_context = "\n".join(parts)

            findings[coder] = self.analyze(
                requirements=requirements,
                pr_packet=json.dumps(packet, indent=2),
                diff_hunks=hunks[:8000],                          # full budget for new code
                already_merged_hunks=merged_context[:6000],        # separate budget
                feedback_digest=_build_feedback_digest(review_ledger),
            )

        # ── Stage 2: Interaction graph (deterministic) ──
        edges = self._interaction_graph(coder_packets)

        # ── Stage 3: Semantic interaction checks (only on edges) ──
        interactions: dict[str, Any] = {}
        for a, b in edges:
            files_a = set(coder_packets[a].get("files_changed", []))
            files_b = set(coder_packets[b].get("files_changed", []))
            shared = files_a & files_b
            interactions[f"{a}+{b}"] = self.check_interaction(
                summary_a=coder_packets[a].get("summary", ""),
                summary_b=coder_packets[b].get("summary", ""),
                shared_files=", ".join(shared),
            )

        # ── Stage 4: Merge decision (reduce) ──
        # Only show commits that are actually merge candidates (present in
        # scratch_merge_results).  Already-merged commits aren't candidates
        # and their absence from scratch results confuses the model.
        tested_shas = set((scratch_merge_results or {}).get("commits_tested", {}).keys())
        mergeable_commits = {
            coder: [sha for sha in shas if sha in tested_shas]
            for coder, shas in coder_commits.items()
        }
        decision = self.decide(
            requirements=requirements,
            feedback_digest=_build_feedback_digest(review_ledger),
            all_findings=json.dumps(
                {c: {"gaps": f.requirements_gaps, "risk": f.risk_score, "issues": f.issues_json}
                 for c, f in findings.items()},
                indent=2,
            ),
            interactions=json.dumps(
                {k: {"conflict": v.has_conflict, "desc": v.description}
                 for k, v in interactions.items()},
                indent=2,
            ) if interactions else "{}",
            available_commits=json.dumps(mergeable_commits, indent=2),
            harness_issues="\n".join(f"- {i}" for i in harness_issues) if harness_issues else "None",
            review_ledger=json.dumps(review_ledger or {}, indent=2),
            scratch_merge_results=json.dumps(scratch_merge_results or {}, indent=2),
        )

        # ── Stage 5: Feedback (per-coder, only if rework needed) ──
        # Build per-coder prior feedback from the review ledger so
        # WriteFeedback can reference what was already requested.
        prior_by_coder: dict[str, str] = {}
        if review_ledger:
            for entry in (review_ledger.get("entries") or []):
                if entry.get("decision") != "rework":
                    continue
                rnd = entry.get("round_index", "?")
                # Prefer coder_feedback_map (keyed by coder) over open_issues (flat list).
                fb_map = entry.get("coder_feedback_map") or {}
                if fb_map:
                    for coder_name, fb_text in fb_map.items():
                        if coder_name in findings:
                            prior_by_coder.setdefault(coder_name, "")
                            prior_by_coder[coder_name] += f"[Round {rnd}] {fb_text}\n"
                else:
                    for issue in (entry.get("open_issues") or []):
                        for coder in findings:
                            prior_by_coder.setdefault(coder, "")
                            prior_by_coder[coder] += f"[Round {rnd}] {issue}\n"

        # Derive request_rework deterministically: if any coder has
        # candidates that weren't nominated for merge, they need rework.
        nominated = _safe_json(decision.merge_commits_json) or {}
        nominated_shas = set()
        if isinstance(nominated, dict):
            for shas in nominated.values():
                if isinstance(shas, list):
                    nominated_shas.update(shas)
        all_candidate_shas = set()
        for shas in mergeable_commits.values():
            all_candidate_shas.update(shas)
        derived_rework = bool(all_candidate_shas - nominated_shas)

        coder_feedback: dict[str, str] = {}
        if derived_rework:
            for coder, finding in findings.items():
                fb = self.write_feedback(
                    coder_findings=f"Gaps: {finding.requirements_gaps}\nIssues: {finding.issues_json}",
                    decision_reasoning=decision.reasoning,
                    requirements=requirements,
                    prior_feedback=prior_by_coder.get(coder, ""),
                )
                coder_feedback[coder] = (
                    f"Required changes:\n{fb.required_changes}\n"
                    f"Locations: {fb.locations}\n"
                    f"Done when: {fb.acceptance_test}"
                )

        return dspy.Prediction(
            summary=decision.reasoning,
            merge_commits_json=decision.merge_commits_json,
            request_rework=derived_rework,
            coder_feedback_json=json.dumps(coder_feedback),
        )

    @staticmethod
    def _inspect_commits(
        shas: list[str],
        coder: str,
        inline_diffs: dict[str, list[dict]] | None,
    ) -> str:
        """Use harness-provided inline diffs (from coder worktrees).

        Falls back to local git show only if inline_diffs are unavailable,
        but the reviewer's worktree typically lacks coder commit objects.
        """
        # Try inline diffs first (pre-materialized from coder worktrees)
        if inline_diffs and coder in inline_diffs:
            parts = []
            for entry in inline_diffs[coder][:5]:
                sha = entry.get("sha", "?")[:10]
                subject = entry.get("subject", "")
                diff_content = entry.get("diff_content", "")
                parts.append(f"=== {sha} ({subject}) ===\n{diff_content[:3000]}")
            if parts:
                _mark_inspected(shas)
                return "\n".join(parts)

        # Fallback: local git show (may fail if objects don't exist here)
        parts = []
        for sha in shas[:5]:
            diff = git_show(sha, max_lines=200)
            parts.append(f"=== {sha[:10]} ===\n{diff[:3000]}")
        return "\n".join(parts)

    @staticmethod
    def _interaction_graph(packets: dict[str, dict]) -> list[tuple[str, str]]:
        """Deterministic: edges only where files overlap."""
        edges = []
        coders = list(packets.keys())
        for i, a in enumerate(coders):
            files_a = set(packets[a].get("files_changed", []))
            for b in coders[i + 1:]:
                files_b = set(packets[b].get("files_changed", []))
                if files_a & files_b:
                    edges.append((a, b))
        return edges


def _mark_inspected(shas: list[str]) -> None:
    for sha in shas:
        _inspected_commits.setdefault("_all", set()).add(sha.strip()[:40])


# ── Helpers ───────────────────────────────────────────────────


def _read_task_readme() -> str:
    for name in ("README.task.md", "README.md"):
        p = WORKTREE / name
        if p.exists():
            return p.read_text(encoding="utf-8")[:8000]
    return "(no task description found)"


def _tracked_files() -> str:
    r = subprocess.run(
        ["git", "ls-files"], cwd=str(WORKTREE),
        capture_output=True, text=True, timeout=10,
    )
    return r.stdout[:3000] if r.stdout else "(no tracked files)"


def _read_all_knowledge() -> str:
    if KNOWLEDGE_DIR is None:
        return "(no knowledge surfaces)"
    parts = []
    for name in ("directive", "task_understanding", "failure_patterns", "workflow_insights"):
        p = KNOWLEDGE_DIR / f"{name}.md"
        if p.exists():
            content = p.read_text(encoding="utf-8").strip()
            if content:
                parts.append(f"## {name}\n{content[:2000]}")
    return "\n\n".join(parts) or "(no knowledge surfaces yet)"


def _round_summary(ctx: dict) -> str:
    merged = ctx.get("merged_commits_this_round", {})
    review = ctx.get("review_decision_summary", {})
    val = ctx.get("validation_result", {})
    parts = [
        f"Merged commits: {json.dumps(merged)}",
        f"Rework requested: {review.get('request_rework', False)}",
        f"Validation: {'pass' if val.get('ok') else 'fail'}",
    ]
    stderr = val.get("stderr", "")
    if stderr:
        parts.append(f"Stderr tail:\n{stderr[-2000:]}")
    fb = review.get("coder_feedback", {})
    if fb:
        parts.append(f"Coder feedback: {json.dumps(fb)}")
    return "\n".join(parts)


def _prior_causes(ctx: dict) -> str:
    ledger = ctx.get("review_ledger", {})
    entries = ledger.get("entries", [])
    summaries = ledger.get("prior_summaries", [])
    parts = list(summaries)
    for e in entries:
        parts.append(f"round {e.get('round_index')}: {e.get('decision')} cause={e.get('cause', '?')}")
    return "\n".join(parts) or "(first round)"


def _build_pr_packets(ctx: dict) -> dict[str, dict]:
    """Extract PR packets from coder outputs passed by the harness.

    If the coders used the new driver (with PR packet fields), we get
    structured packets. Otherwise, fall back to minimal metadata.
    """
    packets: dict[str, dict] = {}
    coder_outputs = ctx.get("latest_coder_outputs", {})
    if not isinstance(coder_outputs, dict):
        coder_outputs = {}

    for coder, outputs in coder_outputs.items():
        # outputs may be a list of output dicts or a single dict
        if isinstance(outputs, list):
            items = outputs
        elif isinstance(outputs, dict):
            items = [outputs]
        else:
            items = []

        # Merge all subtask outputs for this coder
        summaries: list[str] = []
        files: list[str] = []
        decisions: list[str] = []
        tests: list[str] = []
        risks: list[str] = []
        for out in items:
            if not isinstance(out, dict):
                continue
            s = out.get("summary", "")
            if s:
                subtask_key = out.get("output_key", "")
                label = subtask_key.split(":")[-1] if subtask_key else ""
                summaries.append(f"[{label}] {s}" if label else s)
            files.extend(_parse_list(out.get("files_changed", "")))
            decisions.extend(_parse_list(out.get("key_decisions", "")))
            tests.extend(_parse_list(out.get("tests_run", "")))
            risks.extend(_parse_list(out.get("known_risks", "")))

        packets[coder] = {
            "summary": "\n\n".join(summaries),
            "files_changed": list(dict.fromkeys(files)),  # dedupe, preserve order
            "key_decisions": decisions,
            "tests_run": tests,
            "known_risks": risks,
        }
    return packets


def _parse_list(val: Any) -> list[str]:
    """Parse a value that might be a comma/newline separated string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        items = [s.strip() for s in val.replace(",", "\n").split("\n")]
        return [s for s in items if s and s.lower() != "none"]
    return []


def _safe_json(text: str, fallback: Any = None) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback if fallback is not None else {}


# ── Tool Schemas (OpenAI function-calling format) ─────────────

EXECUTE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Run a tool by name. See system instructions for available tools and arguments.",
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "description": "Tool name"},
                "args": {"type": "object", "description": "Tool arguments as key-value pairs"},
            },
            "required": ["tool"],
        },
    },
}

_COMMON_TOOLS = {"list_files", "read_file", "exec_command", "git_status", "git_diff", "llm_query"}
_CODER_TOOLS = _COMMON_TOOLS | {"write_file", "patch_file", "git_add", "git_commit"}

PHASE_TOOL_NAMES: dict[str, set[str]] = {
    "bootstrap": _COMMON_TOOLS,
    "implementation": _CODER_TOOLS,
    "rework": _CODER_TOOLS,
    "finalize": _CODER_TOOLS,
}

TOOL_SIGNATURES: dict[str, str] = {
    "list_files":     "list_files(path?) — directory listing with sizes",
    "read_file":      "read_file(path, offset?, limit?) — file content (truncated at 10KB)",
    "exec_command":   "exec_command(command, timeout_sec?) — shell command (60s default, long builds OK)",
    "git_status":     "git_status() — working tree status",
    "git_diff":       "git_diff(ref?) — diff vs HEAD or ref",
    "llm_query":      "llm_query(question, context?) — sub-question to another LLM",
    "write_file":     "write_file(path, content) — create/overwrite file",
    "patch_file":     "patch_file(path, old, new) — replace exact text (old must be unique)",
    "git_add":        "git_add(paths?) — stage files (default: '.')",
    "git_commit":     "git_commit(message) — commit staged changes",
    "read_knowledge": "read_knowledge(surface) — directive, task_understanding, failure_patterns, workflow_insights",
}

BOOTSTRAP_SUBMIT_SCHEMA: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the bootstrap plan and subtask decomposition",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_markdown": {"type": "string", "description": "Full plan in markdown format"},
                    "subtasks": {
                        "type": "array",
                        "description": "List of subtask objects with id, role, title, paths, acceptance",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string", "enum": ["coder_a", "coder_b"]},
                                "title": {"type": "string"},
                                "paths": {"type": "array", "items": {"type": "string"}},
                                "acceptance": {"type": "string"},
                            },
                        },
                    },
                    "summary": {"type": "string", "description": "Short summary of the plan"},
                },
                "required": ["plan_markdown", "subtasks"],
            },
        },
    },
]

def _tools_for_phase(phase: str) -> list[dict]:
    """Assemble correct tool set per phase."""
    if phase == "bootstrap":
        return [EXECUTE_SCHEMA, BOOTSTRAP_SUBMIT_SCHEMA[0]]
    # All other phases: execute only, no submit
    return [EXECUTE_SCHEMA]


def _tool_reference(phase: str) -> str:
    """Compact tool reference text for the system prompt."""
    names = PHASE_TOOL_NAMES.get(phase, _COMMON_TOOLS)
    if KNOWLEDGE_DIR is not None:
        names = names | {"read_knowledge"}
    lines = ["Available tools (use via execute(tool, args)):"]
    for name in sorted(names):
        if name in TOOL_SIGNATURES:
            lines.append(f"  - {TOOL_SIGNATURES[name]}")
    return "\n".join(lines)


# ── Prompt Builders (for native tool loop) ────────────────────


def _build_system_prompt(phase: str, ctx: dict) -> str:
    """Build system prompt for the native tool-calling loop."""
    base = (
        "You are part of a 3-agent coding team in a benchmark harness. "
        "Prefer minimal, correct edits. Keep files syntactically valid. "
        "Use read_file to understand existing code before making changes."
    )

    if phase == "bootstrap":
        role_desc = (
            "You are a technical planner. Read the task description, explore the repository, "
            "and decompose the work into subtasks for your 2 coders.\n\n"
            "Call submit(plan_markdown=..., subtasks=[...], summary=...) when your plan is ready. "
            "submit() is a separate tool, not called via execute().\n\n"
            "subtasks format: [{\"id\": \"A\", \"role\": \"coder_a\", \"title\": \"...\", "
            "\"paths\": [\"file.py\"], \"acceptance\": \"...\"}]\n"
            "Valid roles: \"coder_a\", \"coder_b\" only."
        )
    elif phase in ("implementation", "rework"):
        role_desc = (
            "You are a coder in a 3-agent team. Implement your assigned changes.\n\n"
            "Use the tools to read, edit, and test code. "
            "When you're done, simply stop — your changes will be committed automatically."
        )
    elif phase == "finalize":
        role_desc = (
            "You are doing final integration checks. Verify the code works end-to-end.\n\n"
            "Use the tools to read, edit, and test code. Fix any remaining issues. "
            "When you're done, simply stop — your changes will be committed automatically."
        )
    else:
        role_desc = f"You are working on the {phase} phase."

    parts = [base, "", role_desc, "", _tool_reference(phase)]

    # Add time budget urgency if available
    deadline = ctx.get("run_deadline_epoch")
    if deadline:
        import time
        remaining = max(0, int(deadline - time.time()))
        if remaining < 120:
            parts.append(
                f"\nURGENT: Only ~{remaining}s remaining. Submit whatever you have NOW. "
                "A timeout with no code is the worst possible outcome — you lose entirely. "
                "Imperfect code that compiles is infinitely better than no code at all."
            )
        elif remaining < 300:
            parts.append(
                f"\nTime pressure: ~{remaining}s remaining. Wrap up soon. "
                "If time runs out with nothing committed, you get a score of zero. "
                "Prioritize having working code over perfect code."
            )
        elif remaining < 900:
            parts.append(
                f"\nTime remaining: ~{remaining // 60}min. Stay focused on implementation, "
                "avoid over-engineering. Shipped code beats perfect plans."
            )

    return "\n".join(parts)


def _build_user_prompt(phase: str, ctx: dict) -> str:
    """Format existing input builder output as a markdown user prompt."""
    if phase == "bootstrap":
        inputs = _bootstrap_inputs(ctx)
        coders = inputs.get("coders", "coder_a, coder_b")
        is_rebootstrap = ctx.get("is_rebootstrap", False)
        parts = []
        if is_rebootstrap:
            parts.append("**IMPORTANT: Previous plan failed. You must change your approach.**\n")
        parts.append(f"# Task Description\n\n{inputs['task_description']}\n")
        parts.append(f"## Coders\n{coders}\n")
        parts.append(f"## Repository Structure\n```\n{inputs['repo_tree']}\n```\n")
        parts.append("## Your Job\nCreate a plan and call submit(plan_markdown, subtasks, summary).")
        return "\n".join(parts)

    if phase in ("implementation", "rework"):
        inputs = _coder_inputs(ctx)
        parts = [
            f"# Task Description\n\n{inputs['task_description']}\n",
            f"## Your Assignment\n{inputs['assignment']}\n",
            f"## Implementation Plan\n{inputs['plan']}\n",
            f"## Feedback\n{inputs['feedback']}\n",
        ]
        if inputs.get("current_changes"):
            parts.append(f"## Current Changes (git diff)\n```\n{inputs['current_changes']}\n```\n")
        parts.append("## Your Job\nImplement the changes and test them. Your changes are committed automatically when you stop.")
        return "\n".join(parts)

    if phase == "finalize":
        inputs = _finalize_inputs(ctx)
        return (
            f"# Task Description\n\n{inputs['task_description']}\n\n"
            f"## Current Status\n```json\n{inputs['status']}\n```\n\n"
            "## Your Job\nVerify the code works end-to-end. Fix any issues. Your changes are committed automatically when you stop."
        )

    return f"Phase: {phase}\nContext: {json.dumps(ctx, indent=2)[:4000]}"


# ── Tool Execution Dispatch ───────────────────────────────────


def _dispatch_execute(phase: str, tool_name: str, args: dict) -> str:
    """Dispatch tool name + args to existing Python tool functions with phase gating."""
    allowed = PHASE_TOOL_NAMES.get(phase, _COMMON_TOOLS)
    if KNOWLEDGE_DIR is not None:
        allowed = allowed | {"read_knowledge"}
    if tool_name not in allowed:
        return f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(allowed))}"
    try:
        if tool_name == "list_files":
            return list_files(path=args.get("path", "."))
        if tool_name == "read_file":
            return read_file(
                path=args.get("path", ""),
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 0)),
            )
        if tool_name == "exec_command":
            return exec_command(
                command=args.get("command", ""),
                timeout_sec=int(args.get("timeout_sec", 60)),
            )
        if tool_name == "git_status":
            return git_status()
        if tool_name == "git_diff":
            return git_diff(ref=args.get("ref", "HEAD"))
        if tool_name == "llm_query":
            return llm_query(
                question=args.get("question", ""),
                context=args.get("context", ""),
            )
        if tool_name == "write_file":
            return write_file(
                path=args.get("path", ""),
                content=args.get("content", ""),
            )
        if tool_name == "patch_file":
            return patch_file(
                path=args.get("path", ""),
                old=args.get("old", ""),
                new=args.get("new", ""),
            )
        if tool_name == "git_add":
            return git_add(paths=args.get("paths", "."))
        if tool_name == "git_commit":
            return git_commit(message=args.get("message", ""))
        if tool_name == "read_knowledge":
            return read_knowledge(surface=args.get("surface", ""))
        return f"Unknown tool '{tool_name}'"
    except Exception as e:
        return f"Error: {e}"


# ── Trace Artifact Writers ────────────────────────────────────


def _role_trace_path(output_path: Path, suffix: str, extension: str) -> Path:
    """Build a trace artifact path from the output path."""
    stem = output_path.stem
    if stem.endswith("_output"):
        stem = stem[:-7]
    return output_path.with_name(f"{stem}_{suffix}{extension}")


def _write_trace_artifacts(
    output_path: Path,
    *,
    role: str,
    phase: str,
    model: str,
    messages: list[dict],
    conversation_turns: list[dict],
    total_usage: dict,
    request_payload: dict,
    last_response_text: str,
    continued_from_prior: bool = False,
) -> dict[str, str]:
    """Write conversation.json, conversation.txt, openrouter_request.json, openrouter_response.txt."""
    artifact_paths: dict[str, str] = {}
    try:
        conv_path = _role_trace_path(output_path, "conversation", ".json")
        conv_data = {
            "role": role,
            "phase": phase,
            "model": model,
            "system_message": messages[0] if messages and messages[0].get("role") == "system" else None,
            "user_message": messages[1] if len(messages) > 1 and messages[1].get("role") == "user" else None,
            "turns": conversation_turns,
            "total_usage": total_usage,
            "raw_messages": messages,
            "continued_from_prior": continued_from_prior,
        }
        conv_path.parent.mkdir(parents=True, exist_ok=True)
        conv_path.write_text(json.dumps(conv_data, indent=2), encoding="utf-8")
        artifact_paths[conv_path.name] = str(conv_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        txt_path = _role_trace_path(output_path, "conversation", ".txt")
        transcript = _render_conversation_transcript(messages, conversation_turns, role, phase, model)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(transcript, encoding="utf-8")
        artifact_paths[txt_path.name] = str(txt_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        req_path = _role_trace_path(output_path, "openrouter_request", ".json")
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
        artifact_paths[req_path.name] = str(req_path)
    except Exception:  # noqa: BLE001
        pass

    try:
        resp_path = _role_trace_path(output_path, "openrouter_response", ".txt")
        resp_path.parent.mkdir(parents=True, exist_ok=True)
        resp_path.write_text(last_response_text, encoding="utf-8")
        artifact_paths[resp_path.name] = str(resp_path)
    except Exception:  # noqa: BLE001
        pass

    return artifact_paths


def _render_conversation_transcript(
    messages: list[dict],
    conversation_turns: list[dict],
    role: str,
    phase: str,
    model: str,
) -> str:
    """Render a human-readable transcript of the agentic conversation."""
    lines: list[str] = [f"{role} / {phase}  ({model})", ""]

    # System and user prompts (truncated)
    for msg in messages[:2]:
        msg_role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated)"
        lines.append(f"[{msg_role}]")
        lines.append(content)
        lines.append("")

    lines.append("---")
    lines.append("")

    for turn_data in conversation_turns:
        turn_num = turn_data.get("turn", 0)
        assistant_msg = turn_data.get("assistant_message", {})
        tool_results = turn_data.get("tool_results", [])
        usage = turn_data.get("usage", {})

        tokens = usage.get("total_tokens", 0)
        lines.append(f"[turn {turn_num}]  ({tokens} tok)")

        content = ""
        if isinstance(assistant_msg, dict):
            content = assistant_msg.get("content") or ""
        if content:
            lines.append(content.strip())
            lines.append("")

        tool_calls = assistant_msg.get("tool_calls", []) if isinstance(assistant_msg, dict) else []
        result_map = {tr.get("tool_call_id"): tr for tr in tool_results}

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "?")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                args = raw_args

            # Unwrap execute calls for readable output
            if fn_name == "execute" and isinstance(args, dict):
                fn_name = args.get("tool", "?")
                inner = args.get("args", {})
                # Fallback: model may flatten args into top-level execute dict
                if not inner and len(args) > 1:
                    inner = {k: v for k, v in args.items() if k not in ("tool", "args")}
                args = inner

            if isinstance(args, dict):
                if fn_name in ("read_file", "list_files"):
                    arg_str = args.get("path", str(args))
                elif fn_name == "write_file":
                    path = args.get("path", "?")
                    content_len = len(args.get("content", ""))
                    arg_str = f"{path}  ({content_len} chars)"
                elif fn_name == "exec_command":
                    arg_str = args.get("command", str(args))
                elif fn_name == "submit":
                    arg_str = (args.get("summary") or "")[:120]
                elif fn_name == "patch_file":
                    arg_str = f"{args.get('path', '?')} ({len(args.get('old', ''))} -> {len(args.get('new', ''))} chars)"
                else:
                    arg_str = json.dumps(args, ensure_ascii=False)
                    if len(arg_str) > 200:
                        arg_str = arg_str[:200] + "..."
            else:
                arg_str = str(args)[:200]

            lines.append(f"  > {fn_name}({arg_str})")

            tc_id = tc.get("id", "")
            tr = result_map.get(tc_id, {})
            result_content = tr.get("content", "")
            if result_content:
                if len(result_content) > 800:
                    result_content = result_content[:800] + "\n    ... (truncated)"
                for rline in result_content.split("\n"):
                    lines.append(f"    {rline}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def _build_rework_continuation_prompt(phase: str, ctx: dict) -> str:
    """Build a continuation message for rework phases with carried-forward conversation."""
    parts = ["## Reviewer Feedback"]
    feedback_by_role = ctx.get("planner_feedback_by_role", {})
    role = ctx.get("role", "")
    feedback = feedback_by_role.get(role, "") if isinstance(feedback_by_role, dict) else ""
    if feedback:
        parts.append(feedback)
    else:
        parts.append("The reviewer requested changes but provided no specific feedback.")

    own_diff = ctx.get("own_diff", "")
    if own_diff:
        parts.append(f"\n## Your Current Diff\n```\n{own_diff[:4000]}\n```")

    parts.append("\nAddress the feedback above. Your changes will be committed automatically when you stop.")
    return "\n".join(parts)


# ── Conversation Compaction (OpenHands-style condenser) ──────
#
# When the conversation approaches the context window ceiling, we split
# messages into head / middle / tail, summarize the middle with an LLM
# call, and replace it with a single summary message.  The summary
# absorbs any previous summary so only one ever exists in the history.
#
# Amortized: fires at 100% of budget, reduces to ~50%, so you get many
# turns of headroom before the next condensation.
#
# Mechanical fallback (truncation → stub → drop) if the LLM call fails.

_CONDENSER_PROMPT = """\
You are a context condenser for a coding agent. You will be given a sequence of \
conversation messages (assistant reasoning, tool calls, and tool results) from an \
agent working on a coding task. Produce a concise state summary that preserves all \
information the agent needs to continue working effectively.

Track:
- TASK: What the agent is trying to accomplish (preserve exact requirements)
- FILES_MODIFIED: File paths and what was changed in each
- CODE_STATE: Current implementation state (key functions, data structures)
- TESTS: Test results, failing cases, error messages
- WHAT_WORKED: Approaches that succeeded
- WHAT_FAILED: Approaches tried and failed (critical to avoid repeating)
- CURRENT_STATUS: Where the agent left off, what it was about to do next

Rules:
- Be concise but preserve ALL actionable information
- Include exact file paths, function names, line numbers, error messages
- Distinguish completed work from pending work
- If debugging, preserve the full diagnosis chain
- Never lose information about what was already tried
"""


def _estimate_message_tokens(messages: list[dict]) -> int:
    """Rough token estimate: ~4 chars per token across all message content."""
    total = 0
    for m in messages:
        total += len(str(m.get("content") or ""))
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(str(fn.get("name", "")))
            total += len(str(fn.get("arguments", "")))
    return total // 4


def _get_context_budget(model: str) -> int:
    """Compute available input token budget for messages.

    Returns the number of tokens we can spend on messages before hitting
    the context window ceiling.  Reserves a flat 20K for output + tools +
    safety margin.
    """
    context_window = 0
    try:
        import litellm as _li
        info = _li.get_model_info(f"openrouter/{model}")
        context_window = info.get("max_input_tokens") or info.get("max_tokens") or 0
    except Exception:
        pass
    if not context_window:
        context_window = int(os.environ.get("OPENROUTER_MAX_CONTEXT", "128000"))
    return max(context_window - 20_000, 8000)


_SUMMARY_PREFIX = "[Context Summary"


def _render_messages_for_condenser(messages: list[dict], max_chars_per_msg: int = 10_000) -> str:
    """Render a message sequence into text for the condenser LLM."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = str(m.get("content") or "")
        tool_calls = m.get("tool_calls")

        if role == "assistant" and tool_calls:
            calls_desc = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if len(args) > 1000:
                    args = args[:500] + "...[truncated]..." + args[-300:]
                calls_desc.append(f"  → {name}({args})")
            reasoning = content[:500] + "..." if len(content) > 500 else content
            block = reasoning + "\n" + "\n".join(calls_desc) if reasoning.strip() else "\n".join(calls_desc)
            parts.append(f"[assistant]\n{block}")
        elif role == "tool":
            tc_id = m.get("tool_call_id", "")
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg // 2] + "\n...[truncated]...\n" + content[-max_chars_per_msg // 4:]
            parts.append(f"[tool result {tc_id}]\n{content}")
        elif role == "assistant":
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg]
            parts.append(f"[assistant]\n{content}")
        elif role == "user":
            if len(content) > max_chars_per_msg:
                content = content[:max_chars_per_msg]
            parts.append(f"[user]\n{content}")
    return "\n\n".join(parts)


def _llm_summarize(
    middle: list[dict],
    previous_summary: str,
    model: str,
    api_key: str,
) -> tuple[str | None, dict[str, int]]:
    """Call the LLM to summarize the middle section of the conversation.

    Returns (summary_text, usage_dict).  usage_dict has input_tokens,
    output_tokens, total_tokens — all zero when the call fails.
    """
    import litellm

    zero_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    prev_section = ""
    if previous_summary:
        prev_section = f"<PREVIOUS_SUMMARY>\n{previous_summary}\n</PREVIOUS_SUMMARY>\n\n"

    messages_text = _render_messages_for_condenser(middle)
    user_msg = (
        f"{prev_section}"
        f"Here are the conversation messages to summarize:\n\n"
        f"{messages_text}\n\n"
        f"Produce a concise state summary following the format above."
    )

    try:
        resp = litellm.completion(
            model=f"openrouter/{model}",
            messages=[
                {"role": "system", "content": _CONDENSER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=2000,
            num_retries=2,
            api_key=api_key,
        )
        usage = dict(zero_usage)
        if hasattr(resp, "usage") and resp.usage:
            usage["input_tokens"] = getattr(resp.usage, "prompt_tokens", 0) or 0
            usage["output_tokens"] = getattr(resp.usage, "completion_tokens", 0) or 0
            usage["total_tokens"] = getattr(resp.usage, "total_tokens", 0) or 0
            if usage["total_tokens"] == 0:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        text = resp.choices[0].message.content or ""
        return (text.strip() if text.strip() else None, usage)
    except Exception as exc:
        print(f"[condenser] LLM summarization failed: {exc}", file=sys.stderr)
        return (None, zero_usage)


def _split_head_middle_tail(
    messages: list[dict], tail_budget_tokens: int,
) -> tuple[list[dict], str, list[dict], list[dict]]:
    """Split messages into (head, previous_summary, middle, tail).

    head:  system + first user message(s), minus any previous summary
    previous_summary: extracted text from a prior condensation (or "")
    middle: older turns to be summarized/dropped
    tail:   recent turns preserved intact, sized to fit tail_budget_tokens
    """
    # Head = everything before the first assistant message
    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not asst_indices:
        return messages, "", [], []
    head_end = asst_indices[0]

    # Extract previous summary from head (if any) so it can be absorbed
    previous_summary = ""
    head = []
    for m in messages[:head_end]:
        content = str(m.get("content") or "")
        if m.get("role") == "user" and content.startswith(_SUMMARY_PREFIX):
            previous_summary = content
        else:
            head.append(m)

    # Tail: walk backwards from end, accumulate tokens until tail budget
    tail_tokens = 0
    tail_start = len(messages)
    for i in range(len(messages) - 1, head_end - 1, -1):
        msg_tokens = _estimate_message_tokens([messages[i]])
        if tail_tokens + msg_tokens > tail_budget_tokens:
            break
        tail_tokens += msg_tokens
        tail_start = i

    # Snap to clean assistant boundary (don't start tail mid-turn on a tool msg)
    while tail_start < len(messages) and messages[tail_start].get("role") == "tool":
        tail_start += 1

    if tail_start <= head_end:
        return messages, "", [], []

    middle = messages[head_end:tail_start]
    tail = messages[tail_start:]
    return head, previous_summary, middle, tail


def _mechanical_fallback(
    head: list[dict], middle: list[dict], tail: list[dict], token_budget: int,
) -> list[dict]:
    """Fallback compaction without LLM: truncate → stub → drop."""
    # Phase 1: truncate long tool results in middle
    compressed = []
    for m in middle:
        if m.get("role") == "tool":
            content = str(m.get("content") or "")
            if len(content) > 600:
                compressed.append({**m, "content": content[:250] + "\n...[trimmed]...\n" + content[-250:]})
            else:
                compressed.append(m)
        else:
            compressed.append(m)

    candidate = head + compressed + tail
    if _estimate_message_tokens(candidate) <= token_budget:
        return candidate

    # Phase 2: gut tool results to stubs
    for i, m in enumerate(compressed):
        if m.get("role") == "tool":
            compressed[i] = {**m, "content": "[output omitted]"}
        elif m.get("role") == "assistant" and not m.get("tool_calls"):
            content = str(m.get("content") or "")
            if len(content) > 400:
                compressed[i] = {**m, "content": content[:200] + "..."}

    candidate = head + compressed + tail
    if _estimate_message_tokens(candidate) <= token_budget:
        return candidate

    # Phase 3: drop middle entirely
    marker = {"role": "user", "content": f"{_SUMMARY_PREFIX} — mechanical]\n[Middle turns dropped to fit context window.]"}
    candidate = head + [marker] + tail
    if _estimate_message_tokens(candidate) <= token_budget:
        return candidate

    # Phase 4: trim tail from the front
    trimmed_tail = list(tail)
    while _estimate_message_tokens(head + [marker] + trimmed_tail) > token_budget and len(trimmed_tail) > 3:
        # Drop first turn group from tail (assistant + its tool messages)
        trimmed_tail.pop(0)
        while trimmed_tail and trimmed_tail[0].get("role") == "tool":
            trimmed_tail.pop(0)

    return head + [marker] + trimmed_tail


def _compact_messages(
    messages: list[dict],
    token_budget: int,
    *,
    model: str = "",
    api_key: str = "",
) -> tuple[list[dict], dict[str, int]]:
    """Condense conversation to fit within token budget (OpenHands-style).

    Amortized: fires at 100% of budget, reduces to ~50%.  Uses LLM
    summarization for the dropped middle section; falls back to mechanical
    truncation if the LLM call fails.  Phase-agnostic.

    Returns (compacted_messages, condenser_usage).
    """
    zero_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if _estimate_message_tokens(messages) <= token_budget:
        return messages, zero_usage

    asst_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if len(asst_indices) < 2:
        return messages, zero_usage

    # Target 50% of budget for tail (amortized headroom)
    tail_budget = int(token_budget * 0.5)
    head, previous_summary, middle, tail = _split_head_middle_tail(messages, tail_budget)

    if not middle:
        return messages, zero_usage  # nothing to compact

    # Try LLM summarization
    summary_text = None
    condenser_usage = zero_usage
    if model and api_key:
        summary_text, condenser_usage = _llm_summarize(middle, previous_summary, model, api_key)

    if summary_text:
        summary_msg = {"role": "user", "content": f"{_SUMMARY_PREFIX}]\n{summary_text}"}
        candidate = head + [summary_msg] + tail
        if _estimate_message_tokens(candidate) <= token_budget:
            return candidate, condenser_usage
        # Summary itself is too large — truncate it
        max_summary_tokens = token_budget - _estimate_message_tokens(head + tail) - 100
        max_summary_chars = max(max_summary_tokens * 4, 500)
        summary_msg = {"role": "user", "content": f"{_SUMMARY_PREFIX}]\n{summary_text[:max_summary_chars]}"}
        candidate = head + [summary_msg] + tail
        if _estimate_message_tokens(candidate) <= token_budget:
            return candidate, condenser_usage

    # Fallback: mechanical compaction
    return _mechanical_fallback(head, middle, tail, token_budget), condenser_usage


# ── Native Tool-Calling Loop ─────────────────────────────────


def _run_tool_loop(
    phase: str,
    ctx: dict,
    output_path: Path,
    model: str,
    api_key: str,
) -> dict:
    """Core agentic loop using litellm.completion() with native tool calling.

    Replaces _run_react: sends system+user messages, iterates tool calls
    until the model calls submit() or the turn budget is exhausted.
    """
    # Capture starting commit SHA so we can diff against it later
    try:
        _start = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKTREE), capture_output=True, text=True, timeout=5,
        )
        ctx["_start_sha"] = _start.stdout.strip() if _start.returncode == 0 else "HEAD~1"
    except Exception:
        ctx["_start_sha"] = "HEAD~1"

    system_prompt = _build_system_prompt(phase, ctx)
    user_prompt = _build_user_prompt(phase, ctx)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Session continuity: restore prior conversation for rework phases
    prior_messages = ctx.get("prior_conversation_messages")
    if prior_messages and isinstance(prior_messages, list) and len(prior_messages) > 2:
        # Restore prior conversation, keeping the NEW system prompt
        # (it may differ from the prior phase's system prompt)
        messages = [messages[0]]  # keep new system prompt
        messages.extend(prior_messages[1:])  # skip old system, keep everything else
        # Append continuation message with rework feedback
        messages.append({
            "role": "user",
            "content": _build_rework_continuation_prompt(phase, ctx),
        })

    tools_schema = _tools_for_phase(phase)
    max_turns = MAX_ITERS.get(phase, DEFAULT_MAX_ITERS)

    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
    num_retries = int(os.environ.get("OPENROUTER_HTTP_RETRIES", "4"))

    # Build request payload for tracing
    request_payload: dict = {
        "model": f"openrouter/{model}",
        "messages": messages[:],
        "tools": tools_schema,
        "temperature": temperature,
    }
    reasoning = _build_reasoning_payload()
    extra_body: dict | None = None
    if reasoning is not None:
        request_payload["reasoning"] = reasoning
        extra_body = {"reasoning": reasoning}

    # Write request artifact
    try:
        req_path = _role_trace_path(output_path, "openrouter_request", ".json")
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    context_budget = _get_context_budget(model)

    total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    conversation_turns: list[dict] = []
    submit_args: dict | None = None
    last_response_text = ""
    last_turn = 0

    for turn in range(1, max_turns + 1):
        last_turn = turn

        # Compact if approaching context window ceiling
        messages, condenser_usage = _compact_messages(messages, context_budget, model=model, api_key=api_key)
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            total_usage[k] += condenser_usage[k]

        litellm_kwargs: dict[str, Any] = {
            "model": f"openrouter/{model}",
            "messages": messages,
            "tools": tools_schema,
            "temperature": temperature,
            "num_retries": num_retries,
            "api_key": api_key,
        }
        if extra_body:
            litellm_kwargs["extra_body"] = extra_body

        try:
            response = litellm.completion(**litellm_kwargs)
        except Exception as e:
            # Write whatever trace we have, then re-raise so the harness sees the failure
            _write_trace_artifacts(
                output_path,
                role=ctx.get("role", ""),
                phase=phase,
                model=model,
                messages=messages,
                conversation_turns=conversation_turns,
                total_usage=total_usage,
                request_payload=request_payload,
                last_response_text=f"API error on turn {turn}: {e}",
                continued_from_prior=bool(ctx.get("prior_conversation_messages")),
            )
            raise

        # Extract usage
        turn_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            turn_usage["input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
            turn_usage["output_tokens"] = getattr(response.usage, "completion_tokens", 0) or 0
            turn_usage["total_tokens"] = getattr(response.usage, "total_tokens", 0) or 0
            if turn_usage["total_tokens"] == 0:
                turn_usage["total_tokens"] = turn_usage["input_tokens"] + turn_usage["output_tokens"]
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            total_usage[k] += turn_usage[k]

        # Extract assistant message
        choice = response.choices[0] if response.choices else None
        if choice is None:
            break
        assistant_message = choice.message

        # Build serializable assistant message dict
        assistant_dict: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        last_response_text = assistant_message.content or ""

        tool_calls_raw = assistant_message.tool_calls
        if tool_calls_raw:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls_raw
            ]

        messages.append(assistant_dict)
        turn_tool_results: list[dict] = []

        if not tool_calls_raw:
            # No tool calls — model is done
            conversation_turns.append({
                "turn": turn,
                "assistant_message": assistant_dict,
                "tool_results": [],
                "usage": turn_usage,
            })
            break

        for tc in tool_calls_raw:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}
            if not isinstance(fn_args, dict):
                fn_args = {}

            tc_id = tc.id

            if fn_name == "submit":
                submit_args = fn_args
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "submit() accepted — session ending.",
                })
                turn_tool_results.append({
                    "tool_call_id": tc_id,
                    "name": "submit",
                    "content": "submit() accepted — session ending.",
                })
                break

            if fn_name == "execute":
                inner_tool = fn_args.get("tool", "")
                inner_args = fn_args.get("args") or {}
                if isinstance(inner_args, str):
                    try:
                        inner_args = json.loads(inner_args)
                    except (json.JSONDecodeError, TypeError):
                        inner_args = {}
                # Fallback: model may flatten args into top-level execute dict
                # e.g. {"tool":"write_file","path":"x","content":"y"} instead of
                #      {"tool":"write_file","args":{"path":"x","content":"y"}}
                if not inner_args and len(fn_args) > 1:
                    inner_args = {k: v for k, v in fn_args.items() if k not in ("tool", "args")}
                result_text = _dispatch_execute(phase, inner_tool, inner_args)
            else:
                # Fallback: model called tool name directly
                result_text = _dispatch_execute(phase, fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_text,
            })
            turn_tool_results.append({
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": result_text,
            })

        conversation_turns.append({
            "turn": turn,
            "assistant_message": assistant_dict,
            "tool_results": turn_tool_results,
            "usage": turn_usage,
        })

        if submit_args is not None:
            break

    # Auto-commit any uncommitted changes for non-bootstrap phases
    auto_commit_msg = ""
    _AUTOCOMMIT_EXCLUDES = [
        ":(exclude).loopbench/",
        ":(exclude)__pycache__/",
        ":(exclude)*.pyc",
        ":(exclude).pytest_cache/",
        ":(exclude).coverage",
        ":(exclude).coverage.*",
        ":(exclude)htmlcov/",
        ":(exclude).mypy_cache/",
        ":(exclude).ruff_cache/",
        ":(exclude)node_modules/",
        ":(exclude).tox/",
        ":(exclude)*.egg-info/",
    ]
    if phase != "bootstrap":
        try:
            # Exclude harness internals and transient test artifacts from auto-commit
            status_out = subprocess.run(
                ["git", "status", "--porcelain", "--", "."] + _AUTOCOMMIT_EXCLUDES,
                cwd=str(WORKTREE),
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if status_out:
                subprocess.run(
                    ["git", "add", "--", "."] + _AUTOCOMMIT_EXCLUDES,
                    cwd=str(WORKTREE),
                    capture_output=True, timeout=10,
                )
                role = ctx.get("role", "agent")
                auto_commit_msg = f"{role}: {phase} changes"
                subprocess.run(
                    ["git", "commit", "-m", auto_commit_msg],
                    cwd=str(WORKTREE),
                    capture_output=True, timeout=10,
                )
        except Exception:
            pass  # best-effort

    # Write trace artifacts
    artifact_paths = _write_trace_artifacts(
        output_path,
        role=ctx.get("role", ""),
        phase=phase,
        model=model,
        messages=messages,
        conversation_turns=conversation_turns,
        total_usage=total_usage,
        request_payload=request_payload,
        last_response_text=last_response_text,
        continued_from_prior=bool(ctx.get("prior_conversation_messages")),
    )

    if phase == "bootstrap" and submit_args is not None:
        result = _build_submit_output(phase, ctx, submit_args, total_usage, last_turn, model)
    elif phase == "bootstrap":
        result = _build_fallback_submit_output(phase, ctx, messages, total_usage, last_turn, model)
    else:
        result = _build_auto_output(phase, ctx, messages, total_usage, last_turn, model, auto_commit_msg)

    result["artifact_paths"] = artifact_paths
    return result


# ── Output Builders (for native tool loop) ────────────────────


def _build_submit_output(
    phase: str,
    ctx: dict,
    submit_args: dict,
    usage: dict,
    turn_count: int,
    model: str,
) -> dict:
    """Map submit tool args to harness output per phase."""
    base = _base_output(phase, ctx)
    base["model"] = model
    base["openrouter_usage"] = usage
    base["openrouter_turn_count"] = turn_count

    if phase == "bootstrap":
        base["plan_markdown"] = submit_args.get("plan_markdown", "")
        base["subtasks"] = submit_args.get("subtasks", [])
        base["summary"] = submit_args.get("summary", "bootstrap plan created")
    elif phase in ("implementation", "rework"):
        base["summary"] = submit_args.get("summary", "")
        base["commit_message"] = submit_args.get("commit_message", "")
        base["changed"] = True
        # PR packet fields — reviewer pipeline depends on these for interaction checks
        base["files_changed"] = submit_args.get("files_changed", "")
        base["key_decisions"] = submit_args.get("key_decisions", "")
        base["tests_run"] = submit_args.get("tests_run", "")
        base["known_risks"] = submit_args.get("known_risks", "")
    elif phase == "finalize":
        base["summary"] = submit_args.get("summary", "")
        cm = submit_args.get("commit_message")
        if cm:
            base["commit_message"] = cm
    else:
        base["summary"] = submit_args.get("summary", "")

    return base


def _build_fallback_submit_output(
    phase: str,
    ctx: dict,
    messages: list[dict],
    usage: dict,
    turn_count: int,
    model: str,
) -> dict:
    """Synthesize output when loop ends without submit."""
    # Find last assistant text
    last_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                last_text = content.strip()
                break

    summary = last_text[:500] if last_text else f"{ctx.get('role', '')} {phase} (budget exhausted)"

    if phase == "bootstrap":
        submit_args = {
            "plan_markdown": last_text or "# Plan\n\nBudget exhausted before plan was submitted.",
            "subtasks": [],
            "summary": summary,
        }
    elif phase in ("implementation", "rework"):
        submit_args = {
            "summary": summary,
            "commit_message": f"{ctx.get('role', '')}: {phase} update (budget exhausted)",
        }
    elif phase == "finalize":
        submit_args = {"summary": summary}
    else:
        submit_args = {"summary": summary}

    return _build_submit_output(phase, ctx, submit_args, usage, turn_count, model)


def _build_auto_output(
    phase: str,
    ctx: dict,
    messages: list[dict],
    usage: dict,
    turn_count: int,
    model: str,
    auto_commit_msg: str,
) -> dict:
    """Build output for non-bootstrap phases (auto-committed, no submit call needed)."""
    base = _base_output(phase, ctx)
    base["model"] = model
    base["openrouter_usage"] = usage
    base["openrouter_turn_count"] = turn_count

    # Find last assistant text for summary
    last_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                last_text = content.strip()
                break

    base["commit_message"] = auto_commit_msg or f"{ctx.get('role', 'agent')}: {phase} changes"

    # Derive changed + files_changed from whether HEAD moved since phase start
    # (covers both auto-commit and model-initiated git_commit calls)
    start_ref = ctx.get("_start_sha")
    head_moved = False
    if start_ref:
        try:
            cur = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(WORKTREE), capture_output=True, text=True, timeout=5,
            )
            head_moved = cur.returncode == 0 and cur.stdout.strip() != start_ref
        except Exception:
            pass

    base["changed"] = head_moved

    if head_moved and start_ref:
        try:
            diff_stat = subprocess.run(
                ["git", "diff", "--name-only", start_ref, "--", ".", ":(exclude).loopbench/", ":(exclude)__pycache__/", ":(exclude)*.pyc"],
                cwd=str(WORKTREE),
                capture_output=True, text=True, timeout=10,
            )
            if diff_stat.returncode == 0 and diff_stat.stdout.strip():
                base["files_changed"] = [f for f in diff_stat.stdout.strip().split("\n") if f]
            else:
                base["files_changed"] = []
        except Exception:
            base["files_changed"] = []
    else:
        base["files_changed"] = []

    # Build structured summary from commit data, not chat messages
    parts = []
    if base.get("files_changed"):
        parts.append(f"Changed: {', '.join(base['files_changed'][:10])}")
    if auto_commit_msg and auto_commit_msg != f"{ctx.get('role', 'agent')}: {phase} changes":
        parts.append(f"Commit: {auto_commit_msg}")
    if last_text:
        # Take first sentence only, strip chain-of-thought
        first_line = last_text.split('\n')[0].strip()
        if not first_line.lower().startswith(("let me", "now i", "i'll", "i need")):
            parts.append(first_line[:200])
    base["summary"] = " | ".join(parts)[:500] if parts else f"{ctx.get('role', '')} {phase}"

    base["key_decisions"] = ""
    base["tests_run"] = ""
    base["known_risks"] = ""

    return base


# ── Phase Dispatch ────────────────────────────────────────────


def _run_phase(phase: str, ctx: dict, output_path: Path, model: str, api_key: str) -> dict:
    """Dispatch to the right scaffold per phase."""

    if phase == "bootstrap":
        return _run_tool_loop(phase, ctx, output_path, model, api_key)

    if phase in ("implementation", "rework"):
        return _run_tool_loop(phase, ctx, output_path, model, api_key)

    if phase in ("review", "review_verify"):
        return _run_review_pipeline(ctx)

    if phase == "reflect":
        with dspy.context(temperature=0.6):
            pred = dspy.Predict(AdversarialReflect)(
                round_summary=_round_summary(ctx),
                current_knowledge=_read_all_knowledge(),
                prior_causes=_prior_causes(ctx),
            )
        return _base_output(phase, ctx, **{
            "directive": pred.directive,
            "task_understanding": pred.task_understanding,
            "failure_patterns": pred.failure_patterns,
            "workflow_insights": pred.workflow_insights,
        })

    if phase == "finalize":
        return _run_tool_loop(phase, ctx, output_path, model, api_key)

    raise ValueError(f"Unknown phase: {phase}")


def _run_review_pipeline(ctx: dict) -> dict:
    """Run the orchestrator review pipeline (not ReAct)."""
    phase = ctx.get("phase", "review")
    requirements = _read_task_readme()

    # Augment requirements with the specific subtask acceptance criteria
    # so the reviewer judges against the actual assignment, not just
    # the README feature list (which describes pre-existing features).
    acceptance_parts: list[str] = []
    # Track assigned paths per role for coverage checking
    assigned_paths_by_role: dict[str, list[str]] = {}
    for msg in (ctx.get("implementation_messages") or []):
        body = msg.get("body", {})
        if not isinstance(body, dict):
            continue
        subtask = body.get("subtask", {})
        acceptance = subtask.get("acceptance", "")
        title = subtask.get("title", "")
        role = subtask.get("role", "")
        paths = subtask.get("paths", [])
        if acceptance and title:
            paths_str = f" (assigned paths: {', '.join(paths)})" if paths else ""
            acceptance_parts.append(f"- [{role}] {title}: {acceptance}{paths_str}")
        if role and paths:
            assigned_paths_by_role.setdefault(role, []).extend(paths)
    if acceptance_parts:
        requirements += "\n\n## Acceptance Criteria (what must pass)\n" + "\n".join(acceptance_parts)

    coder_commits = ctx.get("coder_commits", {})
    harness_issues = ctx.get("harness_issues", [])
    if not isinstance(harness_issues, list):
        harness_issues = []

    # Inject time urgency — reviewer must understand that endless rework
    # loops burn time. Accepting partial progress beats timing out.
    deadline = ctx.get("run_deadline_epoch")
    if deadline:
        import time as _time
        remaining = max(0, int(deadline - _time.time()))
        if remaining < 180:
            requirements += (
                f"\n\nURGENT TIME CONSTRAINT: Only ~{remaining}s left. "
                "A timeout with no accepted code is the WORST outcome — score = 0. "
                "ACCEPT whatever is mergeable NOW. Do NOT request more rework."
            )
        elif remaining < 600:
            requirements += (
                f"\n\nTIME PRESSURE: ~{remaining // 60}min remaining. "
                "Each rework round costs ~3-5 min. If you request rework, the coders may "
                "not finish before the deadline. Prefer merging partial progress over "
                "requesting another round. A timeout with no code = score of zero."
            )

    # Build PR packets from coder outputs (structured metadata)
    packets = _build_pr_packets(ctx)
    # If no packets available (old driver), create minimal ones from commit info
    if not packets:
        for coder in coder_commits:
            packets[coder] = {"summary": f"{coder} commits available", "files_changed": []}


    reviewer = OrchestratorReview()
    pred = reviewer(
        requirements=requirements,
        coder_packets=packets,
        coder_commits=coder_commits if isinstance(coder_commits, dict) else {},
        harness_issues=harness_issues if isinstance(harness_issues, list) else [],
        review_ledger=ctx.get("review_ledger", {}),
        scratch_merge_results=ctx.get("scratch_merge_results", {}),
        inline_diffs=ctx.get("inline_diffs"),
        already_merged_diffs=ctx.get("already_merged_diffs"),
    )

    merge_commits = _safe_json(getattr(pred, "merge_commits_json", "{}"))
    coder_feedback = _safe_json(getattr(pred, "coder_feedback_json", "{}"))

    # Fallback: if reviewer returned empty merge_commits but scratch_merge shows
    # conflict-free commits, auto-nominate them. The LLM often mistakenly thinks
    # request_rework=true means "merge nothing," losing all progress.
    if not merge_commits or (isinstance(merge_commits, dict) and not any(merge_commits.values())):
        smr = ctx.get("scratch_merge_results", {})
        commits_tested = smr.get("commits_tested", {}) if isinstance(smr, dict) else {}
        auto_nominated: dict[str, list[str]] = {}
        for sha, info in commits_tested.items():
            if isinstance(info, dict) and info.get("ok"):
                coder_name = info.get("coder", "")
                if coder_name:
                    auto_nominated.setdefault(coder_name, []).append(sha)
        if auto_nominated:
            print(
                f"[review] auto-nominating {sum(len(v) for v in auto_nominated.values())} "
                f"conflict-free commit(s) that reviewer failed to nominate: {auto_nominated}",
                file=sys.stderr,
            )
            merge_commits = auto_nominated
            # Mark as inspected — the DSPy AnalyzeDiff already reviewed the
            # inline_diffs for these commits; they just weren't explicitly
            # shown via git_show tool calls.
            for shas in auto_nominated.values():
                _mark_inspected(shas)

    # Map inspected commits to role-based structure
    all_inspected = _inspected_commits.get("_all", set())
    inspected_by_role: dict[str, list[str]] = {}
    if isinstance(merge_commits, dict):
        for role_name, shas in merge_commits.items():
            inspected_by_role[role_name] = [
                s for s in (shas if isinstance(shas, list) else []) if s in all_inspected
            ]

    return _base_output(phase, ctx, **{
        "summary": getattr(pred, "summary", ""),
        "merge_commits": merge_commits if isinstance(merge_commits, dict) else {},
        "request_rework": getattr(pred, "request_rework", False),
        "coder_feedback": coder_feedback if isinstance(coder_feedback, dict) else {},
        "inspected_commits": inspected_by_role,
    })


# ── Input Builders ────────────────────────────────────────────


def _bootstrap_inputs(ctx: dict) -> dict:
    coders = ctx.get("coders", ["coder_a", "coder_b"])
    desc = _read_task_readme()
    if ctx.get("is_rebootstrap"):
        ledger = ctx.get("review_ledger", {})
        desc = (
            "IMPORTANT: The previous plan failed after 3+ rework rounds with no progress.\n"
            f"Prior decisions: {json.dumps(ledger.get('entries', []), indent=2)}\n"
            "You must change your approach — different file decomposition, different strategy.\n\n"
            + desc
        )
    return {
        "task_description": desc,
        "coders": ", ".join(coders),
        "repo_tree": _tracked_files(),
    }


def _coder_inputs(ctx: dict) -> dict:
    claimed = ctx.get("claimed_task", {})
    title = claimed.get("title", "implementation task")
    acceptance = claimed.get("acceptance", "")
    paths = claimed.get("paths", [])
    assignment = title
    if acceptance:
        assignment += f"\nAcceptance: {acceptance}"
    if paths:
        assignment += f"\nAssigned paths: {', '.join(paths)}"
    feedback_by_role = ctx.get("planner_feedback_by_role", {})
    role = ctx.get("role", "")
    feedback = feedback_by_role.get(role, "") if isinstance(feedback_by_role, dict) else ""
    if not feedback:
        feedback = "First implementation — no prior feedback."
    # Cross-coder visibility: append peer progress summaries
    peer = ctx.get("peer_progress", {})
    peer_lines = []
    for peer_role, info in peer.items():
        if peer_role == role:
            continue
        files = info.get("files_changed", [])
        files_str = ", ".join(files[:5]) if isinstance(files, list) else str(files)
        peer_lines.append(f"- {peer_role}: {info.get('summary', '(no summary)')} [files: {files_str}]")
    if peer_lines:
        feedback += "\n\nPeer progress:\n" + "\n".join(peer_lines)
    return {
        "task_description": _read_task_readme(),
        "assignment": assignment,
        "plan": ctx.get("planner_summary", ""),
        "feedback": feedback,
        "current_changes": ctx.get("own_diff", ""),
    }


def _finalize_inputs(ctx: dict) -> dict:
    return {
        "task_description": _read_task_readme(),
        "status": json.dumps(ctx.get("coordination_summary", {}), indent=2),
    }


# ── Output Extraction ─────────────────────────────────────────


def _base_output(phase: str, ctx: dict, **extra: Any) -> dict:
    out: dict[str, Any] = {
        "status": "completed",
        "role": ctx.get("role", ""),
        "phase": phase,
        "execution_mode": "agentic_tool_loop",
        "failed_commands": _failed_commands[-10:],
    }
    out.update(extra)
    return out



def _build_reasoning_payload() -> dict | None:
    """Build OpenRouter reasoning config from env vars (matches openrouter driver)."""
    payload: dict[str, Any] = {}
    for env_name, key in (
        ("OPENROUTER_REASONING_ENABLED", "enabled"),
        ("OPENROUTER_REASONING_EXCLUDE", "exclude"),
    ):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        val = raw.strip().lower()
        if val in ("1", "true", "yes", "on"):
            payload[key] = True
        elif val in ("0", "false", "no", "off"):
            payload[key] = False
    effort = (os.environ.get("OPENROUTER_REASONING_EFFORT") or "").strip()
    if effort:
        payload["effort"] = effort
    return payload or None


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    global WORKTREE, KNOWLEDGE_DIR

    context_path = Path(os.environ["LB_CONTEXT_JSON"])
    output_path = Path(os.environ["LB_OUTPUT_JSON"])
    WORKTREE = Path(os.environ.get("LB_WORKTREE", "."))
    model = os.environ.get("LB_MODEL", "moonshotai/kimi-k2.5")
    phase = os.environ.get("LB_PHASE", "implementation")

    ctx = json.loads(context_path.read_text(encoding="utf-8"))
    ctx["phase"] = phase  # ensure available

    # Knowledge dir
    kt = ctx.get("knowledge_tool", {})
    if isinstance(kt, dict) and kt.get("knowledge_dir"):
        kd = Path(kt["knowledge_dir"])
        if kd.is_dir():
            KNOWLEDGE_DIR = kd

    # Configure DSPy → LiteLLM → OpenRouter
    api_key_env = os.environ.get("OPENROUTER_API_KEY_ENV", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"Error: {api_key_env} not set", file=sys.stderr)
        sys.exit(1)

    lm_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096")),
        "num_retries": int(os.environ.get("OPENROUTER_HTTP_RETRIES", "4")),
        "cache": False,
    }
    reasoning = _build_reasoning_payload()
    if reasoning is not None:
        lm_kwargs["extra_body"] = {"reasoning": reasoning}
    lm = dspy.LM(f"openrouter/{model}", **lm_kwargs)
    dspy.configure(lm=lm)

    try:
        with dspy.track_usage() as usage_tracker:
            output = _run_phase(phase, ctx, output_path, model, api_key)
    except Exception as e:
        output = {"status": "error", "error": str(e), "phase": phase, "role": ctx.get("role", "")}

    # Inject model name — tool loop phases set this already; DSPy Predict phases need it.
    output.setdefault("model", model)

    # Merge DSPy-tracked usage (from review/reflect Predict phases) into output.
    try:
        totals = usage_tracker.get_total_tokens()
        # totals: {"openrouter/model": {"prompt_tokens": N, "completion_tokens": N, ...}}
        dspy_combined: dict[str, int] = {}
        for _lm, entry in totals.items():
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                dspy_combined[k] = dspy_combined.get(k, 0) + int(entry.get(k, 0) or 0)
        if any(v > 0 for v in dspy_combined.values()):
            # Merge with existing openrouter_usage if present (tool loop already set it)
            existing = output.get("openrouter_usage", {})
            if existing:
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    # Map prompt_tokens -> input_tokens for consistency
                    src_key = k
                    dst_key = k.replace("prompt_tokens", "input_tokens").replace("completion_tokens", "output_tokens")
                    existing[dst_key] = existing.get(dst_key, 0) + dspy_combined.get(src_key, 0)
            else:
                output["openrouter_usage"] = dspy_combined
    except Exception:  # noqa: BLE001
        pass

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
