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
from pydantic import BaseModel, Field

# ── Globals (set in main) ────────────────────────────────────

WORKTREE = Path(".")
KNOWLEDGE_DIR: Path | None = None
_inspected_commits: dict[str, set[str]] = {}
_failed_commands: list[dict[str, Any]] = []

MAX_ITERS = {"bootstrap": 15, "reflect": 8, "finalize": 10}
DEFAULT_MAX_ITERS = 25


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
            timeout=min(timeout_sec, 120),
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
    return exec_command("git status --short")


def git_diff(ref: str = "HEAD") -> str:
    """Show unified diff versus HEAD or a specific ref."""
    return exec_command(f"git diff {ref}")


def git_add(paths: str = ".") -> str:
    """Stage files for commit."""
    return exec_command(f"git add {paths}")


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
    """Decide which commits to merge based on per-coder findings and interaction analysis."""

    all_findings: str = dspy.InputField(desc="Per-coder analysis results")
    interactions: str = dspy.InputField(desc="Cross-coder interaction findings")
    available_commits: str = dspy.InputField(desc="Commit SHAs available per coder")
    harness_issues: str = dspy.InputField(desc="Issues detected by harness")
    review_ledger: str = dspy.InputField(desc="Prior decisions with causal reasoning (why rework was needed)")
    scratch_merge_results: str = dspy.InputField(desc="Deterministic merge feasibility per commit")

    merge_commits_json: str = dspy.OutputField(
        desc='JSON: {"coder_a": ["sha"], "coder_b": []} — commits to merge'
    )
    request_rework: bool = dspy.OutputField(desc="Whether rework is needed")
    reasoning: str = dspy.OutputField(desc="Explanation of the decision")


class WriteFeedback(dspy.Signature):
    """Write actionable feedback for a coder who needs rework.
    Be specific: exact files, exact changes, exact acceptance test."""

    coder_findings: str = dspy.InputField(desc="This coder's review findings")
    decision_reasoning: str = dspy.InputField(desc="Why rework was requested")
    requirements: str = dspy.InputField(desc="Task requirements")

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
    ) -> dspy.Prediction:
        # ── Stage 1: Per-coder analysis (map) ──
        # For each coder, inspect their commits via git tools,
        # then run a bounded LLM analysis.
        findings: dict[str, Any] = {}
        for coder, packet in coder_packets.items():
            # Retrieve key hunks via git (tool-grounded, not LLM)
            shas = coder_commits.get(coder, [])
            hunks = self._inspect_commits(shas)
            _mark_inspected(shas)

            findings[coder] = self.analyze(
                requirements=requirements,
                pr_packet=json.dumps(packet, indent=2),
                diff_hunks=hunks[:6000],
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
        decision = self.decide(
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
            available_commits=json.dumps(coder_commits, indent=2),
            harness_issues="\n".join(f"- {i}" for i in harness_issues) if harness_issues else "None",
            review_ledger=json.dumps(review_ledger or {}, indent=2),
            scratch_merge_results=json.dumps(scratch_merge_results or {}, indent=2),
        )

        # ── Stage 5: Feedback (per-coder, only if rework needed) ──
        coder_feedback: dict[str, str] = {}
        if decision.request_rework:
            for coder, finding in findings.items():
                fb = self.write_feedback(
                    coder_findings=f"Gaps: {finding.requirements_gaps}\nIssues: {finding.issues_json}",
                    decision_reasoning=decision.reasoning,
                    requirements=requirements,
                )
                coder_feedback[coder] = (
                    f"Required changes:\n{fb.required_changes}\n"
                    f"Locations: {fb.locations}\n"
                    f"Done when: {fb.acceptance_test}"
                )

        return dspy.Prediction(
            summary=decision.reasoning,
            merge_commits_json=decision.merge_commits_json,
            request_rework=decision.request_rework,
            coder_feedback_json=json.dumps(coder_feedback),
        )

    def _inspect_commits(self, shas: list[str]) -> str:
        """Git-grounded: actually inspect the commits via tools."""
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
            out = outputs[-1] if outputs else {}
        elif isinstance(outputs, dict):
            out = outputs
        else:
            out = {}

        packets[coder] = {
            "summary": out.get("summary", ""),
            "files_changed": _parse_list(out.get("files_changed", "")),
            "key_decisions": _parse_list(out.get("key_decisions", "")),
            "tests_run": _parse_list(out.get("tests_run", "")),
            "known_risks": _parse_list(out.get("known_risks", "")),
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


# ── Phase Dispatch ────────────────────────────────────────────

COMMON = [list_files, read_file, exec_command, git_status, git_diff, llm_query]
CODER = COMMON + [write_file, patch_file, git_add, git_commit]


def _run_phase(phase: str, ctx: dict) -> dict:
    """Dispatch to the right scaffold per phase."""

    if phase == "bootstrap":
        return _run_react(Bootstrap, COMMON, ctx, _bootstrap_inputs(ctx))

    if phase in ("implementation", "rework"):
        return _run_react(Implement, CODER, ctx, _coder_inputs(ctx))

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
        return _run_react(Finalize, CODER, ctx, _finalize_inputs(ctx))

    raise ValueError(f"Unknown phase: {phase}")


def _run_react(sig, tools, ctx, inputs) -> dict:
    """Run a standard ReAct agent and extract output."""
    if KNOWLEDGE_DIR is not None:
        tools = tools + [read_knowledge]
    max_iters = MAX_ITERS.get(ctx.get("phase", ""), DEFAULT_MAX_ITERS)
    agent = dspy.ReAct(sig, tools=tools, max_iters=max_iters)
    pred = agent(**inputs)
    return _extract_react_output(ctx.get("phase", ""), pred, ctx)


def _run_review_pipeline(ctx: dict) -> dict:
    """Run the orchestrator review pipeline (not ReAct)."""
    phase = ctx.get("phase", "review")
    requirements = _read_task_readme()
    coder_commits = ctx.get("coder_commits", {})
    harness_issues = ctx.get("harness_issues", [])

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
    )

    merge_commits = _safe_json(getattr(pred, "merge_commits_json", "{}"))
    coder_feedback = _safe_json(getattr(pred, "coder_feedback_json", "{}"))

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


def _extract_react_output(phase: str, pred: dspy.Prediction, ctx: dict) -> dict:
    if phase == "bootstrap":
        subtasks = _safe_json(getattr(pred, "subtasks_json", "[]"), [])
        return _base_output(phase, ctx,
            plan_markdown=getattr(pred, "plan_markdown", ""),
            subtasks=subtasks if isinstance(subtasks, list) else [],
            summary=getattr(pred, "summary", ""),
        )

    if phase in ("implementation", "rework"):
        return _base_output(phase, ctx,
            summary=getattr(pred, "summary", ""),
            commit_message=getattr(pred, "commit_message", ""),
            changed=True,
            # PR packet fields
            files_changed=getattr(pred, "files_changed", ""),
            key_decisions=getattr(pred, "key_decisions", ""),
            tests_run=getattr(pred, "tests_run", ""),
            known_risks=getattr(pred, "known_risks", ""),
        )

    if phase == "finalize":
        return _base_output(phase, ctx, summary=getattr(pred, "summary", ""))

    return _base_output(phase, ctx, summary=getattr(pred, "summary", ""))


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
        output = _run_phase(phase, ctx)
    except Exception as e:
        output = {"status": "error", "error": str(e), "phase": phase, "role": ctx.get("role", "")}

    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
