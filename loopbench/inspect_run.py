"""
loopbench.inspect_run

Artifact viewer for benchmark runs — renders the full run narrative
as rich terminal panels or a markdown report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    wall_clock_sec: float = 0.0
    review_iterations: int = 0
    merge_conflicts: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0
    public_validate_policy: str = ""
    public_validation_attempts: int = 0
    llm_tokens_by_role: Dict[str, Any] = field(default_factory=dict)
    llm_tokens_by_phase: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subtask:
    id: str
    role: str
    title: str


@dataclass
class PlanSummary:
    text: str = ""
    subtasks: List[Subtask] = field(default_factory=list)


@dataclass
class ReviewRound:
    round_index: int = 0
    accepted: bool = False
    force_rework: bool = False
    request_rework: bool = False
    merge_ok: bool = False
    review_summary: str = ""
    review_notes: str = ""
    verify_summary: str = ""
    verify_notes: str = ""
    merged_commits_this_round: Dict[str, List[str]] = field(default_factory=dict)
    public_validation_ok: bool = False
    public_validation_state: str = ""
    public_validation_policy: str = ""


@dataclass
class ReflectionSnapshot:
    round_index: int = 0
    directive: str = ""
    task_understanding: str = ""
    failure_patterns: str = ""
    workflow_insights: str = ""
    superseded: List[str] = field(default_factory=list)


@dataclass
class DiffSummary:
    final_patch: str = ""
    per_role_changed_files: Dict[str, List[str]] = field(default_factory=dict)
    per_role_diffs: Dict[str, str] = field(default_factory=dict)


@dataclass
class HodoscopeStatus:
    analysis_path: str = ""
    trajectory_count: int = 0
    summary_count: int = 0
    has_embeddings: bool = False


@dataclass
class InspectResult:
    run_id: str = ""
    task_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    public_pass: Optional[bool] = None
    hidden_pass: Optional[bool] = None
    roles: List[str] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    plan: PlanSummary = field(default_factory=PlanSummary)
    review_rounds: List[ReviewRound] = field(default_factory=list)
    reflections: List[ReflectionSnapshot] = field(default_factory=list)
    diff: DiffSummary = field(default_factory=DiffSummary)
    hodoscope: Optional[HodoscopeStatus] = None
    trace_url: Optional[str] = None
    open_questions: str = ""
    status_log: str = ""
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inspector — reads artifacts into InspectResult
# ---------------------------------------------------------------------------

# Matches the ```json ... ``` block written by run_artifacts.py:70-73
_JSON_BLOCK_RE = re.compile(
    r"^## output\s*\n```json\s*\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


class RunInspector:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()

    def inspect(self) -> InspectResult:
        result = InspectResult()
        self._load_manifest(result)
        self._load_plan(result)
        self._load_review_rounds(result)
        self._load_reflections(result)
        self._load_diffs(result)
        self._load_hodoscope(result)
        self._load_trace_url(result)
        self._load_extras(result)
        return result

    # -- manifest -----------------------------------------------------------

    def _load_manifest(self, result: InspectResult) -> None:
        path = self.run_dir / "manifest.json"
        if not path.exists():
            result.errors.append("manifest.json not found")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            result.errors.append(f"manifest.json parse error: {exc}")
            return

        result.run_id = data.get("run_id", "")
        result.task_id = data.get("task_id", "")
        result.started_at = data.get("started_at", "")
        result.finished_at = data.get("finished_at", "")
        result.public_pass = data.get("public_pass")
        result.hidden_pass = data.get("hidden_pass")
        result.roles = data.get("roles", [])

        m = data.get("metrics", {})
        result.metrics = RunMetrics(
            wall_clock_sec=m.get("wall_clock_sec", 0.0),
            review_iterations=m.get("review_iterations", 0),
            merge_conflicts=m.get("merge_conflicts", 0),
            llm_input_tokens=m.get("llm_input_tokens", 0),
            llm_output_tokens=m.get("llm_output_tokens", 0),
            llm_total_tokens=m.get("llm_total_tokens", 0),
            public_validate_policy=m.get("public_validate_policy", ""),
            public_validation_attempts=m.get("public_validation_attempts", 0),
            llm_tokens_by_role=m.get("llm_tokens_by_role", {}),
            llm_tokens_by_phase=m.get("llm_tokens_by_phase", {}),
            raw=m,
        )

    # -- plan ---------------------------------------------------------------

    def _load_plan(self, result: InspectResult) -> None:
        plan_path = self.run_dir / "plans" / "plan.md"
        subtasks_path = self.run_dir / "plans" / "subtasks.yaml"

        plan_text = self._read(plan_path)
        subtasks: List[Subtask] = []

        if subtasks_path.exists():
            try:
                raw = yaml.safe_load(subtasks_path.read_text(encoding="utf-8"))
                for st in raw.get("subtasks", []) if isinstance(raw, dict) else []:
                    if isinstance(st, dict):
                        subtasks.append(Subtask(
                            id=str(st.get("id", "?")),
                            role=str(st.get("role", "?")),
                            title=str(st.get("title", "")),
                        ))
            except Exception as exc:
                result.errors.append(f"subtasks.yaml parse error: {exc}")

        result.plan = PlanSummary(text=plan_text, subtasks=subtasks)

    # -- review rounds ------------------------------------------------------

    def _load_review_rounds(self, result: InspectResult) -> None:
        audit_dir = self.run_dir / "artifacts" / "review_audit"
        if not audit_dir.is_dir():
            return

        paths = sorted(audit_dir.glob("round_*.json"))
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                result.errors.append(f"review audit parse error ({path.name}): {exc}")
                continue

            pv = data.get("public_validation", {})
            result.review_rounds.append(ReviewRound(
                round_index=data.get("round_index", 0),
                accepted=data.get("accepted", False),
                force_rework=data.get("force_rework", False),
                request_rework=data.get("request_rework", False),
                merge_ok=data.get("merge_ok", False),
                review_summary=data.get("review_summary", ""),
                review_notes=data.get("review_notes", ""),
                verify_summary=data.get("verify_summary", ""),
                verify_notes=data.get("verify_notes", ""),
                merged_commits_this_round=data.get("merged_commits_this_round", {}),
                public_validation_ok=pv.get("ok", False),
                public_validation_state=pv.get("state", ""),
                public_validation_policy=pv.get("policy", ""),
            ))

    # -- reflections --------------------------------------------------------

    def _load_reflections(self, result: InspectResult) -> None:
        summaries_dir = self.run_dir / "role_summaries"
        if not summaries_dir.is_dir():
            return

        paths = sorted(summaries_dir.glob("*_reflect_round_*.md"))
        for path in paths:
            text = self._read(path)
            if not text:
                continue

            m = _JSON_BLOCK_RE.search(text)
            if not m:
                result.errors.append(f"reflection parse: no JSON block in {path.name}")
                continue

            try:
                data = json.loads(m.group(1))
            except Exception as exc:
                result.errors.append(f"reflection JSON error ({path.name}): {exc}")
                continue

            # Extract round index from filename (*_reflect_round_N.md)
            round_match = re.search(r"_reflect_round_(\d+)", path.stem)
            round_idx = int(round_match.group(1)) if round_match else 0

            result.reflections.append(ReflectionSnapshot(
                round_index=round_idx,
                directive=data.get("directive", ""),
                task_understanding=data.get("task_understanding", ""),
                failure_patterns=data.get("failure_patterns", ""),
                workflow_insights=data.get("workflow_insights", ""),
                superseded=data.get("superseded", []),
            ))

    # -- diffs --------------------------------------------------------------

    def _load_diffs(self, result: InspectResult) -> None:
        final_patch_path = self.run_dir / "final.patch"
        result.diff.final_patch = self._read(final_patch_path)

        repo_state = self.run_dir / "repo_state"
        if not repo_state.is_dir():
            return

        for role_dir in sorted(repo_state.iterdir()):
            if not role_dir.is_dir() or role_dir.name == "base_commit.txt":
                continue
            role = role_dir.name

            changed_files_path = role_dir / "changed_files.txt"
            if changed_files_path.exists():
                lines = changed_files_path.read_text(encoding="utf-8").strip().splitlines()
                result.diff.per_role_changed_files[role] = lines

            diff_path = role_dir / "diff_from_base.patch"
            if diff_path.exists():
                result.diff.per_role_diffs[role] = diff_path.read_text(encoding="utf-8")

    # -- hodoscope ----------------------------------------------------------

    def _load_hodoscope(self, result: InspectResult) -> None:
        traj_dir = self.run_dir / "hodoscope" / "trajectories"
        analysis_path = self.run_dir / "hodoscope" / "analysis.hodoscope.json"

        if not traj_dir.is_dir():
            return

        trajectory_count = len(list(traj_dir.glob("*.json")))
        if trajectory_count == 0:
            return

        status = HodoscopeStatus(trajectory_count=trajectory_count)

        if analysis_path.exists():
            status.analysis_path = str(analysis_path)
            try:
                data = json.loads(analysis_path.read_text(encoding="utf-8"))
                summaries = data.get("summaries", [])
                status.summary_count = len(summaries)
                status.has_embeddings = any(
                    s.get("embedding") is not None for s in summaries[:10]
                )
            except Exception:
                pass

        result.hodoscope = status

    # -- trace url ----------------------------------------------------------

    def _load_trace_url(self, result: InspectResult) -> None:
        path = self.run_dir / "trace" / "session.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            url = data.get("trace_url")
            if url:
                result.trace_url = url
        except Exception:
            pass

    # -- extras (status, open_questions) ------------------------------------

    def _load_extras(self, result: InspectResult) -> None:
        result.status_log = self._read(self.run_dir / "status.md")
        result.open_questions = self._read(self.run_dir / "open_questions.md")

    # -- helpers ------------------------------------------------------------

    def _read(self, path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Rich renderer
# ---------------------------------------------------------------------------


class RichRenderer:
    def __init__(self, diff_lines: int = 60):
        self.diff_lines = diff_lines

    def render(self, result: InspectResult) -> None:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.text import Text

        console = Console()

        # 1. Header
        title = Text(result.run_id or "(unknown run)", style="bold cyan")
        badges = Text()
        badges.append("public:", style="dim")
        badges.append(
            " PASS " if result.public_pass else " FAIL ",
            style="bold white on green" if result.public_pass else "bold white on red",
        )
        badges.append("  hidden:", style="dim")
        badges.append(
            " PASS " if result.hidden_pass else " FAIL ",
            style="bold white on green" if result.hidden_pass else "bold white on red",
        )

        header_lines = Text()
        header_lines.append(f"task_id: {result.task_id}\n")
        header_lines.append(f"started: {result.started_at}\n")
        header_lines.append(f"finished: {result.finished_at}\n")
        header_lines.append(f"roles: {', '.join(result.roles)}\n")
        header_lines.append_text(badges)

        console.print(Panel(header_lines, title=title, border_style="cyan"))

        # 2. Metrics
        m = result.metrics
        metrics_table = Table(title="Metrics", show_header=True, header_style="bold")
        metrics_table.add_column("Metric", style="dim")
        metrics_table.add_column("Value")
        metrics_table.add_row("Wall clock", f"{m.wall_clock_sec:.0f}s")
        metrics_table.add_row("Review iterations", str(m.review_iterations))
        metrics_table.add_row("Merge conflicts", str(m.merge_conflicts))
        metrics_table.add_row("LLM tokens (in/out/total)", f"{m.llm_input_tokens:,} / {m.llm_output_tokens:,} / {m.llm_total_tokens:,}")
        metrics_table.add_row("Public validate policy", m.public_validate_policy or "n/a")
        metrics_table.add_row("Public validation attempts", str(m.public_validation_attempts))
        console.print(metrics_table)

        # 3. Plan
        if result.plan.text:
            console.print(Panel(Markdown(result.plan.text), title="Plan", border_style="green"))

        if result.plan.subtasks:
            st_table = Table(title="Subtasks", show_header=True, header_style="bold")
            st_table.add_column("ID")
            st_table.add_column("Role")
            st_table.add_column("Title")
            for st in result.plan.subtasks:
                st_table.add_row(st.id, st.role, st.title)
            console.print(st_table)

        # 4. Review Story
        if result.review_rounds:
            console.print(Text("\nReview Story", style="bold underline"))
            for rr in result.review_rounds:
                outcome = "ACCEPTED" if rr.accepted else ("REWORK" if rr.request_rework else "REJECTED")
                outcome_style = "green" if rr.accepted else "yellow"
                rr_title = Text()
                rr_title.append(f"Round {rr.round_index}", style="bold")
                rr_title.append(f"  [{outcome}]", style=outcome_style)

                lines = []
                if rr.review_summary:
                    lines.append(f"Review: {rr.review_summary}")
                if rr.review_notes:
                    lines.append(f"  Notes: {rr.review_notes}")
                if rr.verify_summary:
                    lines.append(f"Verify: {rr.verify_summary}")
                if rr.verify_notes:
                    lines.append(f"  Notes: {rr.verify_notes}")
                if rr.merged_commits_this_round:
                    for role, commits in rr.merged_commits_this_round.items():
                        lines.append(f"Merged ({role}): {', '.join(commits[:5])}")
                lines.append(f"Public validation: {rr.public_validation_state} (policy={rr.public_validation_policy})")
                lines.append(f"Merge OK: {rr.merge_ok}")

                console.print(Panel("\n".join(lines), title=rr_title, border_style="blue"))

        # 5. Reflection Evolution
        if result.reflections:
            console.print(Text("\nReflection Evolution", style="bold underline"))
            for ref in result.reflections:
                ref_table = Table(
                    title=f"Round {ref.round_index}",
                    show_header=True,
                    header_style="bold",
                )
                ref_table.add_column("Surface", style="dim", width=20)
                ref_table.add_column("Content", overflow="fold")
                if ref.directive:
                    ref_table.add_row("directive", ref.directive)
                if ref.task_understanding:
                    ref_table.add_row("task_understanding", ref.task_understanding)
                if ref.failure_patterns:
                    ref_table.add_row("failure_patterns", ref.failure_patterns)
                if ref.workflow_insights:
                    ref_table.add_row("workflow_insights", ref.workflow_insights)
                if ref.superseded:
                    ref_table.add_row("superseded", ", ".join(ref.superseded))
                console.print(ref_table)

        # 6. Hodoscope
        if result.hodoscope:
            h = result.hodoscope
            hodo_table = Table(title="Hodoscope", show_header=True, header_style="bold")
            hodo_table.add_column("Field", style="dim")
            hodo_table.add_column("Value")
            hodo_table.add_row("Trajectories", str(h.trajectory_count))
            hodo_table.add_row("Action summaries", str(h.summary_count))
            hodo_table.add_row("Embeddings", "yes" if h.has_embeddings else "no")
            if h.analysis_path:
                hodo_table.add_row("Analysis", h.analysis_path)
            console.print(hodo_table)

        # 7. Final Diff
        if result.diff.final_patch:
            patch_lines = result.diff.final_patch.splitlines()
            truncated = len(patch_lines) > self.diff_lines
            display_patch = "\n".join(patch_lines[: self.diff_lines])
            if truncated:
                display_patch += f"\n... ({len(patch_lines) - self.diff_lines} more lines)"
            console.print(Panel(
                Syntax(display_patch, "diff", theme="monokai"),
                title="Final Diff",
                border_style="magenta",
            ))

        if result.diff.per_role_changed_files:
            cf_table = Table(title="Changed Files by Role", show_header=True, header_style="bold")
            cf_table.add_column("Role")
            cf_table.add_column("Files")
            for role, files in result.diff.per_role_changed_files.items():
                cf_table.add_row(role, "\n".join(files))
            console.print(cf_table)

        # 8. Trace
        if result.trace_url:
            console.print(Panel(result.trace_url, title="LangSmith Trace", border_style="yellow"))

        # 9. Issues
        has_issues = result.open_questions or result.errors
        if has_issues:
            issue_parts = []
            if result.open_questions:
                issue_parts.append(f"Open Questions:\n{result.open_questions}")
            if result.errors:
                issue_parts.append("Parse Errors:\n" + "\n".join(f"  - {e}" for e in result.errors))
            console.print(Panel("\n\n".join(issue_parts), title="Issues", border_style="red"))


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


class MarkdownRenderer:
    def __init__(self, diff_lines: int = 60):
        self.diff_lines = diff_lines

    def render(self, result: InspectResult, run_dir: Path) -> Path:
        sections: List[str] = []

        # 1. Header
        pub = "PASS" if result.public_pass else "FAIL"
        hid = "PASS" if result.hidden_pass else "FAIL"
        sections.append(
            f"# Inspect: {result.run_id}\n\n"
            f"| Field | Value |\n|---|---|\n"
            f"| task_id | {result.task_id} |\n"
            f"| started | {result.started_at} |\n"
            f"| finished | {result.finished_at} |\n"
            f"| roles | {', '.join(result.roles)} |\n"
            f"| public | **{pub}** |\n"
            f"| hidden | **{hid}** |\n"
        )

        # 2. Metrics
        m = result.metrics
        sections.append(
            "## Metrics\n\n"
            "| Metric | Value |\n|---|---|\n"
            f"| Wall clock | {m.wall_clock_sec:.0f}s |\n"
            f"| Review iterations | {m.review_iterations} |\n"
            f"| Merge conflicts | {m.merge_conflicts} |\n"
            f"| LLM tokens (in/out/total) | {m.llm_input_tokens:,} / {m.llm_output_tokens:,} / {m.llm_total_tokens:,} |\n"
            f"| Public validate policy | {m.public_validate_policy or 'n/a'} |\n"
        )

        # 3. Plan
        if result.plan.text:
            sections.append(f"## Plan\n\n{result.plan.text}\n")

        if result.plan.subtasks:
            rows = "| ID | Role | Title |\n|---|---|---|\n"
            for st in result.plan.subtasks:
                rows += f"| {st.id} | {st.role} | {st.title} |\n"
            sections.append(f"### Subtasks\n\n{rows}")

        # 4. Review Story
        if result.review_rounds:
            parts = ["## Review Story\n"]
            for rr in result.review_rounds:
                outcome = "ACCEPTED" if rr.accepted else ("REWORK" if rr.request_rework else "REJECTED")
                parts.append(f"### Round {rr.round_index} — {outcome}\n")
                if rr.review_summary:
                    parts.append(f"- **Review:** {rr.review_summary}")
                if rr.review_notes:
                    parts.append(f"  - Notes: {rr.review_notes}")
                if rr.verify_summary:
                    parts.append(f"- **Verify:** {rr.verify_summary}")
                if rr.verify_notes:
                    parts.append(f"  - Notes: {rr.verify_notes}")
                if rr.merged_commits_this_round:
                    for role, commits in rr.merged_commits_this_round.items():
                        parts.append(f"- Merged ({role}): {', '.join(commits[:5])}")
                parts.append(f"- Public validation: {rr.public_validation_state} (policy={rr.public_validation_policy})")
                parts.append(f"- Merge OK: {rr.merge_ok}\n")
            sections.append("\n".join(parts))

        # 5. Reflection Evolution
        if result.reflections:
            parts = ["## Reflection Evolution\n"]
            for ref in result.reflections:
                parts.append(f"### Round {ref.round_index}\n")
                if ref.directive:
                    parts.append(f"- **directive:** {ref.directive}")
                if ref.task_understanding:
                    parts.append(f"- **task_understanding:** {ref.task_understanding}")
                if ref.failure_patterns:
                    parts.append(f"- **failure_patterns:** {ref.failure_patterns}")
                if ref.workflow_insights:
                    parts.append(f"- **workflow_insights:** {ref.workflow_insights}")
                if ref.superseded:
                    parts.append(f"- **superseded:** {', '.join(ref.superseded)}")
                parts.append("")
            sections.append("\n".join(parts))

        # 6. Hodoscope
        if result.hodoscope:
            h = result.hodoscope
            rows = (
                "## Hodoscope\n\n"
                "| Field | Value |\n|---|---|\n"
                f"| Trajectories | {h.trajectory_count} |\n"
                f"| Action summaries | {h.summary_count} |\n"
                f"| Embeddings | {'yes' if h.has_embeddings else 'no'} |\n"
            )
            if h.analysis_path:
                rows += f"| Analysis | `{h.analysis_path}` |\n"
            sections.append(rows)

        # 7. Final Diff
        if result.diff.final_patch:
            patch_lines = result.diff.final_patch.splitlines()
            truncated = len(patch_lines) > self.diff_lines
            display = "\n".join(patch_lines[: self.diff_lines])
            if truncated:
                display += f"\n... ({len(patch_lines) - self.diff_lines} more lines)"
            sections.append(f"## Final Diff\n\n```diff\n{display}\n```\n")

        if result.diff.per_role_changed_files:
            parts = ["### Changed Files by Role\n"]
            for role, files in result.diff.per_role_changed_files.items():
                parts.append(f"**{role}:** {', '.join(files)}")
            sections.append("\n".join(parts) + "\n")

        # 8. Trace
        if result.trace_url:
            sections.append(f"## Trace\n\n{result.trace_url}\n")

        # 9. Issues
        if result.open_questions or result.errors:
            parts = ["## Issues\n"]
            if result.open_questions:
                parts.append(f"### Open Questions\n\n{result.open_questions}")
            if result.errors:
                parts.append("### Parse Errors\n")
                for e in result.errors:
                    parts.append(f"- {e}")
            sections.append("\n".join(parts) + "\n")

        report = "\n".join(sections)
        out_path = run_dir / "_inspect_report.md"
        out_path.write_text(report, encoding="utf-8")
        return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_inspect(run_dir: str | Path, output_format: str = "rich", diff_lines: int = 60) -> None:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"Error: {run_dir / 'manifest.json'} not found")

    inspector = RunInspector(run_dir)
    result = inspector.inspect()

    if output_format == "markdown":
        renderer = MarkdownRenderer(diff_lines=diff_lines)
        out_path = renderer.render(result, run_dir)
        try:
            from rich.console import Console
            Console().print(f"[green]report written:[/green] {out_path}")
        except ModuleNotFoundError:
            print(f"report written: {out_path}")
    else:
        renderer = RichRenderer(diff_lines=diff_lines)
        renderer.render(result)
