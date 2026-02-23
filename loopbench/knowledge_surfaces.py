"""
loopbench.knowledge_surfaces

Persistent knowledge surfaces for LLM-driven reflection.

Surfaces are markdown files under ``runs/<run_id>/knowledge/`` that are
**overwritten** each round (not appended) to prevent context poisoning.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


SURFACE_NAMES = ("task_understanding", "failure_patterns", "workflow_insights")


class KnowledgeSurfaces:
    """Manage per-run knowledge surfaces written by the reflection LLM."""

    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Bootstrap seeding (no LLM call)
    # ------------------------------------------------------------------

    def seed_from_bootstrap(self, plan_md: str, subtasks: List[Dict[str, Any]]) -> None:
        """Initialize ``task_understanding.md`` from bootstrap output."""
        subtask_lines = []
        for st in subtasks:
            if not isinstance(st, dict):
                continue
            st_id = st.get("id", "?")
            role = st.get("role", "?")
            title = st.get("title", "")
            subtask_lines.append(f"- {st_id} ({role}): {title}")

        content = (
            "# Task Understanding\n\n"
            "## Plan\n\n"
            f"{plan_md.strip()}\n\n"
            "## Subtasks\n\n"
            + ("\n".join(subtask_lines) if subtask_lines else "(none)")
            + "\n"
        )
        self._write_surface("task_understanding", content)

    # ------------------------------------------------------------------
    # Reflection update (overwrites all surfaces)
    # ------------------------------------------------------------------

    def update_from_reflection(self, round_index: int, reflection_output: Dict[str, Any]) -> None:
        """Overwrite all surfaces from reflection LLM structured output."""
        directive_text = _safe_str(reflection_output.get("directive"), max_chars=600)
        if directive_text:
            header = f"# Reflection Directive (round {round_index})\n\n"
            self._write_surface("directive", header + directive_text + "\n")

        for name in SURFACE_NAMES:
            value = _safe_str(reflection_output.get(name))
            if value:
                self._write_surface(name, f"# {_title(name)} (round {round_index})\n\n{value}\n")

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def directive(self) -> str:
        """Return the current directive text.  Empty string if none yet."""
        return self._read_surface("directive")

    def surface(self, name: str) -> str:
        """Return a named surface's content."""
        if name not in SURFACE_NAMES and name != "directive":
            return ""
        return self._read_surface(name)

    def summary_index(self) -> str:
        """One-line-per-surface index with char counts (for progressive disclosure)."""
        lines = []
        for name in ("directive", *SURFACE_NAMES):
            content = self._read_surface(name)
            chars = len(content)
            status = f"{chars} chars" if content else "empty"
            lines.append(f"  {name}: {status}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_surface(self, name: str, content: str) -> None:
        path = self.knowledge_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")

    def _read_surface(self, name: str) -> str:
        path = self.knowledge_dir / f"{name}.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return ""


def _safe_str(value: Any, max_chars: int = 0) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def _title(surface_name: str) -> str:
    return surface_name.replace("_", " ").title()
