"""
loopbench.knowledge_tool_context

Deploy knowledge_tool.py into the planner worktree and return command
templates (mirrors review_diff_context.py pattern).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def build_knowledge_tool_context(
    *,
    role_path: Path,
    knowledge_dir: Path,
) -> Dict[str, Any]:
    """Deploy knowledge_tool.py and return command templates."""
    tool_root = role_path / ".loopbench" / "artifacts" / "knowledge"
    tool_root.mkdir(parents=True, exist_ok=True)

    script_path = tool_root / "knowledge_tool.py"
    if not script_path.exists():
        template_path = Path(__file__).with_name("knowledge_tool.py")
        script_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")

    knowledge_dir_str = str(knowledge_dir)
    command_prefix = f"python .loopbench/artifacts/knowledge/knowledge_tool.py --knowledge-dir {knowledge_dir_str}"

    return {
        "command_prefix": command_prefix,
        "knowledge_dir": knowledge_dir_str,
        "commands": {
            "summary": f"{command_prefix} summary",
            "show": f"{command_prefix} show --surface <task_understanding|failure_patterns|workflow_insights>",
        },
    }
