#!/usr/bin/env python3
"""
Standalone knowledge-surface inspector for progressive disclosure.

Deployed into planner worktree alongside review_diff_tool.py.
Reads knowledge surfaces from a directory and prints them.

Commands:
  summary                              — Print the directive + surface index
  show --surface <name>                — Print a specific surface's full content
  show --surface task_understanding
  show --surface failure_patterns
  show --surface workflow_insights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SURFACE_NAMES = ("task_understanding", "failure_patterns", "workflow_insights")


def main() -> int:
    parser = argparse.ArgumentParser(description="LoopBench knowledge surface inspector")
    parser.add_argument("--knowledge-dir", required=True, help="Path to knowledge directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="Print directive + surface index")
    show_parser = sub.add_parser("show", help="Print a specific surface")
    show_parser.add_argument("--surface", required=True, choices=list(SURFACE_NAMES))

    args = parser.parse_args()
    knowledge_dir = Path(args.knowledge_dir)

    if not knowledge_dir.is_dir():
        print(f"knowledge directory not found: {knowledge_dir}", file=sys.stderr)
        return 2

    if args.cmd == "summary":
        directive = _read_surface(knowledge_dir, "directive")
        if directive:
            print(directive)
        else:
            print("(no directive yet)")
        print()
        print("Available surfaces:")
        for name in SURFACE_NAMES:
            content = _read_surface(knowledge_dir, name)
            chars = len(content)
            status = f"{chars} chars" if content else "empty"
            print(f"  {name}: {status}")
        return 0

    if args.cmd == "show":
        content = _read_surface(knowledge_dir, args.surface)
        if content:
            print(content)
        else:
            print(f"(surface '{args.surface}' is empty)")
        return 0

    return 1


def _read_surface(knowledge_dir: Path, name: str) -> str:
    path = knowledge_dir / f"{name}.md"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
