"""
loopbench.interfaces

Core protocol boundaries across runtime components.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from .schema import Budget, TaskPack, ToolCall, ToolResult


class Sandbox(Protocol):
    """
    Agent-facing execution environment.
    Hidden evaluation assets must not be mounted here.
    """

    def name(self) -> str:
        ...

    def exec(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout_sec: int = 600,
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        ...

    def read_file(self, path: str) -> str:
        ...

    def write_file(self, path: str, content: str) -> None:
        ...

    def apply_patch(self, patch_text: str) -> None:
        ...


class MultiAgentHarness(Protocol):
    """
    Owns role scheduling, review loop, and merge strategy.
    """

    def run(self, task: TaskPack, budget: Budget, tools: "ToolRouter") -> Dict[str, Any]:
        ...


class ToolRouter(Protocol):
    """
    Stable tool contract exposed to roles/harness.
    """

    def call(self, call: ToolCall) -> ToolResult:
        ...

    def remaining_budget(self) -> Budget:
        ...
