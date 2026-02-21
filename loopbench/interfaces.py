"""
loopbench.interfaces

Core protocol boundaries across runtime components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .schema import Budget, TaskPack, ToolCall, ToolResult


@dataclass(frozen=True)
class Workspace:
    """
    Run workspace layout.
    """

    root: str
    base_repo_path: str
    worktrees_dir: str
    artifacts_dir: str


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


class Substrate(Protocol):
    """
    Runs the system-under-test (SUT) for a specific worktree.
    """

    def kind(self) -> str:
        ...

    def up(self) -> None:
        ...

    def down(self) -> None:
        ...

    def status(self) -> Dict[str, Any]:
        ...

    def run_public_validation(self) -> ToolResult:
        ...

    def logs_tail(self, service: str, lines: int = 200) -> str:
        ...

    def logs_query(self, query: str) -> str:
        ...

    def metrics_query(self, query: str) -> str:
        ...

    def http_request(self, method: str, url: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...


class Judge(Protocol):
    """
    Hidden validation in an isolated environment.
    """

    def run_hidden_validation(self, final_patch_path: str) -> ToolResult:
        ...


class AgentRole(Protocol):
    """
    Abstract agent role instance.
    """

    role_name: str

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
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
