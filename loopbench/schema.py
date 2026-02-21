"""
loopbench.schema

Typed schemas for task packs, runs, and tool calls.
Schemas are explicit so benchmark behavior is auditable and replayable.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TaskKind = Literal[
    "repo_patch",
    "repo_gen",
    "devops_cycle",
    "abcbench",
    "devopsgym",
    "repogenesis",
]


class Budget(BaseModel):
    wall_clock_sec: int = 3600
    tool_calls: int = 800
    env_cycles: int = 50
    log_queries: int = 200
    metric_queries: int = 200
    http_requests: int = 300


class WorkspaceSpec(BaseModel):
    base_repo: str = "repo.bundle"
    entrypoint: str = "README.task.md"


class SubstrateSpec(BaseModel):
    kind: Literal["compose", "kind", "none"] = "none"
    worktree_isolation: bool = True
    up_cmd: Optional[str] = None
    down_cmd: Optional[str] = None
    public_validate_cmd: Optional[str] = None


class FeedbackSurfaces(BaseModel):
    logs: bool = True
    metrics: bool = True
    db: bool = False
    browser: bool = False


class JudgeSpec(BaseModel):
    hidden_validate_cmd: str = "bash hidden/validate.sh"
    timeout_sec: int = 1800


class TaskPack(BaseModel):
    """
    Normalized on-disk task definition.
    """

    task_id: str
    kind: TaskKind
    difficulty: Optional[str] = None
    language: Optional[str] = None

    workspace: WorkspaceSpec = Field(default_factory=WorkspaceSpec)
    substrate: SubstrateSpec = Field(default_factory=SubstrateSpec)
    feedback_surfaces: FeedbackSurfaces = Field(default_factory=FeedbackSurfaces)
    judge: JudgeSpec = Field(default_factory=JudgeSpec)
    budget: Budget = Field(default_factory=Budget)

    # Materialized by loader.
    root_dir: str
    task_yaml: str = "task.yaml"
    public_dir: str = "public"
    hidden_dir: str = "hidden"


class ToolCall(BaseModel):
    ts_ms: int
    role: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ts_ms: int
    ok: bool
    tool: str
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    task_id: str
    attempt: int = 1
    roles: List[str]
    sandbox_backend: str
    substrate: str
    started_at: str
    finished_at: Optional[str] = None
    base_commit: Optional[str] = None
    role_heads: Dict[str, str] = Field(default_factory=dict)
    inputs_dir: Optional[str] = None
    repo_state_dir: Optional[str] = None

    final_patch_path: Optional[str] = None
    public_pass: Optional[bool] = None
    hidden_pass: Optional[bool] = None

    metrics: Dict[str, Any] = Field(default_factory=dict)
