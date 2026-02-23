"""
loopbench.config

Typed loaders for runtime and agent configuration files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .io_utils import read_yaml_mapping
from .schema import Budget


class E2BBackendConfig(BaseModel):
    template: Optional[str] = None
    timeout_sec: int = 3600
    allow_internet_access: bool = False
    api_key_env: str = "E2B_API_KEY"


class SandboxBackendConfig(BaseModel):
    kind: Literal[
        "docker_container",
        "kata_container",
        "firecracker_microvm",
        "unikernel",
        "local_process",
        "e2b_firecracker",
    ] = "e2b_firecracker"
    image: Optional[str] = None
    network: Optional[str] = None
    e2b: E2BBackendConfig = Field(default_factory=E2BBackendConfig)


class ObservabilityConfig(BaseModel):
    logs: str = "none"
    metrics: str = "none"
    traces: str = "langsmith"
    logs_endpoint: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    traces_endpoint: Optional[str] = None


class EnvRunnerConfig(BaseModel):
    kind: Literal["docker_service", "host_process"] = "host_process"
    image: Optional[str] = None
    substrate_default: Literal["compose", "kind", "none"] = "none"
    docker_host: Optional[str] = None
    buildx_config: Optional[str] = None
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)


class JudgeRuntimeConfig(BaseModel):
    kind: Literal["docker_container", "local_process", "e2b_firecracker"] = "local_process"
    image: Optional[str] = None
    network: Optional[str] = None
    docker_host: Optional[str] = None
    buildx_config: Optional[str] = None
    e2b_template: Optional[str] = None


class StorageConfig(BaseModel):
    workspace_root: str = "workspaces"
    runs_root: str = "runs"
    cache_root: str = "caches"


class RuntimeConfig(BaseModel):
    sandbox_backend: SandboxBackendConfig = Field(default_factory=SandboxBackendConfig)
    env_runner: EnvRunnerConfig = Field(default_factory=EnvRunnerConfig)
    judge: JudgeRuntimeConfig = Field(default_factory=JudgeRuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


class RoleConfig(BaseModel):
    name: str
    model: Optional[str] = None
    max_tokens: int = 200000
    driver: Literal["noop", "shell"] = "shell"
    command: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_driver_command(self) -> "RoleConfig":
        if self.driver == "shell" and not (self.command and self.command.strip()):
            raise ValueError(f"role '{self.name}' uses driver=shell but command is empty")
        return self


class PhaseConfig(BaseModel):
    name: str
    active_roles: List[str]
    exit_condition: str


class SchedulingConfig(BaseModel):
    mode: Literal["phased", "round_robin", "event_driven"] = "phased"
    phases: List[PhaseConfig] = Field(default_factory=list)
    max_review_rounds: int = 3
    reflection_enabled: bool = True


class AgentsConfig(BaseModel):
    roles: List[RoleConfig]
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    budgets: Budget = Field(default_factory=Budget)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    cfg_path = Path(path)
    return RuntimeConfig.model_validate(read_yaml_mapping(cfg_path, label="runtime config"))


def load_agents_config(path: str | Path) -> AgentsConfig:
    cfg_path = Path(path)
    return AgentsConfig.model_validate(read_yaml_mapping(cfg_path, label="agents config"))
