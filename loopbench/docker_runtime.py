"""
loopbench.docker_runtime

Shared Docker runtime helpers for judge/env checks.
"""
from __future__ import annotations

from typing import Dict

from .config import EnvRunnerConfig, JudgeRuntimeConfig
from .shell import CommandResult, run_command


def docker_env_from_host(
    docker_host: str | None,
    *,
    buildx_config: str | None = None,
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if docker_host:
        env["DOCKER_HOST"] = docker_host
    if buildx_config:
        env["BUILDX_CONFIG"] = buildx_config
    return env


def judge_docker_env(judge_cfg: JudgeRuntimeConfig) -> Dict[str, str]:
    return docker_env_from_host(
        judge_cfg.docker_host,
        buildx_config=judge_cfg.buildx_config,
    )


def env_runner_docker_env(
    env_runner_cfg: EnvRunnerConfig,
    judge_cfg: JudgeRuntimeConfig | None = None,
) -> Dict[str, str]:
    """
    Build Docker env for substrate/env-runner commands.

    - `DOCKER_HOST` is sourced only from `env_runner_cfg`.
    - `BUILDX_CONFIG` may fall back to `judge_cfg` for compatibility when
      env-runner does not specify one.
    """
    # Do not inherit judge DOCKER_HOST for env-runner/substrate operations.
    # Sharing a remote daemon must be explicitly configured on env_runner.
    docker_host = env_runner_cfg.docker_host
    buildx_config = env_runner_cfg.buildx_config
    if not buildx_config and judge_cfg is not None:
        buildx_config = judge_cfg.buildx_config
    return docker_env_from_host(docker_host, buildx_config=buildx_config)


def docker_info(*, env: Dict[str, str] | None = None, timeout_sec: int = 20) -> CommandResult:
    return run_command(["docker", "info"], timeout_sec=timeout_sec, env=env)
