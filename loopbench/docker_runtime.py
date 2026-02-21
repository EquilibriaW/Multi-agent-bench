"""
loopbench.docker_runtime

Shared Docker runtime helpers for judge/env checks.
"""
from __future__ import annotations

from typing import Dict

from .config import JudgeRuntimeConfig
from .shell import CommandResult, run_command


def judge_docker_env(judge_cfg: JudgeRuntimeConfig) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if judge_cfg.docker_host:
        env["DOCKER_HOST"] = judge_cfg.docker_host
    return env


def docker_info(*, env: Dict[str, str] | None = None, timeout_sec: int = 20) -> CommandResult:
    return run_command(["docker", "info"], timeout_sec=timeout_sec, env=env)

