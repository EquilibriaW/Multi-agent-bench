"""
loopbench.preflight

Infra readiness checks for reproducible experiments.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict

from .config import RuntimeConfig
from .docker_runtime import docker_info, judge_docker_env


def run_preflight(runtime_cfg: RuntimeConfig) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "python": _binary_exists("python"),
        "git": _binary_exists("git"),
        "docker": _binary_exists("docker"),
    }
    warnings: list[str] = []

    sandbox_kind = runtime_cfg.sandbox_backend.kind
    checks["sandbox_backend_kind"] = sandbox_kind

    if sandbox_kind == "e2b_firecracker":
        env_name = runtime_cfg.sandbox_backend.e2b.api_key_env
        api_key = os.environ.get(env_name)
        checks["e2b_api_key_env"] = env_name
        checks["e2b_api_key_present"] = bool(api_key)
        checks["e2b_python_sdk"] = _python_module_exists("e2b_code_interpreter")
    else:
        checks["e2b_api_key_present"] = None
        checks["e2b_python_sdk"] = _python_module_exists("e2b_code_interpreter")

    obs = runtime_cfg.env_runner.observability
    checks["observability_logs_backend"] = obs.logs
    checks["observability_metrics_backend"] = obs.metrics
    if obs.logs.lower() == "loki":
        checks["observability_logs_endpoint_configured"] = bool(obs.logs_endpoint)
        if not obs.logs_endpoint:
            warnings.append("observability.logs=loki but logs_endpoint is empty; env.logs_query will use artifact fallback.")
    else:
        checks["observability_logs_endpoint_configured"] = None

    if obs.metrics.lower() == "prometheus":
        checks["observability_metrics_endpoint_configured"] = bool(obs.metrics_endpoint)
        if not obs.metrics_endpoint:
            warnings.append(
                "observability.metrics=prometheus but metrics_endpoint is empty; env.metrics_query will be unavailable."
            )
    else:
        checks["observability_metrics_endpoint_configured"] = None

    ok = all(
        value is True
        for key, value in checks.items()
        if key in {"python", "git"}
    )

    if runtime_cfg.env_runner.substrate_default == "compose":
        ok = ok and checks["docker"] is True

    if runtime_cfg.judge.kind == "docker_container":
        checks["judge_requires_docker"] = True
        judge_env = judge_docker_env(runtime_cfg.judge)
        checks["judge_docker_host"] = judge_env.get("DOCKER_HOST")
        checks["judge_docker_daemon_ready"] = _docker_daemon_ready(judge_env)
        ok = ok and checks["docker"] is True and checks["judge_docker_daemon_ready"] is True
    else:
        checks["judge_requires_docker"] = False
        checks["judge_docker_host"] = None
        checks["judge_docker_daemon_ready"] = None

    if sandbox_kind == "e2b_firecracker":
        ok = ok and checks["e2b_api_key_present"] is True and checks["e2b_python_sdk"] is True

    return {"ok": ok, "checks": checks, "warnings": warnings}


def _binary_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _python_module_exists(name: str) -> bool:
    import importlib

    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _docker_daemon_ready(env: Dict[str, str] | None = None) -> bool:
    if shutil.which("docker") is None:
        return False
    result = docker_info(env=env, timeout_sec=20)
    return result.ok
