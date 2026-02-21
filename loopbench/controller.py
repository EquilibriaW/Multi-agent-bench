"""
loopbench.controller

Top-level orchestration for `run`, `judge`, `pack`, and `replay` CLI commands.
"""
from __future__ import annotations

import json
import os
import shlex
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .agents import build_role_driver
from .config import AgentsConfig, RuntimeConfig, load_agents_config, load_runtime_config
from .events import EventLogger
from .harness import DeterministicHarness
from .judge import build_judge
from .observability import ObservabilitySettings
from .reproducibility import snapshot_role_repo_state, write_run_inputs_snapshot
from .sandbox_factory import build_sandbox
from .schema import Budget, RunManifest, TaskPack, ToolCall, ToolResult
from .shell import shell_quote
from .task_loader import load_task_pack
from .time_utils import now_ms
from .tools import DefaultToolRouter
from .workspace import WorkspaceManager
from .substrate import LocalSubstrate


def run_benchmark(
    task_dir: str,
    run_id: str,
    runtime_config_path: str,
    agents_config_path: str,
    project_root: str | Path,
    allow_noop: bool = False,
) -> RunManifest:
    started_at = _now_iso()

    project_root_path = Path(project_root).resolve()
    runtime_cfg = load_runtime_config(runtime_config_path)
    agents_cfg = load_agents_config(agents_config_path)
    agents_cfg = _resolve_role_commands(
        agents_cfg=agents_cfg,
        project_root=project_root_path,
        agents_config_path=Path(agents_config_path).resolve(),
    )
    agents_cfg = _inject_runtime_role_env(
        agents_cfg=agents_cfg,
        sandbox_backend_kind=runtime_cfg.sandbox_backend.kind,
    )
    agents_cfg = _inject_provider_env(agents_cfg=agents_cfg)
    _validate_role_drivers(agents_cfg=agents_cfg, allow_noop=allow_noop)
    task = load_task_pack(task_dir)

    roles = [r.name for r in agents_cfg.roles]
    workspace_mgr = WorkspaceManager(
        project_root=project_root_path,
        workspace_root=runtime_cfg.storage.workspace_root,
        runs_root=runtime_cfg.storage.runs_root,
    )
    workspace, run_paths = workspace_mgr.provision(run_id=run_id, task=task, roles=roles)
    write_run_inputs_snapshot(
        run_dir=run_paths.run_dir,
        run_id=run_id,
        runtime_config_path=runtime_config_path,
        runtime_cfg=runtime_cfg,
        agents_config_path=agents_config_path,
        agents_cfg=agents_cfg,
        task=task,
        base_commit=run_paths.base_commit,
    )

    event_logger = EventLogger(run_paths.run_dir / "events.jsonl")
    event_logger.log(
        "run_started",
        {
            "run_id": run_id,
            "task_id": task.task_id,
            "task_kind": task.kind,
            "roles": roles,
            "drivers": {cfg.name: cfg.driver for cfg in agents_cfg.roles},
            "allow_noop": allow_noop,
            "started_at": started_at,
        },
    )

    sandboxes = {}
    try:
        substrates: Dict[str, LocalSubstrate] = {}
        obs_settings = _build_observability_settings(runtime_cfg)
        for role in roles:
            sandboxes[role] = build_sandbox(
                runtime_cfg=runtime_cfg,
                role=role,
                worktree_root=run_paths.role_paths[role],
            )
            substrates[role] = LocalSubstrate(
                role=role,
                worktree_path=run_paths.role_paths[role],
                spec=task.substrate,
                run_artifacts_dir=run_paths.run_dir,
                observability=obs_settings,
            )

        merged_budget = _merge_budget(task.budget, agents_cfg.budgets)
        router = DefaultToolRouter(
            run_id=run_id,
            sandboxes=sandboxes,
            substrates=substrates,
            budget=merged_budget,
            event_logger=event_logger,
        )

        role_drivers = {
            cfg.name: build_role_driver(cfg, sandbox=sandboxes[cfg.name])
            for cfg in agents_cfg.roles
        }

        harness = DeterministicHarness(
            run_id=run_id,
            run_dir=run_paths.run_dir,
            role_paths=run_paths.role_paths,
            role_drivers=role_drivers,
            base_commit=run_paths.base_commit,
            max_review_rounds=agents_cfg.scheduling.max_review_rounds,
            event_logger=event_logger,
        )

        # Optionally bring up planner substrate once before orchestration.
        planner_role = "planner_reviewer" if "planner_reviewer" in roles else roles[0]
        if task.substrate.up_cmd:
            _ = router.call(
                ToolCall(ts_ms=now_ms(), role=planner_role, tool="env.up", args={})
            )

        harness_result = harness.run(task=task, budget=merged_budget, tools=router)
        role_heads = snapshot_role_repo_state(
            run_dir=run_paths.run_dir,
            role_paths=run_paths.role_paths,
            base_commit=run_paths.base_commit,
        )

        if task.substrate.down_cmd:
            for role in roles:
                _ = router.call(ToolCall(ts_ms=now_ms(), role=role, tool="env.down", args={}))

        final_patch_path = harness_result["final_patch_path"]
        public_pass = bool(harness_result.get("public_pass"))

        judge = build_judge(task=task, run_dir=run_paths.run_dir, judge_cfg=runtime_cfg.judge)
        hidden_result = judge.run_hidden_validation(final_patch_path=final_patch_path)

        manifest = RunManifest(
            run_id=run_id,
            task_id=task.task_id,
            roles=roles,
            sandbox_backend=runtime_cfg.sandbox_backend.kind,
            substrate=task.substrate.kind,
            started_at=started_at,
            finished_at=_now_iso(),
            base_commit=run_paths.base_commit,
            role_heads=role_heads,
            inputs_dir=str(run_paths.run_dir / "inputs"),
            repo_state_dir=str(run_paths.run_dir / "repo_state"),
            final_patch_path=str(final_patch_path),
            public_pass=public_pass,
            hidden_pass=hidden_result.ok,
            metrics={
                **(harness_result.get("metrics") or {}),
                "remaining_budget": router.remaining_budget().model_dump(),
                "hidden_infra_error": bool(hidden_result.data.get("infra_error")),
            },
        )

        manifest_path = run_paths.run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        event_logger.log(
            "run_finished",
            {
                "manifest_path": str(manifest_path),
                "public_pass": public_pass,
                "hidden_pass": hidden_result.ok,
                "hidden_exit_code": hidden_result.exit_code,
                "hidden_infra_error": bool(hidden_result.data.get("infra_error")),
            },
        )

        return manifest
    finally:
        _close_sandboxes(sandboxes)


def run_judge_only(task_dir: str, run_dir: str, runtime_config_path: str) -> ToolResult:
    task = load_task_pack(task_dir)
    run_path = Path(run_dir).resolve()
    final_patch = run_path / "final.patch"
    if not final_patch.exists():
        raise FileNotFoundError(f"final patch not found: {final_patch}")

    runtime_cfg = load_runtime_config(runtime_config_path)
    judge = build_judge(task=task, run_dir=run_path, judge_cfg=runtime_cfg.judge)
    result = judge.run_hidden_validation(final_patch_path=str(final_patch))

    return result


def pack_task(task_dir: str, out_path: str) -> Path:
    task = load_task_pack(task_dir)
    src = Path(task.root_dir)
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname=src.name)

    return out


def replay_events(run_dir: str) -> Dict[str, Any]:
    path = Path(run_dir).resolve() / "events.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"events.jsonl not found at {path}")

    counts: Dict[str, int] = {}
    tool_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            event = json.loads(line)
            event_type = event.get("type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
            if event_type == "tool_call":
                payload = event.get("payload") or {}
                tool = payload.get("tool", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

    return {
        "event_counts": counts,
        "tool_call_counts": tool_counts,
        "events_path": str(path),
    }


def _merge_budget(task_budget: Budget, agent_budget: Budget) -> Budget:
    return Budget(
        wall_clock_sec=min(task_budget.wall_clock_sec, agent_budget.wall_clock_sec),
        tool_calls=min(task_budget.tool_calls, agent_budget.tool_calls),
        env_cycles=min(task_budget.env_cycles, agent_budget.env_cycles),
        log_queries=min(task_budget.log_queries, agent_budget.log_queries),
        metric_queries=min(task_budget.metric_queries, agent_budget.metric_queries),
        http_requests=min(task_budget.http_requests, agent_budget.http_requests),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_role_drivers(*, agents_cfg: AgentsConfig, allow_noop: bool) -> None:
    noop_roles = [cfg.name for cfg in agents_cfg.roles if cfg.driver == "noop"]
    if noop_roles and not allow_noop:
        names = ", ".join(noop_roles)
        raise ValueError(
            f"noop roles are disabled by default for benchmark runs. "
            f"Found: {names}. "
            f"Use real shell drivers, or pass --allow-noop for plumbing smoke runs."
        )


def _resolve_role_commands(
    *,
    agents_cfg: AgentsConfig,
    project_root: Path,
    agents_config_path: Path,
) -> AgentsConfig:
    config_dir = agents_config_path.parent
    resolved_roles = []

    for role in agents_cfg.roles:
        command = role.command
        if role.driver != "shell" or not command:
            resolved_roles.append(role)
            continue

        parts = shlex.split(command)
        if not parts:
            resolved_roles.append(role)
            continue

        first = Path(parts[0])
        if first.is_absolute():
            resolved_roles.append(role)
            continue

        candidate_cfg = (config_dir / first).resolve()
        candidate_root = (project_root / first).resolve()
        replacement: Path | None = None
        if candidate_cfg.exists():
            replacement = candidate_cfg
        elif candidate_root.exists():
            replacement = candidate_root

        if replacement is None:
            resolved_roles.append(role)
            continue

        parts[0] = str(replacement)
        resolved_command = shell_quote(parts)
        resolved_roles.append(role.model_copy(update={"command": resolved_command}))

    return agents_cfg.model_copy(update={"roles": resolved_roles})


def _build_observability_settings(runtime_cfg: RuntimeConfig) -> ObservabilitySettings:
    obs_cfg = runtime_cfg.env_runner.observability
    return ObservabilitySettings(
        logs=obs_cfg.logs,
        metrics=obs_cfg.metrics,
        traces=obs_cfg.traces,
        logs_endpoint=obs_cfg.logs_endpoint,
        metrics_endpoint=obs_cfg.metrics_endpoint,
        traces_endpoint=obs_cfg.traces_endpoint,
    )


def _inject_runtime_role_env(
    *,
    agents_cfg: AgentsConfig,
    sandbox_backend_kind: str,
) -> AgentsConfig:
    roles = []
    for role in agents_cfg.roles:
        env = dict(role.env)
        env.setdefault("LOOPBENCH_SANDBOX_BACKEND", sandbox_backend_kind)
        roles.append(role.model_copy(update={"env": env}))
    return agents_cfg.model_copy(update={"roles": roles})


def _inject_provider_env(*, agents_cfg: AgentsConfig) -> AgentsConfig:
    roles = []
    for role in agents_cfg.roles:
        env = dict(role.env)
        key_var = env.get("OPENROUTER_API_KEY_ENV") or env.get("OPEN_ROUTER_API_KEY_ENV")
        if isinstance(key_var, str) and key_var.strip():
            key_name = key_var.strip()
            if key_name not in env:
                key_value = os.environ.get(key_name)
                if key_value:
                    env[key_name] = key_value
        roles.append(role.model_copy(update={"env": env}))
    return agents_cfg.model_copy(update={"roles": roles})


def _close_sandboxes(sandboxes: Dict[str, Any]) -> None:
    for sandbox in sandboxes.values():
        close_fn = getattr(sandbox, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass
