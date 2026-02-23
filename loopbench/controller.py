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
from typing import Any, Dict, List

from .agents import build_role_driver
from .config import AgentsConfig, RuntimeConfig, load_agents_config, load_runtime_config
from .docker_runtime import env_runner_docker_env
from .events import EventLogger
from .harness import DeterministicHarness
from .hidden_result import hidden_failure_reason
from .judge import build_judge
from .observability import ObservabilitySettings
from .reproducibility import snapshot_role_repo_state, write_run_inputs_snapshot
from .sandbox_factory import build_sandbox
from .schema import Budget, RunManifest, TaskPack, ToolCall, ToolResult
from .shell import shell_quote
from .task_loader import load_task_pack
from .time_utils import now_ms
from .tracing import build_trace_session, write_trace_snapshot
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
    run_paths = workspace_mgr.provision(run_id=run_id, task=task, roles=roles)
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

    obs_settings = _build_observability_settings(runtime_cfg)
    trace_session = build_trace_session(
        settings=obs_settings,
        run_id=run_id,
        task_id=task.task_id,
        task_kind=task.kind,
        roles=roles,
        run_dir=run_paths.run_dir,
    )

    event_logger = EventLogger(
        run_paths.run_dir / "events.jsonl",
        sinks=[trace_session] if trace_session.enabled else None,
    )
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

    trace_error: str | None = None
    trace_outputs: Dict[str, Any] | None = None
    sandboxes = {}
    try:
        substrates: Dict[str, LocalSubstrate] = {}
        substrate_docker_env = env_runner_docker_env(runtime_cfg.env_runner, runtime_cfg.judge)
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
                docker_env=substrate_docker_env,
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
        workflow_summary_path = harness_result.get("workflow_summary_path")
        public_pass = bool(harness_result.get("public_pass"))

        judge = build_judge(task=task, run_dir=run_paths.run_dir, judge_cfg=runtime_cfg.judge)
        hidden_result = judge.run_hidden_validation(final_patch_path=final_patch_path)
        hidden_summary = _summarize_hidden_result(
            hidden_result=hidden_result,
            run_dir=run_paths.run_dir,
        )

        planner_role_name = "planner_reviewer" if "planner_reviewer" in roles else (roles[0] if roles else "")
        usage_metrics, planner_mutation_metrics = _role_output_metrics(
            role_outputs=harness_result.get("role_outputs"),
            planner_role=planner_role_name,
        )
        workflow_summary_str = str(workflow_summary_path) if workflow_summary_path else None

        metrics_payload = {
            **(harness_result.get("metrics") or {}),
            **usage_metrics,
            **planner_mutation_metrics,
            "remaining_budget": router.remaining_budget().model_dump(),
            "public_validate_policy": task.substrate.public_validate_policy,
            "hidden_infra_error": hidden_summary["hidden_infra_error"],
            "hidden_result_log": hidden_summary["hidden_result_log"],
            "trace_backend": trace_session.backend,
            "trace_enabled": bool(trace_session.enabled),
        }
        if hidden_summary["hidden_failure_reason"]:
            metrics_payload["hidden_failure_reason"] = hidden_summary["hidden_failure_reason"]
        if workflow_summary_str:
            metrics_payload["workflow_summary_path"] = workflow_summary_str

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
            hidden_pass=bool(hidden_summary["hidden_pass"]),
            metrics=metrics_payload,
        )

        manifest_path = run_paths.run_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        run_finished_payload, trace_outputs = _run_output_payloads(
            run_id=run_id,
            task_id=task.task_id,
            manifest_path=manifest_path,
            public_pass=public_pass,
            hidden_summary=hidden_summary,
            usage_metrics=usage_metrics,
            planner_mutation_metrics=planner_mutation_metrics,
            run_dir=run_paths.run_dir,
            final_patch_path=Path(final_patch_path),
            workflow_summary_path=workflow_summary_path,
        )
        event_logger.log("run_finished", run_finished_payload)
        return manifest
    except Exception as exc:  # noqa: BLE001
        trace_error = str(exc)
        raise
    finally:
        try:
            trace_session.finish(outputs=trace_outputs, error=trace_error)
        except Exception:  # noqa: BLE001
            pass
        try:
            write_trace_snapshot(run_dir=run_paths.run_dir, snapshot=trace_session.snapshot())
        except Exception:  # noqa: BLE001
            pass
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
    tool_calls = 0
    env_cycles = 0
    log_queries = 0
    metric_queries = 0
    role_phase_failures = 0

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
                tool_calls += 1
                if tool in {"env.up", "env.down"}:
                    env_cycles += 1
                elif tool == "env.logs_query":
                    log_queries += 1
                elif tool == "env.metrics_query":
                    metric_queries += 1
                continue
            if event_type == "role_phase":
                payload = event.get("payload") or {}
                if payload.get("ok") is False:
                    role_phase_failures += 1

    return {
        "event_counts": counts,
        "tool_call_counts": tool_counts,
        "tool_calls": tool_calls,
        "env_cycles_used": env_cycles,
        "log_queries_used": log_queries,
        "metric_queries_used": metric_queries,
        "role_phase_failures": role_phase_failures,
        "events_path": str(path),
    }


def _role_output_metrics(*, role_outputs: Any, planner_role: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(role_outputs, dict):
        empty_planner = {"planner_mutation_count": 0, "planner_mutations_by_phase": {}} if planner_role else {}
        return {}, empty_planner

    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    per_role: Dict[str, Dict[str, int]] = {}
    per_phase: Dict[str, Dict[str, int]] = {}
    usage_entries = 0

    planner_by_phase: Dict[str, List[str]] = {}
    planner_mutation_count = 0

    for key, output in role_outputs.items():
        if not isinstance(output, dict):
            continue
        key_parts = str(key).split(":")
        role = str(output.get("role") or (key_parts[0] if key_parts else "unknown"))
        phase = str(output.get("phase") or (key_parts[1] if len(key_parts) >= 2 else "unknown"))

        usage = output.get("openrouter_usage")
        if isinstance(usage, dict):
            input_tokens = _as_nonnegative_int(usage.get("input_tokens"))
            output_tokens = _as_nonnegative_int(usage.get("output_tokens"))
            total_tokens = _as_nonnegative_int(usage.get("total_tokens"))
            if input_tokens is not None and output_tokens is not None and total_tokens is not None:
                for bucket in (
                    totals,
                    per_role.setdefault(role, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
                    per_phase.setdefault(phase, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
                ):
                    bucket["input_tokens"] += input_tokens
                    bucket["output_tokens"] += output_tokens
                    bucket["total_tokens"] += total_tokens
                usage_entries += 1

        if role != planner_role:
            continue
        applied_paths = output.get("applied_paths")
        if not isinstance(applied_paths, list):
            continue
        clean_paths = [str(path) for path in applied_paths if isinstance(path, str)]
        if not clean_paths:
            continue
        planner_by_phase.setdefault(phase, []).extend(clean_paths)
        planner_mutation_count += len(clean_paths)

    usage_metrics = (
        {
            "llm_role_phase_count": usage_entries,
            "llm_input_tokens": totals["input_tokens"],
            "llm_output_tokens": totals["output_tokens"],
            "llm_total_tokens": totals["total_tokens"],
            "llm_tokens_by_role": per_role,
            "llm_tokens_by_phase": per_phase,
        }
        if usage_entries > 0
        else {}
    )
    planner_metrics = (
        {"planner_mutation_count": planner_mutation_count, "planner_mutations_by_phase": planner_by_phase}
        if planner_role
        else {}
    )
    return usage_metrics, planner_metrics

def _as_nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:  # noqa: BLE001
        return None
    if parsed < 0:
        return None
    return parsed

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

def _summarize_hidden_result(*, hidden_result: ToolResult, run_dir: Path) -> Dict[str, Any]:
    hidden_infra_error = bool(hidden_result.data.get("infra_error"))
    hidden_failure_reason_text: str | None = None
    if not hidden_result.ok and not hidden_infra_error:
        hidden_failure_reason_text = hidden_failure_reason(run_dir)
    return {
        "hidden_pass": bool(hidden_result.ok),
        "hidden_exit_code": hidden_result.exit_code,
        "hidden_infra_error": hidden_infra_error,
        "hidden_failure_reason": hidden_failure_reason_text,
        "hidden_result_log": str(hidden_result.data.get("result_log") or ""),
    }

def _run_output_payloads(
    *,
    run_id: str,
    task_id: str,
    manifest_path: Path,
    public_pass: bool,
    hidden_summary: Dict[str, Any],
    usage_metrics: Dict[str, Any],
    planner_mutation_metrics: Dict[str, Any],
    run_dir: Path,
    final_patch_path: Path,
    workflow_summary_path: str | Path | None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    workflow_summary = str(workflow_summary_path) if workflow_summary_path else None
    common = {
        k: hidden_summary[k]
        for k in ("hidden_pass", "hidden_infra_error", "hidden_failure_reason", "hidden_result_log")
    }
    common["public_pass"] = public_pass
    event_payload: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        **common,
        "hidden_exit_code": hidden_summary["hidden_exit_code"],
        "llm_total_tokens": usage_metrics.get("llm_total_tokens"),
        "workflow_summary_path": workflow_summary,
        "planner_mutation_count": planner_mutation_metrics.get("planner_mutation_count"),
    }
    trace_payload: Dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_id,
        **common,
        "llm_usage": usage_metrics,
        "planner_mutations": planner_mutation_metrics,
        "run_dir": str(run_dir),
        "events_path": str(run_dir / "events.jsonl"),
        "manifest_path": str(manifest_path),
        "final_patch_path": str(final_patch_path),
        "inputs_dir": str(run_dir / "inputs"),
        "repo_state_dir": str(run_dir / "repo_state"),
        "coordination_db_path": str(run_dir / "coordination" / "coordination.db"),
        "plans_dir": str(run_dir / "plans"),
        "role_runtime_dir": str(run_dir / "role_runtime"),
        "role_summaries_dir": str(run_dir / "role_summaries"),
        "public_validate_dir": str(run_dir / "public_validate"),
        "workflow_summary_path": workflow_summary,
    }
    return event_payload, trace_payload

def _close_sandboxes(sandboxes: Dict[str, Any]) -> None:
    for sandbox in sandboxes.values():
        close_fn = getattr(sandbox, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass
