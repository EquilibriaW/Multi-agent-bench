"""
loopbench.controller

Top-level orchestration for `run`, `judge`, `pack`, and `replay` CLI commands.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
import tarfile
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    wait as futures_wait,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
)
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
    shutdown_event: threading.Event | None = None,
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
    harness_cancel_event = threading.Event()

    # Thread-safe registry: _create_role_infra adds every sandbox here
    # immediately after creation, so the outer finally can close ALL of
    # them — even ones from futures we abandoned.
    all_created_sandboxes: Dict[str, Any] = {}
    _sandbox_registry_lock = threading.Lock()
    init_cancel_event = threading.Event()

    # Propagate external shutdown signal to both cancel events so that
    # SIGTERM interrupts sandbox init (init_cancel_event) as well as
    # in-flight harness work (harness_cancel_event).
    if shutdown_event is not None:
        def _watch_shutdown():
            shutdown_event.wait()
            init_cancel_event.set()
            harness_cancel_event.set()
        _shutdown_watcher = threading.Thread(target=_watch_shutdown, daemon=True)
        _shutdown_watcher.start()
    try:
        substrates: Dict[str, LocalSubstrate] = {}
        substrate_docker_env = env_runner_docker_env(runtime_cfg.env_runner, runtime_cfg.judge)
        # Daemon workers ensure stalled sandbox init threads cannot keep
        # the CLI process alive after timeout handling returns.
        init_pool = _DaemonThreadPoolExecutor(max_workers=len(roles))
        try:
            futs = {
                init_pool.submit(
                    _create_role_infra, role,
                    runtime_cfg=runtime_cfg,
                    worktree_root=run_paths.role_paths[role],
                    spec=task.substrate,
                    run_artifacts_dir=run_paths.run_dir,
                    obs=obs_settings,
                    docker_env=substrate_docker_env,
                    sandbox_registry=(all_created_sandboxes, _sandbox_registry_lock),
                    cancel_event=init_cancel_event,
                ): role
                for role in roles
            }
            done, not_done = futures_wait(futs, timeout=_SANDBOX_CREATE_TIMEOUT_SEC, return_when=FIRST_EXCEPTION)

            # Collect ALL successful results first so their sandboxes
            # enter the dict (and thus get cleaned up by the outer finally).
            first_exc = None
            closed_stalled_sandbox_ids: set[int] = set()

            def _collect_role_infra_results(completed) -> None:
                nonlocal first_exc
                for fut in completed:
                    try:
                        role, sandbox, substrate = fut.result()
                        sandboxes[role] = sandbox
                        substrates[role] = substrate
                    except _SandboxInitCancelled:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        if first_exc is None:
                            first_exc = exc

            def _close_stalled_role_sandboxes() -> None:
                with _sandbox_registry_lock:
                    leaked = {
                        role: sandbox
                        for role, sandbox in all_created_sandboxes.items()
                        if role not in sandboxes and id(sandbox) not in closed_stalled_sandbox_ids
                    }
                if leaked:
                    _close_sandboxes(leaked)
                    closed_stalled_sandbox_ids.update(id(sandbox) for sandbox in leaked.values())

            _collect_role_infra_results(done)

            if not_done:
                init_cancel_event.set()
                for fut in not_done:
                    fut.cancel()
                # Stop queued futures from starting; cancel_futures=True
                # prevents any not-yet-started workers from launching.
                init_pool.shutdown(wait=False, cancel_futures=True)

                # Allow already-running futures a short drain window so we
                # can close sandboxes created after timeout and avoid leaks.
                drain_deadline = time.monotonic() + _SANDBOX_DRAIN_TIMEOUT_SEC
                pending = set(not_done)
                while pending and time.monotonic() < drain_deadline:
                    remaining = drain_deadline - time.monotonic()
                    wait_timeout = min(0.25, max(0.01, remaining))
                    newly_done, pending = futures_wait(
                        pending,
                        timeout=wait_timeout,
                        return_when=FIRST_COMPLETED,
                    )
                    _collect_role_infra_results(newly_done)
                    _close_stalled_role_sandboxes()
                _close_stalled_role_sandboxes()

                # If a role raised (FIRST_EXCEPTION), propagate that —
                # don't misclassify as a timeout.
                if first_exc is not None:
                    raise first_exc
                raise RuntimeError(
                    f"sandbox creation timed out after {_SANDBOX_CREATE_TIMEOUT_SEC}s "
                    f"({len(not_done)} of {len(futs)} roles stalled)"
                )

            if first_exc is not None:
                raise first_exc
        finally:
            init_pool.shutdown(wait=False)

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
            reflection_enabled=agents_cfg.scheduling.reflection_enabled,
        )

        # Optionally bring up planner substrate once before orchestration.
        planner_role = "planner_reviewer" if "planner_reviewer" in roles else roles[0]
        if task.substrate.up_cmd:
            _ = router.call(
                ToolCall(ts_ms=now_ms(), role=planner_role, tool="env.up", args={})
            )

        # Cancellation token: set on wall-clock timeout so the harness
        # thread stops at its next router.call() instead of running
        # indefinitely in the background.
        _install_cancel_guard(router, harness_cancel_event)

        deadline_pool = _DaemonThreadPoolExecutor(max_workers=1)
        deadline_future = deadline_pool.submit(harness.run, task=task, budget=merged_budget, tools=router)
        try:
            harness_result = deadline_future.result(timeout=merged_budget.wall_clock_sec)
        except FuturesTimeoutError:
            harness_cancel_event.set()
            _cancel_substrates(substrates)
            _close_sandboxes(sandboxes)
            try:
                deadline_future.result(timeout=_HARNESS_CANCEL_TIMEOUT_SEC)
            except FuturesTimeoutError:
                print(
                    "[loopbench] warning: harness did not stop within "
                    f"{_HARNESS_CANCEL_TIMEOUT_SEC}s after timeout cancellation",
                    file=sys.stderr,
                )
            except Exception:  # noqa: BLE001
                pass
            deadline_pool.shutdown(wait=False)
            raise RuntimeError(
                f"wall clock timeout: budget exhausted ({merged_budget.wall_clock_sec}s)"
            )
        finally:
            deadline_pool.shutdown(wait=False)
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

        # Free agent sandboxes before creating the judge sandbox so we
        # don't spike to 4 sandboxes per rollout simultaneously.
        _teardown_substrates(substrates)
        substrates.clear()
        _close_sandboxes(sandboxes)
        sandboxes.clear()
        with _sandbox_registry_lock:
            all_created_sandboxes.clear()

        judge = build_judge(task=task, run_dir=run_paths.run_dir, judge_cfg=runtime_cfg.judge, runtime_cfg=runtime_cfg)
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

        try:
            from .hodoscope_export import auto_analyze
            auto_analyze(run_paths.run_dir)
        except Exception:  # noqa: BLE001
            pass

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
        # Tear down Docker containers/networks before closing sandboxes
        # so that cleanup runs even on timeout, SIGTERM, or crash paths.
        _teardown_substrates(substrates)
        # Close ALL sandboxes ever created (including ones from abandoned
        # futures), not just the ones that made it into the `sandboxes` dict.
        with _sandbox_registry_lock:
            all_to_close = dict(all_created_sandboxes)
        # Merge in any that are only in `sandboxes` (shouldn't happen,
        # but defensive).
        for k, v in sandboxes.items():
            all_to_close.setdefault(k, v)
        _close_sandboxes(all_to_close)


def run_judge_only(task_dir: str, run_dir: str, runtime_config_path: str) -> ToolResult:
    task = load_task_pack(task_dir)
    run_path = Path(run_dir).resolve()
    final_patch = run_path / "final.patch"
    if not final_patch.exists():
        raise FileNotFoundError(f"final patch not found: {final_patch}")

    runtime_cfg = load_runtime_config(runtime_config_path)
    judge = build_judge(task=task, run_dir=run_path, judge_cfg=runtime_cfg.judge, runtime_cfg=runtime_cfg)
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
        if not isinstance(usage, dict):
            usage = output.get("usage")
        if isinstance(usage, dict):
            input_tokens = _as_nonnegative_int(usage.get("input_tokens"))
            if input_tokens is None:
                input_tokens = _as_nonnegative_int(usage.get("prompt_tokens"))

            output_tokens = _as_nonnegative_int(usage.get("output_tokens"))
            if output_tokens is None:
                output_tokens = _as_nonnegative_int(usage.get("completion_tokens"))

            total_tokens = _as_nonnegative_int(usage.get("total_tokens"))
            if total_tokens is None:
                total_tokens = (input_tokens or 0) + (output_tokens or 0)

            usage_totals = {
                "input_tokens": input_tokens or 0,
                "output_tokens": output_tokens or 0,
                "total_tokens": total_tokens,
            }
            for bucket in (
                totals,
                per_role.setdefault(role, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
                per_phase.setdefault(phase, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
            ):
                bucket["input_tokens"] += usage_totals["input_tokens"]
                bucket["output_tokens"] += usage_totals["output_tokens"]
                bucket["total_tokens"] += usage_totals["total_tokens"]
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
            key_candidates = []
            for candidate in (key_name, "OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
                if candidate not in key_candidates:
                    key_candidates.append(candidate)
            if not env.get(key_name):
                key_value = _first_non_empty_env(
                    env,
                    key_candidates,
                )
                if not key_value:
                    raise RuntimeError(
                        f"Required OpenRouter key env var is not set for role '{role.name}' "
                        f"(referenced by OPENROUTER_API_KEY_ENV={key_name}). "
                        f"Set one of: {', '.join(key_candidates)}."
                    )
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

_SANDBOX_CREATE_TIMEOUT_SEC = 300
_SANDBOX_DRAIN_TIMEOUT_SEC = 30
_SANDBOX_CREATE_MAX_ATTEMPTS = 5
_HARNESS_CANCEL_TIMEOUT_SEC = 30


class _SandboxInitCancelled(RuntimeError):
    """Sandbox init was cancelled after a global timeout fired."""


def _first_non_empty_env(local_env: Dict[str, str], names: List[str]) -> str | None:
    for name in names:
        value = local_env.get(name)
        if isinstance(value, str) and value.strip():
            return value
        host_value = os.environ.get(name)
        if isinstance(host_value, str) and host_value.strip():
            return host_value
    return None


def _install_cancel_guard(router, cancel_event: threading.Event) -> None:
    """Wrap *router.call* so it raises immediately when *cancel_event* is set.

    Every harness action flows through ``router.call()``.  After the
    wall-clock deadline fires, setting the event causes the background
    harness thread to abort at its next tool call instead of continuing
    to run commands against closed (or local-process) sandboxes.
    """
    original_call = router.call

    def _guarded_call(tool_call):
        if cancel_event.is_set():
            raise RuntimeError("wall clock timeout: run cancelled")
        return original_call(tool_call)

    router.call = _guarded_call


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are daemonic.

    Daemon threads are killed when the main thread exits, so a timed-out
    harness.run() cannot keep the process alive after the caller raises.
    """

    def _adjust_thread_count(self):
        import weakref
        from concurrent.futures.thread import _worker, _threads_queues

        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            t = threading.Thread(
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _threads_queues[t] = self._work_queue


def _create_role_infra(
    role,
    *,
    runtime_cfg,
    worktree_root,
    spec,
    run_artifacts_dir,
    obs,
    docker_env,
    sandbox_registry=None,
    cancel_event: threading.Event | None = None,
):
    """Create sandbox + substrate for a single role, with retry.

    If *sandbox_registry* is ``(dict, lock)`` the sandbox is registered
    immediately after creation so the caller can close it even if this
    future is later abandoned.
    """
    last_exc = None
    for attempt in range(1, _SANDBOX_CREATE_MAX_ATTEMPTS + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise _SandboxInitCancelled(f"sandbox init cancelled for role '{role}'")
        sandbox = None
        try:
            sandbox = build_sandbox(runtime_cfg=runtime_cfg, role=role, worktree_root=worktree_root)
            if cancel_event is not None and cancel_event.is_set():
                raise _SandboxInitCancelled(f"sandbox init cancelled for role '{role}'")
            if sandbox_registry is not None:
                reg, lock = sandbox_registry
                with lock:
                    reg[role] = sandbox
            substrate = LocalSubstrate(
                role=role,
                worktree_path=worktree_root,
                spec=spec,
                run_artifacts_dir=run_artifacts_dir,
                observability=obs,
                docker_env=docker_env,
            )
            return role, sandbox, substrate
        except _SandboxInitCancelled:
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:  # noqa: BLE001
                    pass
                if sandbox_registry is not None:
                    reg, lock = sandbox_registry
                    with lock:
                        reg.pop(role, None)
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:  # noqa: BLE001
                    pass
                if sandbox_registry is not None:
                    reg, lock = sandbox_registry
                    with lock:
                        reg.pop(role, None)
            if cancel_event is not None and cancel_event.is_set():
                raise _SandboxInitCancelled(f"sandbox init cancelled for role '{role}'") from exc
            if attempt < _SANDBOX_CREATE_MAX_ATTEMPTS:
                backoff_sec = 2 ** attempt
                if cancel_event is not None:
                    if cancel_event.wait(timeout=backoff_sec):
                        raise _SandboxInitCancelled(f"sandbox init cancelled for role '{role}'") from exc
                else:
                    time.sleep(backoff_sec)
    raise last_exc


_SANDBOX_CLOSE_TIMEOUT_SEC = 30


def _close_sandboxes(sandboxes: Dict[str, Any]) -> None:
    def _safe_close(name: str, fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[loopbench] warning: sandbox close failed for {name}: {exc}", file=sys.stderr)

    for name, sandbox in sandboxes.items():
        close_fn = getattr(sandbox, "close", None)
        if not callable(close_fn):
            continue
        t = threading.Thread(target=_safe_close, args=(name, close_fn), daemon=True)
        t.start()
        t.join(timeout=_SANDBOX_CLOSE_TIMEOUT_SEC)
        if t.is_alive():
            print(f"[loopbench] warning: sandbox close timed out for {name} after {_SANDBOX_CLOSE_TIMEOUT_SEC}s", file=sys.stderr)


def _cancel_substrates(substrates: Dict[str, Any]) -> None:
    for substrate in substrates.values():
        cancel_fn = getattr(substrate, "cancel", None)
        if callable(cancel_fn):
            try:
                cancel_fn()
            except Exception:  # noqa: BLE001
                pass


_SUBSTRATE_TEARDOWN_TIMEOUT_SEC = 60


def _teardown_substrates(substrates: Dict[str, Any]) -> None:
    """Best-effort ``docker compose down`` on all substrates.

    Runs teardowns in parallel threads so one slow teardown does not
    block the others.  Errors are logged but never propagated.
    """
    threads: list[threading.Thread] = []
    for name, substrate in substrates.items():
        teardown_fn = getattr(substrate, "teardown", None)
        if not callable(teardown_fn):
            continue

        def _do_teardown(n=name, fn=teardown_fn):
            try:
                fn(timeout_sec=_SUBSTRATE_TEARDOWN_TIMEOUT_SEC)
            except Exception as exc:  # noqa: BLE001
                print(f"[loopbench] warning: substrate teardown failed for {n}: {exc}", file=sys.stderr)

        t = threading.Thread(target=_do_teardown, daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=_SUBSTRATE_TEARDOWN_TIMEOUT_SEC + 10)
