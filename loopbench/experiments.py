"""
loopbench.experiments

Experiment runner for statistically meaningful benchmark rollouts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from .controller import replay_events, run_benchmark
from .hidden_result import hidden_failure_reason, hidden_stderr_excerpt, normalize_error
from .io_utils import read_yaml_mapping


class LineupSpec(BaseModel):
    name: str
    planner_model: str
    coder_a_model: str
    coder_b_model: str
    role_command: str
    role_env: Dict[str, str] = Field(default_factory=dict)


class ExperimentSpec(BaseModel):
    experiment_id: str
    tasks: List[str]
    lineups: List[LineupSpec]
    rollouts_per_task: int = 5
    max_parallel_rollouts: int = 1
    available_agent_sandboxes: Optional[int] = None
    runtime_config: str = "configs/runtime.yaml"


@dataclass
class RolloutRecord:
    experiment_id: str
    lineup: str
    task_dir: str
    task_id: str
    rollout_index: int
    run_id: str
    status: str
    hidden_pass: Optional[bool]
    public_pass: Optional[bool]
    wall_clock_sec: Optional[float]
    review_iterations: Optional[int]
    merge_conflicts: Optional[int]
    tool_calls: Optional[int]
    env_cycles_used: Optional[int]
    log_queries_used: Optional[int]
    metric_queries_used: Optional[int]
    role_phase_failures: Optional[int]
    coordination_messages_total: Optional[int]
    coordination_claim_events_total: Optional[int]
    coordination_tasks_completed: Optional[int]
    coordination_tasks_failed: Optional[int]
    failure_bucket: Optional[str]
    failure_reason: Optional[str]
    manifest_path: Optional[str]


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    spec_path = Path(path).resolve()
    data = read_yaml_mapping(spec_path, label="experiment spec")
    return ExperimentSpec.model_validate(data)


def run_experiment(
    *,
    spec_path: str | Path,
    project_root: str | Path,
    tasks_root: Optional[str | Path] = None,
    allow_noop: bool = False,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    spec_path = Path(spec_path).resolve()
    spec = load_experiment_spec(spec_path)

    experiment_dir = project_root / "experiments" / spec.experiment_id
    if experiment_dir.exists():
        raise FileExistsError(f"experiment already exists: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=False)

    runs_root = project_root / "runs"
    results_path = experiment_dir / "results.jsonl"
    summary_path = experiment_dir / "summary.json"
    feedback_path = experiment_dir / "feedback.md"

    task_dirs = _resolve_task_dirs(
        project_root=project_root,
        tasks_root=Path(tasks_root).resolve() if tasks_root else None,
        task_entries=spec.tasks,
    )

    records: List[RolloutRecord] = []
    runtime_config_path = str((project_root / spec.runtime_config).resolve())

    jobs: List[Dict[str, Any]] = []
    for lineup in spec.lineups:
        agents_cfg_path = _write_agents_config_for_lineup(
            experiment_dir=experiment_dir,
            lineup=lineup,
        )
        for task_dir in task_dirs:
            task_id = task_dir.name
            for rollout_index in range(1, spec.rollouts_per_task + 1):
                jobs.append(
                    {
                        "lineup": lineup,
                        "task_dir": task_dir,
                        "task_id": task_id,
                        "rollout_index": rollout_index,
                        "run_id": _build_run_id(spec.experiment_id, lineup.name, task_id, rollout_index),
                        "agents_cfg_path": agents_cfg_path,
                    }
                )

    if spec.max_parallel_rollouts < 1:
        raise ValueError("max_parallel_rollouts must be >= 1")

    max_workers = min(spec.max_parallel_rollouts, max(len(jobs), 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for job in jobs:
            futures.append(
                pool.submit(
                    _run_rollout_job,
                    experiment_id=spec.experiment_id,
                    lineup=job["lineup"],
                    task_dir=job["task_dir"],
                    task_id=job["task_id"],
                    rollout_index=job["rollout_index"],
                    run_id=job["run_id"],
                    runtime_config_path=runtime_config_path,
                    agents_config_path=str(job["agents_cfg_path"]),
                    project_root=project_root,
                    runs_root=runs_root,
                    allow_noop=allow_noop,
                )
            )

        for fut in as_completed(futures):
            record = fut.result()
            records.append(record)
            _append_jsonl(results_path, asdict(record))

    summary = _summarize_records(records)
    summary_payload = {
        "experiment_id": spec.experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_records": len(records),
        "max_parallel_rollouts": spec.max_parallel_rollouts,
        "required_agent_sandboxes_peak": spec.max_parallel_rollouts * 3,
        "available_agent_sandboxes": spec.available_agent_sandboxes,
        "sandbox_headroom": _sandbox_headroom(spec.available_agent_sandboxes, spec.max_parallel_rollouts * 3),
        "summary": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    feedback_path.write_text(_render_feedback(summary_payload, records), encoding="utf-8")

    return {
        "experiment_dir": str(experiment_dir),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "feedback_path": str(feedback_path),
        "summary": summary_payload,
    }


def _run_rollout_job(
    *,
    experiment_id: str,
    lineup: LineupSpec,
    task_dir: Path,
    task_id: str,
    rollout_index: int,
    run_id: str,
    runtime_config_path: str,
    agents_config_path: str,
    project_root: Path,
    runs_root: Path,
    allow_noop: bool,
) -> RolloutRecord:
    def make_record(**overrides: Any) -> RolloutRecord:
        payload: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "lineup": lineup.name,
            "task_dir": str(task_dir),
            "task_id": task_id,
            "rollout_index": rollout_index,
            "run_id": run_id,
            "status": "infra_error",
            "hidden_pass": None,
            "public_pass": None,
            "wall_clock_sec": None,
            "review_iterations": None,
            "merge_conflicts": None,
            "tool_calls": None,
            "env_cycles_used": None,
            "log_queries_used": None,
            "metric_queries_used": None,
            "role_phase_failures": None,
            "coordination_messages_total": None,
            "coordination_claim_events_total": None,
            "coordination_tasks_completed": None,
            "coordination_tasks_failed": None,
            "failure_bucket": None,
            "failure_reason": None,
            "manifest_path": None,
        }
        payload.update(overrides)
        return RolloutRecord(**payload)

    try:
        manifest = run_benchmark(
            task_dir=str(task_dir),
            run_id=run_id,
            runtime_config_path=runtime_config_path,
            agents_config_path=agents_config_path,
            project_root=project_root,
            allow_noop=allow_noop,
        )
        run_dir = runs_root / run_id
        try:
            extra = replay_events(str(run_dir))
        except FileNotFoundError:
            extra = {
                "tool_calls": None,
                "env_cycles_used": None,
                "log_queries_used": None,
                "metric_queries_used": None,
                "role_phase_failures": None,
            }
        hidden_infra_error = bool(manifest.metrics.get("hidden_infra_error"))
        status = "infra_error" if hidden_infra_error else "ok"
        failure_bucket = _infer_failure_bucket(
            run_dir=run_dir,
            status="ok" if status == "ok" else "error",
            hidden_pass=manifest.hidden_pass,
            fallback_text=None,
        )
        failure_reason = None
        if hidden_infra_error:
            failure_bucket = "infra_runtime"
            failure_reason = hidden_stderr_excerpt(run_dir)
        elif manifest.hidden_pass is False:
            failure_reason = hidden_failure_reason(run_dir)

        return make_record(
            task_id=manifest.task_id,
            status=status,
            hidden_pass=manifest.hidden_pass,
            public_pass=manifest.public_pass,
            wall_clock_sec=_as_float(manifest.metrics.get("wall_clock_sec")),
            review_iterations=_as_int(manifest.metrics.get("review_iterations")),
            merge_conflicts=_as_int(manifest.metrics.get("merge_conflicts")),
            tool_calls=extra["tool_calls"],
            env_cycles_used=extra["env_cycles_used"],
            log_queries_used=extra["log_queries_used"],
            metric_queries_used=extra["metric_queries_used"],
            role_phase_failures=extra["role_phase_failures"],
            coordination_messages_total=_as_int(manifest.metrics.get("coordination_messages_total")),
            coordination_claim_events_total=_as_int(manifest.metrics.get("coordination_claim_events_total")),
            coordination_tasks_completed=_as_int(manifest.metrics.get("coordination_tasks_completed")),
            coordination_tasks_failed=_as_int(manifest.metrics.get("coordination_tasks_failed")),
            failure_bucket=failure_bucket,
            failure_reason=failure_reason,
            manifest_path=str(run_dir / "manifest.json"),
        )
    except Exception as exc:  # noqa: BLE001
        reason = str(exc)
        status = "timeout" if _is_timeout_text(reason) else "infra_error"
        bucket = "timeout" if status == "timeout" else "infra_runtime"
        return make_record(
            status=status,
            failure_bucket=bucket,
            failure_reason=reason,
        )


def _resolve_task_dirs(
    *,
    project_root: Path,
    tasks_root: Optional[Path],
    task_entries: List[str],
) -> List[Path]:
    resolved: List[Path] = []
    for entry in task_entries:
        p = Path(entry)
        if p.is_absolute() and p.exists():
            resolved.append(p.resolve())
            continue

        candidates = []
        if tasks_root is not None:
            candidates.append((tasks_root / entry).resolve())
        candidates.append((project_root / entry).resolve())

        found = next((c for c in candidates if c.exists()), None)
        if not found:
            raise FileNotFoundError(f"task entry not found: {entry}")
        resolved.append(found)

    return resolved


def _write_agents_config_for_lineup(*, experiment_dir: Path, lineup: LineupSpec) -> Path:
    cfg_dir = experiment_dir / "agents_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    role_env = {
        "LOOPBENCH_TEAM_MODE": "agent_team",
        "LOOPBENCH_DISABLE_SUBAGENTS": "1",
    }
    role_env.update(lineup.role_env)
    role_specs = [
        ("planner_reviewer", lineup.planner_model),
        ("coder_a", lineup.coder_a_model),
        ("coder_b", lineup.coder_b_model),
    ]
    roles = [
        {
            "name": name,
            "model": model,
            "max_tokens": 200000,
            "driver": "shell",
            "command": lineup.role_command,
            "env": role_env,
        }
        for name, model in role_specs
    ]

    payload = {
        "roles": roles,
        "scheduling": {
            "mode": "phased",
            "max_review_rounds": 3,
            "phases": [
                {"name": "bootstrap", "active_roles": ["planner_reviewer"], "exit_condition": "plan_written"},
                {"name": "implementation", "active_roles": ["coder_a", "coder_b"], "exit_condition": "coders_done"},
                {"name": "review", "active_roles": ["planner_reviewer", "coder_a", "coder_b"], "exit_condition": "public_validate_pass"},
                {"name": "finalize", "active_roles": ["planner_reviewer"], "exit_condition": "final_patch_written"},
            ],
        },
        "budgets": {
            "wall_clock_sec": 3600,
            "tool_calls": 800,
            "env_cycles": 30,
            "log_queries": 200,
            "metric_queries": 200,
            "http_requests": 300,
        },
    }

    path = cfg_dir / f"{lineup.name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _build_run_id(experiment_id: str, lineup_name: str, task_id: str, rollout_index: int) -> str:
    base = f"{experiment_id}_{lineup_name}_{task_id}_r{rollout_index:02d}"
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base)
    return safe[:120]


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=True) + "\n")


def _summarize_records(records: List[RolloutRecord]) -> Dict[str, Any]:
    by_lineup: Dict[str, List[RolloutRecord]] = defaultdict(list)
    for rec in records:
        by_lineup[rec.lineup].append(rec)

    summary: Dict[str, Any] = {}

    for lineup, recs in by_lineup.items():
        n = len(recs)
        n_ok = sum(1 for r in recs if r.status == "ok")
        hidden_passes = sum(1 for r in recs if r.status == "ok" and r.hidden_pass is True)
        public_passes = sum(1 for r in recs if r.status == "ok" and r.public_pass is True)

        wall = [r.wall_clock_sec for r in recs if r.wall_clock_sec is not None]
        reviews = [r.review_iterations for r in recs if r.review_iterations is not None]
        conflicts = [r.merge_conflicts for r in recs if r.merge_conflicts is not None]
        tool_calls = [r.tool_calls for r in recs if r.tool_calls is not None]
        env_cycles = [r.env_cycles_used for r in recs if r.env_cycles_used is not None]
        role_phase_failures = [r.role_phase_failures for r in recs if r.role_phase_failures is not None]
        coord_messages = [r.coordination_messages_total for r in recs if r.coordination_messages_total is not None]
        coord_claims = [r.coordination_claim_events_total for r in recs if r.coordination_claim_events_total is not None]
        coord_tasks_completed = [r.coordination_tasks_completed for r in recs if r.coordination_tasks_completed is not None]
        coord_tasks_failed = [r.coordination_tasks_failed for r in recs if r.coordination_tasks_failed is not None]

        failure_counter = Counter(
            normalize_error(r.failure_reason)
            for r in recs
            if ((r.status != "ok") or (r.hidden_pass is False)) and r.failure_reason
        )
        bucket_counter = Counter(
            r.failure_bucket or "unknown"
            for r in recs
            if (r.status != "ok") or (r.hidden_pass is False)
        )

        pressure_index = []
        for r in recs:
            if r.review_iterations is None or r.merge_conflicts is None or r.env_cycles_used is None:
                continue
            pressure_index.append(r.review_iterations + r.merge_conflicts + r.env_cycles_used)

        summary[lineup] = {
            "n_rollouts": n,
            "n_evaluable": n_ok,
            "run_completion_rate": _ratio(n_ok, n),
            "hidden_pass_rate": _ratio(hidden_passes, n_ok),
            "public_pass_rate": _ratio(public_passes, n_ok),
            "median_wall_clock_sec": _median_or_none(wall),
            "mean_review_iterations": _mean_or_none(reviews),
            "mean_merge_conflicts": _mean_or_none(conflicts),
            "mean_tool_calls": _mean_or_none(tool_calls),
            "mean_env_cycles_used": _mean_or_none(env_cycles),
            "mean_role_phase_failures": _mean_or_none(role_phase_failures),
            "mean_coordination_messages_total": _mean_or_none(coord_messages),
            "mean_coordination_claim_events_total": _mean_or_none(coord_claims),
            "mean_coordination_tasks_completed": _mean_or_none(coord_tasks_completed),
            "mean_coordination_tasks_failed": _mean_or_none(coord_tasks_failed),
            "coordination_pressure_index_mean": _mean_or_none(pressure_index),
            "failure_buckets": dict(bucket_counter),
            "top_failures": failure_counter.most_common(8),
        }

    return summary


def _render_feedback(summary_payload: Dict[str, Any], records: List[RolloutRecord]) -> str:
    lines = []
    lines.append(f"# Experiment Feedback: {summary_payload['experiment_id']}")
    lines.append("")
    lines.append("## Goal")
    lines.append(
        "Estimate whether orchestration policy (planner/reviewer + two coders) improves hidden pass rate "
        "without causing unsustainable coordination pressure."
    )
    lines.append("")
    lines.append("## Capacity Plan")
    lines.append(f"- max_parallel_rollouts: {summary_payload.get('max_parallel_rollouts')}")
    lines.append(f"- required_agent_sandboxes_peak: {summary_payload.get('required_agent_sandboxes_peak')}")
    lines.append(f"- available_agent_sandboxes: {summary_payload.get('available_agent_sandboxes')}")
    lines.append(f"- sandbox_headroom: {summary_payload.get('sandbox_headroom')}")
    lines.append("")

    lines.append("## Lineup Summary")
    for lineup, metrics in summary_payload["summary"].items():
        lines.append(
            f"- {lineup}: hidden_pass_rate={metrics['hidden_pass_rate']:.3f}, "
            f"run_completion_rate={metrics['run_completion_rate']:.3f}, "
            f"coordination_pressure_index_mean={metrics['coordination_pressure_index_mean']}, "
            f"mean_coordination_messages_total={metrics['mean_coordination_messages_total']}"
        )
    lines.append("")

    failing = [r for r in records if (r.status != "ok") or (r.hidden_pass is False)]
    if failing:
        lines.append("## Top Failures")
        counts = Counter(normalize_error(r.failure_reason) for r in failing if r.failure_reason)
        for reason, count in counts.most_common(10):
            lines.append(f"- {reason}: {count}")
        lines.append("")

    bucket_counts = Counter(
        r.failure_bucket or "unknown"
        for r in records
        if (r.status != "ok") or (r.hidden_pass is False)
    )
    if bucket_counts:
        lines.append("## Failure Buckets (ABC-style taxonomy)")
        for bucket, count in bucket_counts.most_common():
            lines.append(f"- {bucket}: {count}")
        lines.append("")

    lines.append("## Next Feedback Loop")
    lines.append("- Increase rollouts on tasks with high variance and compare hidden_pass_rate confidence intervals.")
    lines.append("- Inspect runs with high coordination pressure index and low hidden pass to identify backpressure bottlenecks.")
    lines.append("- Track role-phase failure causes from role summaries before changing orchestration policy.")
    lines.append("")

    return "\n".join(lines) + "\n"


def _infer_failure_bucket(
    *,
    run_dir: Path,
    status: str,
    hidden_pass: Optional[bool],
    fallback_text: Optional[str],
) -> Optional[str]:
    if status != "ok":
        return _classify_failure_text(fallback_text)
    if hidden_pass is True:
        return None

    hidden_log = run_dir / "hidden_validate" / "result.log"
    if hidden_log.exists():
        text = hidden_log.read_text(encoding="utf-8", errors="replace")
        return _classify_failure_text(text)
    return _classify_failure_text(fallback_text)


def _classify_failure_text(text: Optional[str]) -> str:
    if not text:
        return "other"

    s = text.lower()

    syntax_patterns = [
        "syntaxerror",
        "parse error",
        "unexpected token",
        "invalid syntax",
    ]
    path_patterns = [
        "no such file or directory",
        "file not found",
        "cannot find",
        "path does not exist",
    ]
    dep_patterns = [
        "module not found",
        "cannot find module",
        "importerror",
        "modulenotfounderror",
        "no matching distribution found",
        "dependency missing",
    ]
    compile_patterns = [
        "compilation failed",
        "build failed",
        "compile error",
        "failed to compile",
        "undefined reference",
        "linker",
        "maven",
        "gradle",
        "cargo build",
    ]
    logic_patterns = [
        "assertionerror",
        "assert failed",
        "expected",
        "actual",
        "test failed",
        "status code",
        "wrong output",
    ]

    if any(p in s for p in syntax_patterns):
        return "syntax_errors"
    if any(p in s for p in path_patterns):
        return "path_missing"
    if any(p in s for p in dep_patterns):
        return "dependency_missing"
    if any(p in s for p in compile_patterns):
        return "compilation_errors"
    if any(p in s for p in logic_patterns):
        return "logic_errors"
    return "other"


def _is_timeout_text(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    patterns = [
        "timed out",
        "timeout",
        "deadline exceeded",
        "wall clock",
        "max review rounds",
    ]
    return any(p in s for p in patterns)


def _ratio(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return None


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(statistics.fmean(xs))


def _median_or_none(values: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(statistics.median(xs))


def _sandbox_headroom(available: Optional[int], required: int) -> Optional[int]:
    if available is None:
        return None
    return int(available) - int(required)
