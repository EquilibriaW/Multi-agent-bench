# Experiments

LoopBench experiments should compare **role orchestration policies** under repeated rollouts:
- 1x planner/reviewer
- 2x coders
- no subagent topology (agent team only)

## Why repeated rollouts

Single-run outcomes are noisy. Use repeated rollouts per `(task, lineup)` to estimate:
- hidden pass rate
- completion rate
- coordination pressure

## CLI

```bash
loopbench experiment --spec configs/experiment.example.yaml
```

Outputs:
- `experiments/<experiment_id>/results.jsonl`
- `experiments/<experiment_id>/summary.json`
- `experiments/<experiment_id>/feedback.md`

`summary.json` includes an ABC-style failure taxonomy aggregate:
- `syntax_errors`
- `path_missing`
- `dependency_missing`
- `compilation_errors`
- `logic_errors`
- `other`

Capacity fields are also recorded:
- `max_parallel_rollouts`
- `required_agent_sandboxes_peak` (= `max_parallel_rollouts * 3`)
- `available_agent_sandboxes`
- `sandbox_headroom`

Trial profile aligned with team-debug pass:
- `configs/experiment.trial.2tasks_3rollouts.yaml`
- 2 tasks × 3 rollouts with `max_parallel_rollouts: 6`
- peak team capacity = `6 * 3 = 18` agent sandboxes

Coordination metrics recorded per rollout:
- `coordination_messages_total`
- `coordination_claim_events_total`
- `coordination_tasks_completed`
- `coordination_tasks_failed`

## Rollout design guidance

- Start with at least `rollouts_per_task: 5` for stable comparisons.
- Set `max_parallel_rollouts` explicitly for stress tests.
- Keep role topology fixed when comparing model lineups.
- Add more tasks before adding more knobs.
- Track failure causes (role bootstrap failures, merge failures, hidden validator failures) before policy changes.

## Driver policy

- Production lineups should use `driver: shell` with real role runtimes.
- `driver: noop` is blocked by default in `loopbench run`.
- Use `--allow-noop` only for harness smoke tests.

## Infra preflight

```bash
loopbench preflight --config configs/runtime.yaml
```

If `sandbox_backend.kind` is `e2b_firecracker`, preflight also checks:
- `E2B_API_KEY` env presence
- `e2b_code_interpreter` SDK availability

Preflight also reports observability readiness:
- `observability.logs=loki` expects `logs_endpoint`
- `observability.metrics=prometheus` expects `metrics_endpoint`
- `observability.traces=langsmith` expects:
  - `langsmith` Python package installed
  - `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) set
  - optional `traces_endpoint` for self-hosted LangSmith

LangSmith tracing integration is fail-open:
- if tracing fails at runtime, benchmark execution continues and trace status is written to `runs/<run_id>/trace/session.json`
- baseline runs remain unchanged when `observability.traces: none`
