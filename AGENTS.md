# AGENTS.md (Map)

This file is intentionally short. It points agents to the *real* specs in `docs/`.

## Start here
- `docs/PROJECT_SPEC.md` — overall benchmark spec and invariants
- `docs/ARCHITECTURE.md` — component boundaries & interfaces
- `docs/CONTAINERIZATION.md` — how sandboxes + envs are containerized
- `docs/HOST_MIGRATION.md` — move from nested/containerized dev host to stable eval host
- `docs/ABC_TOOL_PARITY.md` — ABC/Terminal-Bench-derived tool surface requirements
- `docs/TASKS.md` — task packaging, adapters, public/hidden validation
- `docs/ROLES.md` — planner/reviewer + 2 coder protocol & artifacts
- `docs/SECURITY.md` — threat model, anti-leakage, sandbox backends

## Code entrypoints
- `loopbench/cli.py` — CLI: `run`, `judge`, `pack`, `replay`
- `loopbench/controller.py` — run lifecycle orchestration
- `loopbench/harness.py` — deterministic planner/coder/reviewer loop
- `loopbench/coordination.py` — SQLite team coordination protocol (tasks/claims/messages)
- `loopbench/team_protocol.py` — planner/coder coordination conventions over SQLite
- `loopbench/run_artifacts.py` — centralized run artifact/status/summary writing
- `loopbench/reproducibility.py` — durable input/repo-state snapshots for run replay/debug
- `loopbench/observability.py` — logs/metrics query backend selection (artifact, Loki, Prometheus)
- `loopbench/docker_runtime.py` — shared Docker endpoint/env helpers for judge runtime
- `loopbench/repo_materializer.py` — shared repo bootstrap + git identity helpers
- `loopbench/path_utils.py` — root-bounded path resolution helper
- `loopbench/io_utils.py` — shared YAML mapping loader
- `loopbench/time_utils.py` — shared wall-clock timestamp helper
- `loopbench/tools.py` — ToolRouter with budgets + JSONL logging
- `loopbench/workspace.py` — workspace + git worktree provisioning
- `loopbench/task_loader.py` — task normalization from `task.yaml`
- `loopbench/judge.py` — hidden validation on fresh checkout + patch apply
- `loopbench/experiments.py` — multi-rollout experiment matrix runner + summaries
- `loopbench/preflight.py` — infra readiness checks
- `loopbench/e2b_sandbox.py` — E2B Firecracker-backed sandbox backend
- `loopbench/sandbox_factory.py` — sandbox backend selection
- `loopbench/interfaces.py` — typed interfaces for sandboxes, tools, tasks
- `loopbench/schema.py` — Pydantic schemas (TaskPack, RunManifest, ToolCalls)
- `scripts/import_abc_bench.py` — ABC-Bench task importer into LoopBench format
- `scripts/openrouter_role_driver.py` — OpenRouter-backed shell role driver (planner + coders)
- `scripts/build_images.sh` — container image build script
- `scripts/start_docker_daemon.sh` — helper to bring up local docker daemon for judge runtime
- `scripts/with_local_docker.sh` — wrapper to ensure local daemon before a LoopBench command

## Conventions
- All run outputs go under `runs/<run_id>/`
- Any behavior that affects evaluation must be encoded in versioned files, not human memory:
  config YAML, task.yaml, scripts, schemas, docs.
