# Containerization & Sandboxing Plan

This document defines the reference deployment for LoopBench.

## 0) Goals

- reproducible runs on a single Linux host with Docker
- strong isolation between:
  - agents (public-only)
  - environment substrate (deploy + observe)
  - judge (hidden validation)
- no Docker-in-Docker by default
- option to upgrade isolation to microVM / Kata / unikernel later

## 1) Reference topology (v0)

For each `run_id`, the controller creates three execution contexts:

1) **Agent Sandbox**
- unprivileged container
- mounts workspace worktrees as read/write
- network: *disabled* by default (agents use `env.http_request` tool)
- contains compilers/runtime needed to edit/build code, but no Docker socket

2) **EnvRunner**
- privileged *service* container (or host process) that owns:
  - docker compose/kind lifecycle
  - observability stack
  - log/metric querying
  - HTTP driver / browser driver
- has Docker socket access (or runs on the host)

3) **Judge Sandbox**
- separate container (fresh filesystem)
- has access to hidden artifacts
- applies `final.patch` and runs hidden validation
- network disabled by default
- for Docker-heavy hidden suites (e.g., ABC-Bench), judge runtime can require
  host Docker daemon access while agent sandboxes remain unprivileged.

## 2) Why avoid Docker-in-Docker

- security: privileged access would allow agents to introspect the host or mount secrets
- reproducibility: nested Docker introduces fragile layer caching and kernel feature variance
- logging: harder to attribute actions to tool calls

Instead, agents request deploy/observe actions through ToolRouter → EnvRunner.

## 3) Workspace and mounts

Directory layout (within the host):

```
workspaces/<run_id>/
  base/                # checked-out repo
  worktrees/
    planner_reviewer/
    coder_a/
    coder_b/
  caches/
    pip/
    npm/
    go/
    m2/
runs/<run_id>/
  events.jsonl
  manifest.json
  artifacts/
```

Mount policy:
- Agent Sandbox mounts:
  - `workspaces/<run_id>/worktrees/<role>` at `/workspace`
  - shared caches at `/caches` (optional)
- EnvRunner mounts:
  - the same worktree at `/workspace`
  - docker socket
  - `runs/<run_id>/artifacts` for logs/metrics snapshots

Judge mounts:
- `repo.bundle` + hidden assets (read-only)
- `runs/<run_id>/final.patch`

## 4) Substrate execution modes

### 4.1 Docker Compose substrate (default)
- EnvRunner runs `docker compose up -d` with
  - `COMPOSE_PROJECT_NAME=lb_<run_id>_<worktree>`
- Use distinct port ranges per worktree to allow parallelism.

Port allocation rule:
- derive a base port from a hash of `(run_id, worktree_name)`
- reserve a contiguous block, e.g. 20 ports per worktree
- record mapping in `runs/<run_id>/ports.json`

### 4.2 kind/k3d substrate (v1)
- EnvRunner creates one cluster per worktree:
  - `kind create cluster --name lb-<run_id>-<worktree>`
- This enables kubectl-driven troubleshooting tasks (services, ingress, probes, selectors).

## 5) Observability stack

The benchmark treats observability as a first-class **feedback surface**.

In v0:
- logs: Loki (LogQL) or equivalent
- metrics: Prometheus (PromQL) or equivalent
- optional traces: Tempo/OTel

Per worktree:
- observability stack is ephemeral and torn down with the substrate instance
- ToolRouter exposes:
  - `env.logs_query(query)`
  - `env.metrics_query(query)`
  - (optional) `env.traces_query(query)`

Runtime wiring:
- `env_runner.observability.logs_endpoint` is used when `logs: loki`
- `env_runner.observability.metrics_endpoint` is used when `metrics: prometheus`
- if Loki endpoint is unset, logs queries fall back to local run artifacts
- if Prometheus endpoint is unset, metrics queries return structured unavailability

## 6) Network policy

Default policy:
- Agent Sandbox has no network.
- EnvRunner has network only to:
  - substrate containers
  - observability stack
- No internet access for any container in evaluation mode.

If a task requires internet (rare), it must declare:
- explicit allowlisted domains
- deterministic mirrors / cached dependencies preferred

## 7) Sandboxing backends (v2 plan)

LoopBench defines a pluggable `SandboxBackend` interface:

- `docker_container` (v0): fast, simple; weaker isolation (shared kernel)
- `kata_container` (v2): VM-backed containers; OCI compatible; k8s-friendly
- `firecracker_microvm` (v2): strong isolation; snapshots can reduce startup latency
- `unikernel` (research): potential extreme minimal attack surface and fast boot,
  but Python + multi-process support is non-trivial and image build/distribution
  becomes a key design problem.

The backend choice should not affect the **tool contract** or task format.

## 8) Current repository assets (v0)

Implemented image definitions:
- `containers/agent-sandbox.Dockerfile`
- `containers/env-runner.Dockerfile`
- `containers/judge.Dockerfile`

Build helper:
- `scripts/build_images.sh`
- `scripts/start_docker_daemon.sh` (optional helper for constrained hosts where
  Docker service is not already running)
- `scripts/with_local_docker.sh` (ensures daemon before invoking LoopBench command)

Example:

```bash
REGISTRY=ghcr.io/your-org TAG=v0.1.0 scripts/build_images.sh
```

Judge endpoint option (integrity-first):
- configure `judge.docker_host` in runtime YAML to point the judge to a stable
  Docker endpoint (`DOCKER_HOST`), e.g. `ssh://loopbench@judge-vm`.
- this avoids nested-host daemon flakiness while preserving agent sandbox isolation.

## 9) E2B backend option

`sandbox_backend.kind: e2b_firecracker` is supported for agent command isolation.

Notes:
- requires `E2B_API_KEY` (or configured `api_key_env`)
- requires Python package `e2b-code-interpreter`
- currently keeps local worktree as orchestration source-of-truth and syncs to/from E2B for command execution
