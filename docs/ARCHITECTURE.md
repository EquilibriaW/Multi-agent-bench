# Architecture

This document defines **hard boundaries** between components.
The benchmark will be easier to build, audit, and extend if these boundaries stay crisp.

## 1) High-level component graph

```
+--------------------+        +---------------------+
|  Controller (CLI)  |        |   Model Provider    |
|  loopbench run     |<------>| (OpenAI, local, etc)|
+---------+----------+        +----------+----------+
          |
          | provisions run workspace + sandboxes
          v
+---------+----------+   tool calls   +------------------+
| ToolRouter         |<-------------->| MultiAgentHarness |
| budgets + logging  |               | (planner+coders)  |
+---------+----------+               +------------------+
          |
          | routes to concrete implementations
          v
+---------+----------+        +-------------------------+
| Agent Sandbox      |        | EnvRunner + Substrate   |
| (repo edits/comp.) |        | (deploy + observe SUT)  |
+---------+----------+        +-------------------------+
          |
          | final.patch
          v
+--------------------+
| Judge Sandbox       |
| (hidden validation) |
+--------------------+
```

### Boundary A: Controller ↔ Agent Harness
- The controller owns: provisioning, budgets, recording, judging.
- The agent harness owns: role policy (planning, review, merge, iteration).

### Boundary B: ToolRouter ↔ Everything else
Agents never call the OS directly. All actions must be ToolCalls:
- `repo.*` tools: file ops, patch apply, git operations
- `env.*` tools: deploy, validation, logs/metrics/http
- `meta.*`: budgets, time, run info

### Boundary C: Sandbox ↔ Substrate
- Sandbox is *code editing + compilation*
- Substrate is *running system-under-test and emitting runtime signals*
This separation avoids privileged agent sandboxes and improves traceability.

## 2) Controller responsibilities

### 2.1 Run lifecycle
- allocate `run_id`
- materialize workspace
- spin up EnvRunner/substrate resources
- start MultiAgentHarness loop
- collect final patch
- run judge hidden validation
- write `RunManifest`

### 2.2 Deterministic recording
Every run writes:
- `events.jsonl` (tool calls + tool results)
- `manifest.json` (RunManifest)
- `inputs/` (resolved task/runtime/agents snapshots + environment fingerprints)
- `repo_state/` (per-role HEAD/log/diff from base commit)
- `role_runtime/` (per-role runtime context/output and model I/O traces)
- `role_stdio/` (full role stdout/stderr)
- `final.patch`
- per-role summaries

## 3) MultiAgentHarness responsibilities

The harness implements:
- role scheduling (who acts next)
- context propagation across roles
- DB-backed coordination protocol (assign/claim/message persistence)
- reviewer gate: when to accept/reject coder changes
- merge strategy (cherry-pick, squash, rebase)
- safety checks (avoid editing forbidden paths, secrets)

The harness must be **model-agnostic**:
- it can be driven by Codex, DeepAgents, OpenHands, etc
- but tool semantics and artifacts remain stable

Coordination artifact:
- `runs/<run_id>/coordination/coordination.db` (SQLite)
  - `tasks`: pending/claimed/completed/failed task lifecycle
  - `messages`: planner<->coder communication log
  - `claims`: claim/complete/fail audit events

Implementation split (for maintainability):
- `loopbench/team_protocol.py`: coordination message/task conventions
- `loopbench/run_artifacts.py`: status/summaries/public-validate artifact writing
- `loopbench/observability.py`: logs/metrics backend adapters and query facade
- shared utilities (`repo_materializer.py`, `path_utils.py`, `io_utils.py`, `time_utils.py`)
- `loopbench/harness.py`: orchestration flow only

## 4) Sandboxes and substrates

### 4.1 Sandbox (agent-facing)
Minimum capabilities:
- read/write/edit files in its worktree
- run language toolchains (python, node, go, java, etc depending on task)
- NO hidden tests
- ideally, no direct Docker socket access

### 4.2 Substrate (SUT runtime)
A substrate instance is bound to a specific worktree.
Examples:
- docker compose project: `COMPOSE_PROJECT_NAME=lb_<run_id>_<role>`
- kind cluster: `kind create cluster --name lb-<run_id>-<role>`

Substrate must expose:
- up/down/status
- public validation
- logs tail + query
- metrics query
- HTTP request driver (optional)
- browser journey driver (optional)

## 5) Judge sandbox

Judge runs in a separate sandbox backend to prevent leakage.
Protocol:
1. fresh checkout of base repo from `repo.bundle`
2. apply `final.patch`
3. run `hidden/validate.sh`
4. output `hidden_validate/` artifacts

## 6) Experiment runner

LoopBench includes an experiment runner for repeated rollout evaluation:
- fixed triad topology (planner/reviewer + 2 coders)
- lineup matrix across role model assignments
- repeated rollouts per task
- aggregate summary + feedback artifacts

Artifacts:
- `experiments/<experiment_id>/results.jsonl`
- `experiments/<experiment_id>/summary.json`
- `experiments/<experiment_id>/feedback.md`
