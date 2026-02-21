# Project Spec — LoopBench

## 0. Executive summary

LoopBench is a benchmark for **harness-engineered coding agents** operating in **realistic, executable** environments.

It is explicitly designed for tasks where agents must **close the loop**:

1. Inspect repository + docs
2. Implement changes across files
3. Build / configure / deploy the system-under-test (SUT)
4. Drive workloads / tests / user journeys
5. Observe runtime signals (logs / metrics / traces / DB / UI)
6. Iterate until clean
7. Produce a final patch that passes **hidden** validation

LoopBench additionally supports **multi-agent collaboration** with a fixed role topology:
- 1 × Planner+Reviewer (combined role instance by default)
- 2 × Coder agents (coder_a, coder_b)

## 1. Core design constraints (non-negotiable)

### 1.1 Two-plane evaluation: public vs hidden
- Agents can access **public** artifacts only.
- The **judge** runs **hidden** evaluation in a separate sandbox, applying the final patch.

### 1.2 Harness-first tool contract
- Agents interact through a stable set of **tool calls** (ToolRouter).
- Tool calls are:
  - typed
  - budgeted
  - logged (JSONL)
  - replayable

### 1.3 One run = one isolated universe
A run creates isolated resources keyed by `run_id`:
- workspace (repo + worktrees)
- substrate instance(s) (compose project or kind cluster)
- observability stack instance
- artifacts directory (events, diffs, logs)

### 1.4 No Docker-in-Docker (by default)
The default deployment substrate (compose/kind) is managed by an **EnvRunner** component
outside the agent sandbox. Agents call tools like `env.up`, `env.logs_query`, etc.

Rationale:
- avoids privileged agent sandboxes
- reduces leakage risk
- improves reproducibility and traceability

## 2. Terminology

- **TaskPack**: a directory bundle containing base repo + validation scripts.
- **Workspace**: a materialized checkout plus git worktrees for agents.
- **Sandbox**: where agents can read/write files and run compilation.
- **Substrate**: where SUT is deployed (compose/kind/etc).
- **EnvRunner**: controller-owned service that operates the substrate and observability.
- **Judge**: hidden validator that applies final patch and runs hidden checks.
- **ToolRouter**: the only interface agents use.

## 3. Task model

LoopBench supports multiple task kinds via adapters:

### 3.1 Patch tasks (existing repository)
- ABC-Bench-style backend tasks (containerized services + external API tests)
- DevOps-Gym tasks (build/config, monitoring, issue resolving, test gen)

### 3.2 Repo generation tasks (README → repository)
- RepoGenesis-style tasks: generate a full microservice repo from requirements and pass black-box tests.

Each task is packaged as:

```
tasks/<task_id>/
  task.yaml
  repo.bundle              # for patch tasks (or repo_template/ for repo-gen)
  public/
    README.task.md         # natural language task
    smoke/                 # public smoke checks (fast)
    workloads/             # drivers for exercising SUT
    observability/         # public dashboards/queries examples (optional)
  hidden/
    validate.sh            # hidden validator entrypoint
    tests/                 # hidden tests or oracle metadata
    oracle/                # optional: known-good patch, version pins
```

See `docs/TASKS.md` for exact schema and adapter mapping.

## 4. Multi-agent protocol

### 4.1 Roles
- **planner_reviewer**
  - decomposes problem
  - assigns subtasks to coders
  - reviews diffs and requests changes
  - may write missing regression tests (public only)
  - owns final merge decision
- **coder_a**
  - implements assigned subtask in its worktree
- **coder_b**
  - implements assigned subtask in its worktree

### 4.2 Collaboration primitive: git worktrees
Each role operates in a separate worktree:
- `worktrees/planner_reviewer/`
- `worktrees/coder_a/`
- `worktrees/coder_b/`

The planner role can:
- cherry-pick commits from coder worktrees
- request coders to rebase/amend
- run public validation on merge candidate

### 4.3 Required artifacts
For every run, the harness must persist:
- event log: `runs/<run_id>/events.jsonl`
- coordination DB: `runs/<run_id>/coordination/coordination.db`
- per-role summaries: `runs/<run_id>/role_summaries/<role>.md`
- per-role full stdio logs: `runs/<run_id>/role_stdio/`
- per-role role-runtime I/O snapshots (context/output/model traces): `runs/<run_id>/role_runtime/`
- per-role repo state snapshots (head/log/diff): `runs/<run_id>/repo_state/`
- input snapshots (task/config/env fingerprints): `runs/<run_id>/inputs/`
- final patch: `runs/<run_id>/final.patch`
- public validation outputs: `runs/<run_id>/public_validate/`
- judge outputs: `runs/<run_id>/hidden_validate/`

## 5. Scoring

### 5.1 Primary score (binary)
- `hidden_pass`: did the final patch pass hidden validation?

### 5.2 Secondary metrics (per task kind)
- **Deployability / DSR** (RepoGenesis-style): could SUT start successfully?
- API Coverage (RepoGenesis-style, if task defines endpoints)
- For ABC-Bench tasks: pass external API tests (already covered by hidden pass)
- For DevOps-Gym tasks: stage-level success (build/config, monitoring, etc)

### 5.3 Process metrics (diagnostic, not optimized initially)
- number of env cycles (up/down/redeploy)
- number of log/metric queries
- time-to-first-deploy, time-to-final-pass
- collaboration: number of review iterations, merge conflicts

## 6. Implementation milestones

M0 — Spec + schemas + repo layout (this skeleton)

M1 — Local runner (compose substrate)
- workspace provisioning
- ToolRouter w/ budgets
- EnvRunner for compose
- Judge sandbox (hidden)

M2 — Task adapters
- ABC-Bench adapter (import task bundles; map validation)
- DevOps-Gym adapter
- RepoGenesis adapter

M3 — Multi-agent harness baseline
- deterministic scheduling
- worktree merge strategy
- minimal planner/reviewer policy

M4 — Observability stack
- logs (Loki/Vector) + metrics (Prometheus)
- query tools exposed to agents
- ephemeral per worktree/substrate instance

M5 — Secure sandbox backend
- container backend hardening
- optional Firecracker / Kata backend
