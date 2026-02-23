# Tasks & Adapters

LoopBench is intentionally **task-format agnostic**.
It supports multiple upstream datasets by normalizing them into `TaskPack`.

## 1) Canonical task.yaml schema

Each task directory contains a `task.yaml` with these top-level keys:

```yaml
task_id: "lb_000123"
kind: "repo_patch"   # repo_patch | devops_cycle | repo_gen | abcbench | devopsgym | repogenesis
difficulty: "medium"
language: "python"

workspace:
  base_repo: "repo.bundle"        # or "repo_template/" for repo_gen
  entrypoint: "README.task.md"    # shown to agents

substrate:
  kind: "compose"                 # compose | kind | none
  worktree_isolation: true        # one substrate instance per worktree
  up_cmd: "make up"
  down_cmd: "make down"
  public_validate_cmd: "make public-validate"
  public_validate_policy: "advisory"   # off | advisory | required

feedback_surfaces:
  logs: true
  metrics: true
  db: false
  browser: false

judge:
  hidden_validate_cmd: "bash hidden/validate.sh"
  timeout_sec: 1800

budgets:
  wall_clock_sec: 3600
  env_cycles: 30
  log_queries: 200
  metric_queries: 200
```

`public_validate_policy` behavior:
- `off`: do not run `env.public_validate` during review rounds.
- `advisory`: run if configured; failures/no-ops are feedback only (hidden judge still scores).
- `required`: run if configured and require a passing non-noop result before accepting a round.

## 2) Public vs hidden assets

- `public/` is mounted into agent sandboxes
- `hidden/` is only mounted into judge sandboxes

If you adapt a dataset where hidden tests already exist, keep them in `hidden/`.

## 3) Adapters

### 3.1 ABC-Bench adapter
ABC-Bench tasks typically include:
- a repo snapshot
- containerized services
- external HTTP-based integration tests

Adapter strategy:
- store the repo snapshot as `repo.bundle`
- expose only minimal smoke tests (public)
- keep external API verification in `hidden/validate.sh`
- map ABC-Bench’s docker compose assets into `public/workloads/` or `public/env/`
- when no public smoke exists, set `public_validate_policy: off` and omit `public_validate_cmd`

### 3.2 DevOps-Gym adapter
DevOps-Gym tasks cover:
- build/config
- monitoring
- issue resolving
- test generation

Adapter strategy:
- each DevOps-Gym task becomes a `devops_cycle` task
- substrate may be `none` (if the task is local build/config) or `compose/kind` if needed
- monitoring tasks should enable `feedback_surfaces.metrics/logs`

### 3.3 RepoGenesis adapter
RepoGenesis tasks start from a requirement document and expect a deployable microservice.
They evaluate:
- Pass@1 (functional correctness)
- API Coverage (AC)
- Deployment Success Rate (DSR)

Adapter strategy:
- `repo_template/` starts nearly empty with scaffolding rules
- `public/README.task.md` contains the requirements doc
- `hidden/validate.sh` runs black-box tests and deploy checks
- record DSR/AC in `RunManifest.metrics`

## 4) Task authoring guidelines (for closed-loop tasks)

A task should *force* loop closure by ensuring:
- static patching is insufficient; runtime feedback is required
- at least two feedback surfaces matter (e.g., logs + DB, metrics + http)
- the public smoke suite is helpful but incomplete
- hidden validation catches regressions and shortcuts

## 5) ABC-Bench import script

Use `scripts/import_abc_bench.py` to normalize extracted ABC-Bench `task_*` folders:

```bash
python scripts/import_abc_bench.py \
  --abc-root /path/to/abc/tasks \
  --out-root tasks/abcbench \
  --overwrite
```

Importer behavior:
- creates `repo.bundle` from the source repo folder
- writes LoopBench `task.yaml` with `kind: abcbench`
- copies instructions into `public/README.task.md`
- keeps validator assets in `hidden/` (`validate.sh`, `run-tests.sh`, `tests/`)
- copies compose assets into `public/env/` when available

At run time, `public/` assets are staged into every worktree under `.loopbench/public`, and a `public` symlink is created when the repo does not already have a `public/` directory.
