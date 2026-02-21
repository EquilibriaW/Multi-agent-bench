# Host Migration (Container Dev -> Bare-Metal Eval)

This migration removes nested-Docker confounds during benchmark runs while
keeping agent isolation in E2B.

## Target topology

- Development host: Codex in container (current workflow is fine).
- Evaluation host: run `loopbench` commands on host shell (not in container).
- Agent execution: E2B (`sandbox_backend.kind: e2b_firecracker`).
- Judge/substrate Docker: host Docker daemon (or dedicated remote daemon).

## 1) Commit current project state

Run in project root:

```bash
git init
git add .
git commit -m "loopbench: e2b team orchestration + docker-capable judge + infra hardening"
```

Optional push:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## 2) Clone on evaluation machine

```bash
git clone <your-repo-url>
cd loopbench_project_spec
```

## 3) Create conda env and install deps

```bash
conda env create -f environment.yml || conda env update -f environment.yml
conda activate loopbench
pip install -e .
```

## 4) Install and verify Docker on eval host

```bash
docker info
```

If Docker service is unavailable and you are on a constrained host, use:

```bash
scripts/with_local_docker.sh python -m loopbench.cli preflight --config configs/runtime.e2b.openrouter.yaml
```

## 5) Configure runtime for your environment

- `configs/runtime.e2b.openrouter.yaml`
  - keep `sandbox_backend.kind: e2b_firecracker`
  - for local host Docker: `judge.docker_host: null`
  - for remote Docker judge: `judge.docker_host: ssh://<user>@<host>`

## 6) Export required secrets

```bash
export E2B_API_KEY=...
export OPEN_ROUTER_API_KEY=...
```

## 7) Preflight gate (required before experiments)

```bash
python -m loopbench.cli preflight --config configs/runtime.e2b.openrouter.yaml
```

Must be true:
- `ok: true`
- `judge_docker_daemon_ready: true`
- `e2b_api_key_present: true`
- `e2b_python_sdk: true`

## 8) Resume experiments

```bash
python -m loopbench.cli experiment \
  --spec configs/experiment.kimi2_5.trial.2tasks_3rollouts.yaml
```

## Integrity notes

- Infra failures are tracked separately with `hidden_infra_error` in run manifest.
- Do not compare pass-rates from runs that failed preflight.
- Keep task packs immutable across machines to preserve comparability.

