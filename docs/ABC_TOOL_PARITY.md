# ABC-Bench Tool Parity (Implementation Notes)

This note maps ABC-Bench's published implementation details to the minimum tool
surface we should provide to team agents in LoopBench.

## What ABC-Bench actually gives agents

Source inspected: Terminal-Bench implementation used by ABC-Bench quickstart.

- OpenHands runs in a **task-local runtime** with workspace mounted rw:
  - `terminal_bench/agents/installed_agents/openhands/openhands_agent.py`
    sets `RUNTIME=local` and `SANDBOX_VOLUMES=${PWD}:/workspace:rw`.
- Browser capability is explicitly disabled:
  - `AGENT_ENABLE_BROWSING=false`, `ENABLE_BROWSER=false`.
- Task container is launched as **privileged**:
  - task `docker-compose.yaml` has `privileged: true`.
- Client image is Docker-in-Docker style:
  - task `Dockerfile` starts from `cruizba/ubuntu-dind`.
- Hidden tests in ABC tasks call Docker directly (`docker build/run/logs/inspect`)
  and install runtime test deps with `apt-get`, `curl`, `uv`, `pytest`.

## Practical requirement for LoopBench agent tools

Agents need unrestricted terminal workflows for:

1. Repo work: `git`, file reads/writes, search (`rg`/`grep`), edits.
2. Runtime build/deploy: `docker`, `docker compose`.
3. Env bootstrapping: `apt-get`, `curl`/`wget`, `uv`, language package managers.
4. Verification/debugging: `pytest` (or framework equivalent), logs, HTTP checks.

Browser automation is optional for ABC-Bench parity and can stay disabled.

## What we changed

- `scripts/openrouter_role_driver.py` no longer imposes command allow/block lists.
- Command execution policy is now budget-only:
  - `LOOPBENCH_MAX_COMMANDS`
  - `LOOPBENCH_COMMAND_TIMEOUT_SEC`
  with backend-aware defaults (`e2b_firecracker`: larger budgets).
- Safety is delegated to sandbox isolation (E2B/Firecracker), matching the
  experiment assumption that agent code runs only inside sandboxes.
