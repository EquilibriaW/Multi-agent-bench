# Security & Leakage Controls

LoopBench must preserve evaluation validity. The main risk is **hidden test leakage**.

## 1) Threat model

Assume the agent can:
- execute arbitrary code in its sandbox
- attempt to infer hidden tests via probing
- attempt to exfiltrate files via network (if enabled)
- attempt to access host resources if sandbox is privileged

## 2) Primary controls

### 2.1 Hard separation: agent sandbox vs judge sandbox
- agent sandbox: public only
- judge sandbox: hidden only
- judge runs after agent completes
- judge always evaluates from a fresh materialized repo + applied `final.patch`
  (never from mutable agent worktrees), to keep runs comparable.

### 2.2 Minimal privileges
- agent sandbox has no Docker socket
- agent sandbox runs unprivileged (no `--privileged`)
- default no-network for agent sandbox
- judge may require Docker runtime for hidden tests; this privilege is granted to
  the judge runtime only, never to agent sandboxes.
- for nested hosts, prefer a dedicated remote judge Docker endpoint (`judge.docker_host`)
  so model outcomes are not confounded by local daemon instability.

### 2.3 Deterministic tool routing
- if a capability is needed, expose it as a tool (http/logs/metrics)
- tools are budgeted and logged

## 3) Secondary controls

- deny access to host-mounted secrets (no home dir mounts)
- filter environment variables in sandbox
- run tasks with explicit allowlists for filesystem paths
- optionally run agent sandboxes inside stronger isolation backends:
  - Kata (VM-backed containers)
  - Firecracker microVM (strong isolation)
  - research: unikernel / UKL-style approaches

## 4) Unikernel note (research)

Unikernels are attractive for agent sandboxes:
- minimal attack surface
- very fast boot
But Python and multi-process support are constraints, and image build/distribution efficiency
often dominates latency.

LoopBench treats unikernel sandboxes as an optional backend behind the same ToolRouter interface.
