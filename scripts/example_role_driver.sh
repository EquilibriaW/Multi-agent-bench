#!/usr/bin/env bash
set -euo pipefail

# Expected env from harness:
# - LB_ROLE
# - LB_PHASE
# - LB_WORKTREE
# - LB_CONTEXT_JSON
# - LB_OUTPUT_JSON
# - LB_COORD_DB (optional, sqlite coordination DB path)
#
# User must provide LOOPBENCH_AGENT_CMD to call a real agent runtime.

if [[ -z "${LB_OUTPUT_JSON:-}" ]]; then
  echo "LB_OUTPUT_JSON is required" >&2
  exit 2
fi

if [[ -z "${LOOPBENCH_AGENT_CMD:-}" ]]; then
  cat >"$LB_OUTPUT_JSON" <<JSON
{
  "status": "missing_agent_command",
  "message": "Set LOOPBENCH_AGENT_CMD to invoke your real role runtime.",
  "role": "${LB_ROLE:-unknown}",
  "phase": "${LB_PHASE:-unknown}"
}
JSON
  echo "LOOPBENCH_AGENT_CMD is not set. Refusing to run a fake role." >&2
  exit 2
fi

# Execute user-provided runtime command in the role worktree.
(
  cd "${LB_WORKTREE:?LB_WORKTREE is required}"
  bash -lc "$LOOPBENCH_AGENT_CMD"
)

# Ensure harness receives structured output even when agent runtime does not emit one.
if [[ ! -f "$LB_OUTPUT_JSON" ]]; then
  cat >"$LB_OUTPUT_JSON" <<JSON
{
  "status": "completed",
  "role": "${LB_ROLE:-unknown}",
  "phase": "${LB_PHASE:-unknown}",
  "note": "Agent runtime completed but did not write LB_OUTPUT_JSON."
}
JSON
fi
