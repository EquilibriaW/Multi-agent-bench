#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "conda not found; set CONDA_BIN or put conda on PATH" >&2
  exit 1
fi

ENV_NAME="${LOOPBENCH_CONDA_ENV:-loopbench}"

exec "$CONDA_BIN" run -n "$ENV_NAME" "$@"
