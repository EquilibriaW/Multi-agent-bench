#!/usr/bin/env bash
set -euo pipefail

# ---------- preflight: require all 3 keys ----------
fail=0
for var in OPEN_ROUTER_API_KEY E2B_API_KEY LANGSMITH_API_KEY; do
    val=$(eval echo "\${${var}:-}")
    if [ -z "$val" ]; then
        echo "FATAL: $var is not set" >&2
        fail=1
    else
        echo "  $var: set (${#val} chars)" >&2
    fi
done
[ "$fail" -eq 1 ] && exit 1

# ---------- build experiment spec with unique id ----------
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EXP_ID="kimi2_5-scale-check-10x1-${STAMP}"
SPEC_FILE="/tmp/experiment_${EXP_ID}.yaml"

sed "s/^experiment_id:.*/experiment_id: ${EXP_ID}/" \
    configs/experiment.kimi2_5.scale_check.10x1.20260222.yaml \
    > "$SPEC_FILE"

echo "[loopbench] experiment_id: ${EXP_ID}" >&2
echo "[loopbench] spec: ${SPEC_FILE}" >&2
echo "[loopbench] log: experiments/${EXP_ID}/run.log" >&2

# ---------- launch in background ----------
nohup loopbench experiment --spec "$SPEC_FILE" \
    > "experiments_${EXP_ID}.log" 2>&1 &

PID=$!
echo "[loopbench] started as PID ${PID}" >&2
echo "[loopbench] tail -f experiments_${EXP_ID}.log" >&2
echo "$PID"
