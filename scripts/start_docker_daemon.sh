#!/usr/bin/env bash
set -euo pipefail

if docker info >/dev/null 2>&1; then
  echo "docker daemon already available"
  exit 0
fi

DOCKER_DATA_ROOT="${LOOPBENCH_DOCKER_DATA_ROOT:-/tmp/docker-data}"
DOCKER_EXEC_ROOT="${LOOPBENCH_DOCKER_EXEC_ROOT:-/tmp/docker-exec}"
DOCKER_PIDFILE="${LOOPBENCH_DOCKER_PIDFILE:-/tmp/dockerd.pid}"
DOCKER_LOG="${LOOPBENCH_DOCKER_LOG:-/tmp/dockerd.log}"
DOCKER_SOCK="${LOOPBENCH_DOCKER_SOCK:-/var/run/docker.sock}"

mkdir -p "$DOCKER_DATA_ROOT" "$DOCKER_EXEC_ROOT"

nohup dockerd \
  --data-root "$DOCKER_DATA_ROOT" \
  --exec-root "$DOCKER_EXEC_ROOT" \
  --pidfile "$DOCKER_PIDFILE" \
  --host "unix://${DOCKER_SOCK}" \
  --storage-driver=vfs \
  --iptables=false \
  --ip6tables=false \
  --bridge=none \
  --ip-forward=false \
  --ip-masq=false \
  >"$DOCKER_LOG" 2>&1 &

for _ in $(seq 1 30); do
  if docker info >/dev/null 2>&1; then
    echo "docker daemon is ready"
    exit 0
  fi
  sleep 1
done

echo "docker daemon failed to start; tailing log:"
tail -n 120 "$DOCKER_LOG" || true
exit 1
