#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REGISTRY:-ghcr.io/loopbench}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-0}"

build_one() {
  local name="$1"
  local dockerfile="$2"
  local image="${REGISTRY}/${name}:${TAG}"

  echo "building ${image} from ${dockerfile}"
  docker build \
    --file "${dockerfile}" \
    --tag "${image}" \
    "${ROOT_DIR}"

  if [[ "${PUSH}" == "1" ]]; then
    echo "pushing ${image}"
    docker push "${image}"
  fi
}

build_one "agent-sandbox" "${ROOT_DIR}/containers/agent-sandbox.Dockerfile"
build_one "env-runner" "${ROOT_DIR}/containers/env-runner.Dockerfile"
build_one "judge" "${ROOT_DIR}/containers/judge.Dockerfile"

echo "done"
