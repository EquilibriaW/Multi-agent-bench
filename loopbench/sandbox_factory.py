"""
loopbench.sandbox_factory

Select sandbox backend implementation per runtime config.
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import RuntimeConfig
from .e2b_sandbox import E2BFirecrackerSandbox, E2BOptions
from .sandbox import LocalSandbox


def build_sandbox(
    *,
    runtime_cfg: RuntimeConfig,
    role: str,
    worktree_root: str | Path,
):
    kind = runtime_cfg.sandbox_backend.kind

    # Local execution path for currently supported local backends.
    if kind in {"docker_container", "local_process", "kata_container", "firecracker_microvm", "unikernel"}:
        return LocalSandbox(sandbox_name=role, root=worktree_root)

    if kind == "e2b_firecracker":
        e2b_cfg = runtime_cfg.sandbox_backend.e2b
        api_key = os.environ.get(e2b_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"sandbox backend kind=e2b_firecracker requires env var {e2b_cfg.api_key_env}"
            )

        options = E2BOptions(
            api_key=api_key,
            template=e2b_cfg.template,
            timeout_sec=e2b_cfg.timeout_sec,
            allow_internet_access=e2b_cfg.allow_internet_access,
        )
        return E2BFirecrackerSandbox(sandbox_name=role, root=worktree_root, options=options)

    raise ValueError(f"Unsupported sandbox backend kind: {kind}")
