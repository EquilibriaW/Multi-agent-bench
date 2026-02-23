"""
loopbench.path_utils

Path safety helpers for sandbox/workspace boundaries.
"""
from __future__ import annotations

from pathlib import Path


def resolve_within_root(*, root: str | Path, raw_path: str) -> Path:
    root_path = Path(root).resolve()
    path = Path(raw_path)
    resolved = (root_path / path).resolve() if not path.is_absolute() else path.resolve()
    if root_path not in resolved.parents and resolved != root_path:
        raise ValueError(f"path escapes sandbox root: {raw_path}")
    return resolved


def safe_path_component(raw: str | None, *, max_len: int = 120) -> str:
    if not raw:
        return ""
    chars = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)[:max_len]
