"""
loopbench.io_utils

File format loaders with validation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_yaml_mapping(path: str | Path, *, label: str) -> Dict[str, Any]:
    yaml_path = Path(path)
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} at {yaml_path} must contain a mapping")
    return data
