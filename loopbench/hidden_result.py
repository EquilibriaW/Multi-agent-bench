"""
loopbench.hidden_result

Helpers for extracting concise hidden-judge failure signals from run artifacts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def hidden_result_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir).resolve() / "hidden_validate" / "result.log"


def hidden_stderr_excerpt(run_dir: str | Path) -> Optional[str]:
    log_path = hidden_result_log_path(run_dir)
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = "\n\nSTDERR\n"
    idx = text.find(marker)
    if idx == -1:
        return normalize_error(text)
    stderr = text[idx + len(marker):].strip()
    return normalize_error(_pick_meaningful_line(stderr))


def hidden_failure_reason(run_dir: str | Path) -> Optional[str]:
    log_path = hidden_result_log_path(run_dir)
    if not log_path.exists():
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.startswith("FAILED ") or candidate.startswith("ERROR "):
            return normalize_error(candidate)
    return hidden_stderr_excerpt(run_dir) or normalize_error(text)


def normalize_error(text: Optional[str]) -> str:
    if not text:
        return "unknown"
    line = text.strip().splitlines()[0]
    line = re.sub(r"\s+", " ", line)
    return line[:220]


def _pick_meaningful_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    signal_patterns = (
        "FAILED ",
        "ERROR ",
        "Traceback",
        "AssertionError",
        "unable to prepare context:",
        "No such file or directory",
        "ConnectionError",
        "ModuleNotFoundError",
        "RuntimeError",
    )
    noise_prefixes = (
        "DEPRECATED:",
        "Install the buildx component",
        "https://docs.docker.com/go/buildx/",
    )

    for line in lines:
        if any(token in line for token in signal_patterns):
            return line

    for line in lines:
        if line.startswith(noise_prefixes):
            continue
        return line

    return lines[0]
