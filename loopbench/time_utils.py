"""
loopbench.time_utils

Small time helpers shared across runtime components.
"""
from __future__ import annotations

import time


def now_ms() -> int:
    return int(time.time() * 1000)
