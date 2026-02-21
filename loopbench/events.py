"""
loopbench.events

Append-only JSONL event writer for deterministic replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict

from .time_utils import now_ms


class EventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts_ms": now_ms(),
            "type": event_type,
            "payload": payload,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=True) + "\n")
