"""
loopbench.events

Append-only JSONL event writer for deterministic replay.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Protocol

from .time_utils import now_ms


class EventLogger:
    def __init__(self, path: str | Path, sinks: List["EventSink"] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sinks = list(sinks or [])

    def log(self, event_type: str, payload: Dict[str, Any], *, span_id: str | None = None) -> None:
        event: Dict[str, Any] = {
            "ts_ms": now_ms(),
            "type": event_type,
            "payload": payload,
        }
        if span_id is not None:
            event["span_id"] = span_id
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=True) + "\n")
            sinks_snapshot = list(self._sinks)
        for sink in sinks_snapshot:
            try:
                sink.emit_event(event, span_id=span_id)
            except Exception:  # noqa: BLE001
                # Never allow observability sinks to change benchmark behavior.
                continue

    def begin_span(
        self,
        span_id: str,
        *,
        parent_span_id: str | None = None,
        name: str,
        run_type: str = "chain",
        inputs: Dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        event = {
            "ts_ms": now_ms(),
            "type": "span_begin",
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": name,
            "run_type": run_type,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=True) + "\n")
            sinks_snapshot = list(self._sinks)
        for sink in sinks_snapshot:
            try:
                sink.begin_span(
                    span_id,
                    parent_span_id=parent_span_id,
                    name=name,
                    run_type=run_type,
                    inputs=inputs,
                    tags=tags,
                )
            except Exception:  # noqa: BLE001
                continue

    def end_span(
        self,
        span_id: str,
        *,
        outputs: Dict[str, Any] | None = None,
    ) -> None:
        event = {
            "ts_ms": now_ms(),
            "type": "span_end",
            "span_id": span_id,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=True) + "\n")
            sinks_snapshot = list(self._sinks)
        for sink in sinks_snapshot:
            try:
                sink.end_span(span_id, outputs=outputs)
            except Exception:  # noqa: BLE001
                continue


class EventSink(Protocol):
    def emit_event(self, event: Dict[str, Any], *, span_id: str | None = None) -> None:
        ...

    def begin_span(
        self,
        span_id: str,
        *,
        parent_span_id: str | None = None,
        name: str,
        run_type: str = "chain",
        inputs: Dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        ...

    def end_span(self, span_id: str, *, outputs: Dict[str, Any] | None = None) -> None:
        ...
