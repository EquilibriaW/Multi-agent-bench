"""
loopbench.observability

Runtime-query backends for logs and metrics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode
import urllib.error
import urllib.request


@dataclass(frozen=True)
class ObservabilitySettings:
    logs: str = "none"
    metrics: str = "none"
    traces: str = "none"
    logs_endpoint: str | None = None
    metrics_endpoint: str | None = None
    traces_endpoint: str | None = None


class LogsBackend(Protocol):
    def query(self, query: str) -> str:
        ...


class MetricsBackend(Protocol):
    def query(self, query: str) -> str:
        ...


class ArtifactLogsBackend:
    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir).resolve()

    def query(self, query: str) -> str:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        matches = []
        for path in sorted(self.artifacts_dir.glob("*.log")):
            content = path.read_text(encoding="utf-8", errors="replace")
            if query in content:
                matches.append(f"{path.name}: query matched")
        if not matches:
            return "no matches"
        return "\n".join(matches)


class LokiLogsBackend:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def query(self, query: str) -> str:
        url = f"{self.endpoint}/loki/api/v1/query?{urlencode({'query': query, 'limit': '200'})}"
        return _http_get_json(url)


class NoopLogsBackend:
    def __init__(self, reason: str):
        self.reason = reason

    def query(self, query: str) -> str:
        return f"logs query unavailable: {self.reason}"


class PrometheusMetricsBackend:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    def query(self, query: str) -> str:
        url = f"{self.endpoint}/api/v1/query?{urlencode({'query': query})}"
        return _http_get_json(url)


class NoopMetricsBackend:
    def __init__(self, reason: str):
        self.reason = reason

    def query(self, query: str) -> str:
        payload = {"query": query, "status": "unavailable", "reason": self.reason}
        return json.dumps(payload)


@dataclass
class ObservabilityFacade:
    logs_backend: LogsBackend
    metrics_backend: MetricsBackend

    def logs_query(self, query: str) -> str:
        return self.logs_backend.query(query)

    def metrics_query(self, query: str) -> str:
        return self.metrics_backend.query(query)


def build_observability(
    *,
    settings: ObservabilitySettings,
    artifacts_dir: str | Path,
) -> ObservabilityFacade:
    logs_kind = settings.logs.lower().strip()
    metrics_kind = settings.metrics.lower().strip()

    logs_backend = _build_logs_backend(
        logs_kind=logs_kind,
        logs_endpoint=settings.logs_endpoint,
        artifacts_dir=artifacts_dir,
    )
    metrics_backend = _build_metrics_backend(
        metrics_kind=metrics_kind,
        metrics_endpoint=settings.metrics_endpoint,
    )
    return ObservabilityFacade(logs_backend=logs_backend, metrics_backend=metrics_backend)


def _build_logs_backend(*, logs_kind: str, logs_endpoint: str | None, artifacts_dir: str | Path) -> LogsBackend:
    if logs_kind in {"none", ""}:
        return ArtifactLogsBackend(artifacts_dir=artifacts_dir)
    if logs_kind == "loki":
        if logs_endpoint:
            return LokiLogsBackend(endpoint=logs_endpoint)
        return ArtifactLogsBackend(artifacts_dir=artifacts_dir)
    return NoopLogsBackend(reason=f"unsupported logs backend '{logs_kind}'")


def _build_metrics_backend(*, metrics_kind: str, metrics_endpoint: str | None) -> MetricsBackend:
    if metrics_kind in {"none", ""}:
        return NoopMetricsBackend(reason="metrics backend disabled")
    if metrics_kind == "prometheus":
        if metrics_endpoint:
            return PrometheusMetricsBackend(endpoint=metrics_endpoint)
        return NoopMetricsBackend(reason="prometheus selected but metrics_endpoint is empty")
    return NoopMetricsBackend(reason=f"unsupported metrics backend '{metrics_kind}'")


def _http_get_json(url: str) -> str:
    req = urllib.request.Request(url=url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            return json.dumps(parsed)
    except urllib.error.HTTPError as exc:
        return json.dumps(
            {
                "status": "http_error",
                "code": exc.code,
                "body": exc.read().decode("utf-8", errors="replace"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"status": "error", "error": str(exc)})
