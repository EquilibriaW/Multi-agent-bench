"""
loopbench.tracing

Optional run-level trace sessions for orchestration visibility.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .observability import ObservabilitySettings

_DEFAULT_EVENT_TYPES = {
    "run_started",
    "role_phase",
    "run_finished",
    "tool_call",
    "tool_result",
    "tool_denied",
    "merge_commit",
    "merge_commit_failed",
}
_MAX_EVENT_PAYLOAD_CHARS = 16000
_MAX_ERROR_CHARS = 1200
_MAX_PROMPT_PREVIEW_CHARS = 24000
_MAX_RESPONSE_PREVIEW_CHARS = 24000
_MAX_CONTEXT_PREVIEW_CHARS = 12000


class NoopTraceSession:
    def __init__(self, *, reason: str, backend: str = "none"):
        self.enabled = False
        self.backend = backend
        self._reason = reason

    def emit_event(self, event: Dict[str, Any], *, span_id: Optional[str] = None) -> None:
        _ = event

    def begin_span(
        self,
        span_id: str,
        *,
        parent_span_id: Optional[str] = None,
        name: str = "",
        run_type: str = "chain",
        inputs: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        pass

    def end_span(self, span_id: str, *, outputs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def finish(self, *, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        _ = outputs
        _ = error

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "backend": self.backend,
            "reason": self._reason,
            "trace_url": None,
            "event_types": [],
            "runtime_error": None,
        }


class LangSmithTraceSession:
    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        task_kind: str,
        roles: list[str],
        traces_endpoint: Optional[str],
        run_dir: str | Path | None,
    ):
        from langsmith import Client
        from langsmith.run_trees import RunTree

        self.enabled = True
        self.backend = "langsmith"
        self._lock = threading.Lock()
        self._finished = False
        self._drop_events = False
        self._runtime_error: Optional[str] = None
        self._trace_url: Optional[str] = None
        self._run_dir = Path(run_dir).resolve() if run_dir else None

        self._event_types = _resolve_event_types()

        client_kwargs: Dict[str, Any] = {}
        api_key = _langsmith_api_key()
        if api_key:
            client_kwargs["api_key"] = api_key
        if traces_endpoint and traces_endpoint.strip():
            client_kwargs["api_url"] = traces_endpoint.strip().rstrip("/")

        workspace_id = (os.environ.get("LANGSMITH_WORKSPACE_ID") or "").strip()
        if workspace_id:
            client_kwargs["workspace_id"] = workspace_id

        self._client = Client(**client_kwargs)
        self._project_name = _langsmith_project_name()
        self._root = RunTree(
            name=f"loopbench.run.{run_id}",
            run_type="chain",
            project_name=self._project_name,
            inputs={
                "run_id": run_id,
                "task_id": task_id,
                "task_kind": task_kind,
                "roles": roles,
                "run_dir": str(self._run_dir) if self._run_dir else None,
            },
            client=self._client,
            tags=["loopbench", "agent-team", "multi-agent"],
        )
        self._root.add_metadata(
            {
                "loopbench_run_id": run_id,
                "loopbench_task_id": task_id,
                "loopbench_task_kind": task_kind,
                "loopbench_roles": list(roles),
                "loopbench_run_dir": str(self._run_dir) if self._run_dir else None,
            }
        )
        self._root.post()
        self._spans: Dict[str, Any] = {}  # span_id -> RunTree

    def begin_span(
        self,
        span_id: str,
        *,
        parent_span_id: Optional[str] = None,
        name: str = "",
        run_type: str = "chain",
        inputs: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            if self._finished or self._drop_events:
                return
            try:
                parent = self._spans.get(parent_span_id) if parent_span_id else self._root
                if parent is None:
                    parent = self._root
                child = parent.create_child(
                    name=name or span_id,
                    run_type=run_type,
                    inputs=inputs or {},
                    tags=tags or ["loopbench_span"],
                )
                child.post()
                self._spans[span_id] = child
            except Exception as exc:  # noqa: BLE001
                self._runtime_error = _truncate_text(str(exc), _MAX_ERROR_CHARS)

    def end_span(self, span_id: str, *, outputs: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if self._finished:
                return
            try:
                run_tree = self._spans.pop(span_id, None)
                if run_tree is not None:
                    run_tree.end(outputs=outputs or {})
                    run_tree.patch()
            except Exception as exc:  # noqa: BLE001
                self._runtime_error = _truncate_text(str(exc), _MAX_ERROR_CHARS)

    def emit_event(self, event: Dict[str, Any], *, span_id: Optional[str] = None) -> None:
        with self._lock:
            if self._finished or self._drop_events:
                return

            event_type = str(event.get("type") or "unknown")
            if self._event_types is not None and event_type not in self._event_types:
                return

            try:
                parent = self._spans.get(span_id) if span_id else self._root
                if parent is None:
                    parent = self._root
                payload = event.get("payload")
                child = parent.create_child(
                    name=f"event.{event_type}",
                    run_type=_event_run_type(event_type),
                    inputs={
                        "event_type": event_type,
                        "ts_ms": event.get("ts_ms"),
                        "payload_json": _truncate_text(_to_json(payload), _MAX_EVENT_PAYLOAD_CHARS),
                    },
                    tags=["loopbench_event", event_type],
                )
                child.post()
                child.end(outputs={"recorded": True})
                child.patch()

                if event_type == "role_phase":
                    self._emit_role_phase_llm_usage(payload, span_id=span_id)
            except Exception as exc:  # noqa: BLE001
                self._runtime_error = _truncate_text(str(exc), _MAX_ERROR_CHARS)
                self._drop_events = True

    def finish(self, *, outputs: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True

            try:
                # Auto-close any unclosed spans before finishing root.
                for sid in list(self._spans.keys()):
                    try:
                        run_tree = self._spans.pop(sid)
                        run_tree.end(outputs={"auto_closed": True})
                        run_tree.patch()
                    except Exception:  # noqa: BLE001
                        pass

                self._root.end(
                    outputs=outputs or {},
                    error=_truncate_text(error, _MAX_ERROR_CHARS) if error else None,
                )
                self._root.patch()
                self._trace_url = self._safe_trace_url()
            except Exception as exc:  # noqa: BLE001
                self._runtime_error = _truncate_text(str(exc), _MAX_ERROR_CHARS)

    def snapshot(self) -> Dict[str, Any]:
        event_types = sorted(self._event_types) if self._event_types is not None else ["*"]
        return {
            "enabled": True,
            "backend": self.backend,
            "reason": None,
            "project": self._project_name,
            "trace_url": self._trace_url,
            "event_types": event_types,
            "runtime_error": self._runtime_error,
        }

    def _safe_trace_url(self) -> Optional[str]:
        try:
            return self._root.get_url()
        except Exception:  # noqa: BLE001
            return None

    def _emit_role_phase_llm_usage(self, payload: Any, *, span_id: Optional[str] = None) -> None:
        usage_payload = _extract_role_phase_usage(payload)
        if usage_payload is None:
            return

        usage = usage_payload["usage"]
        llm_inputs = {
            "role": usage_payload["role"],
            "phase": usage_payload["phase"],
            "model": usage_payload["model"],
            "attempt_count": usage_payload["attempt_count"],
        }
        llm_outputs = {
            "structured_valid": usage_payload["structured_valid"],
            "raw_chars": usage_payload["raw_chars"],
        }

        role_io = _extract_role_phase_io(payload=payload, run_dir=self._run_dir)
        if role_io:
            llm_inputs.update(role_io.get("inputs") or {})
            llm_outputs.update(role_io.get("outputs") or {})

        parent = self._spans.get(span_id) if span_id else self._root
        if parent is None:
            parent = self._root
        llm_child = parent.create_child(
            name=f"llm.{usage_payload['role']}.{usage_payload['phase']}",
            run_type="llm",
            inputs=llm_inputs,
            outputs=llm_outputs,
            tags=["loopbench_llm", usage_payload["role"], usage_payload["phase"]],
        )
        llm_child.post()
        llm_child.set(usage_metadata=usage)
        llm_child.end(outputs=llm_child.outputs or {})
        llm_child.patch()


def build_trace_session(
    *,
    settings: ObservabilitySettings,
    run_id: str,
    task_id: str,
    task_kind: str,
    roles: list[str],
    run_dir: str | Path | None = None,
) -> NoopTraceSession | LangSmithTraceSession:
    traces_kind = (settings.traces or "none").strip().lower()
    if traces_kind in {"none", ""}:
        return NoopTraceSession(reason="traces backend disabled", backend="none")

    if traces_kind != "langsmith":
        return NoopTraceSession(
            reason=f"unsupported traces backend '{traces_kind}'",
            backend=traces_kind,
        )

    if not _langsmith_api_key():
        return NoopTraceSession(
            reason="LANGSMITH_API_KEY (or LANGCHAIN_API_KEY) is required for traces=langsmith",
            backend="langsmith",
        )

    try:
        return LangSmithTraceSession(
            run_id=run_id,
            task_id=task_id,
            task_kind=task_kind,
            roles=roles,
            traces_endpoint=settings.traces_endpoint,
            run_dir=run_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return NoopTraceSession(
            reason=f"langsmith tracing initialization failed: {_truncate_text(str(exc), _MAX_ERROR_CHARS)}",
            backend="langsmith",
        )


def write_trace_snapshot(*, run_dir: str | Path, snapshot: Dict[str, Any]) -> Path:
    out_path = Path(run_dir).resolve() / "trace" / "session.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return out_path


def _langsmith_api_key() -> str:
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _langsmith_project_name() -> str:
    for name in ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return "loopbench"


def _resolve_event_types() -> Optional[Set[str]]:
    raw = (os.environ.get("LOOPBENCH_LANGSMITH_EVENT_TYPES") or "").strip()
    if raw:
        if raw.lower() in {"all", "*"}:
            return None
        return {token.strip() for token in raw.split(",") if token.strip()}

    event_types = set(_DEFAULT_EVENT_TYPES)
    if _read_bool_env("LOOPBENCH_LANGSMITH_INCLUDE_TOOL_EVENTS", default=False):
        event_types.update({"tool_call", "tool_result"})
    return event_types


def _read_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _event_run_type(event_type: str) -> str:
    if event_type in {"tool_call", "tool_result", "tool_denied"}:
        return "tool"
    return "chain"


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return json.dumps(repr(payload), ensure_ascii=False)


def _truncate_text(value: Optional[str], max_chars: int) -> str:
    if not value:
        return ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def _extract_role_phase_usage(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if not isinstance(output, dict):
        return None

    usage = _normalize_usage_metadata(output.get("openrouter_usage"))
    if usage is None:
        return None

    role = str(payload.get("role") or "unknown")
    phase = str(payload.get("phase") or "unknown")
    model = output.get("model")
    model_name = model if isinstance(model, str) and model.strip() else "unknown"
    attempt_count = _read_non_negative_int(output.get("openrouter_attempt_count"))
    raw_chars = _read_non_negative_int(output.get("openrouter_raw_chars"))
    structured_valid = bool(output.get("openrouter_structured_valid"))

    return {
        "role": role,
        "phase": phase,
        "model": model_name,
        "attempt_count": attempt_count,
        "raw_chars": raw_chars,
        "structured_valid": structured_valid,
        "usage": usage,
    }


def _extract_role_phase_io(payload: Any, run_dir: Path | None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    output = payload.get("output")
    if not isinstance(output, dict):
        return {}

    out: Dict[str, Any] = {"inputs": {}, "outputs": {}}
    artifact_paths = output.get("artifact_paths")
    if isinstance(artifact_paths, dict) and artifact_paths:
        out["inputs"]["artifact_paths"] = artifact_paths
        out["outputs"]["artifact_paths"] = artifact_paths

    if run_dir is None or not isinstance(artifact_paths, dict):
        return out

    request_path = _find_artifact_path(artifact_paths, "_openrouter_request.json")
    response_path = _find_artifact_path(artifact_paths, "_openrouter_response.txt")
    attempts_path = _find_artifact_path(artifact_paths, "_openrouter_attempts.json")
    context_path = _find_artifact_path(artifact_paths, "_context.json")

    if request_path:
        request_json = _load_json_artifact(run_dir=run_dir, artifact_path=request_path)
        if isinstance(request_json, dict):
            out["inputs"]["openrouter_request_preview"] = _truncate_text(
                _to_json(request_json),
                _MAX_PROMPT_PREVIEW_CHARS,
            )
            message_content = _extract_chat_messages(request_json)
            if message_content:
                out["inputs"].update(message_content)

    if context_path:
        context_json = _load_json_artifact(run_dir=run_dir, artifact_path=context_path)
        if isinstance(context_json, dict):
            out["inputs"]["role_context_preview"] = _truncate_text(
                _to_json(context_json),
                _MAX_CONTEXT_PREVIEW_CHARS,
            )

    if response_path:
        response_text = _load_text_artifact(run_dir=run_dir, artifact_path=response_path)
        if response_text:
            out["outputs"]["openrouter_response_preview"] = _truncate_text(
                response_text,
                _MAX_RESPONSE_PREVIEW_CHARS,
            )

    if attempts_path:
        attempts_json = _load_json_artifact(run_dir=run_dir, artifact_path=attempts_path)
        if isinstance(attempts_json, dict):
            out["outputs"]["openrouter_attempts_preview"] = _truncate_text(
                _to_json(attempts_json),
                _MAX_CONTEXT_PREVIEW_CHARS,
            )

    applied_paths = output.get("applied_paths")
    if isinstance(applied_paths, list):
        out["outputs"]["applied_paths"] = [str(x) for x in applied_paths[:50]]
    commands_run = output.get("commands_run")
    if isinstance(commands_run, list):
        out["outputs"]["commands_run"] = commands_run[:20]
    summary = output.get("summary")
    if isinstance(summary, str) and summary.strip():
        out["outputs"]["summary"] = summary.strip()
    notes = output.get("notes")
    if isinstance(notes, str) and notes.strip():
        out["outputs"]["notes"] = notes.strip()

    return out


def _extract_chat_messages(payload: Dict[str, Any]) -> Dict[str, str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}

    system_prompt = ""
    user_prompt = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = message.get("content")
        if isinstance(content, str):
            value = content
        else:
            value = _to_json(content)
        if role == "system" and not system_prompt:
            system_prompt = _truncate_text(value, _MAX_PROMPT_PREVIEW_CHARS)
        if role == "user" and not user_prompt:
            user_prompt = _truncate_text(value, _MAX_PROMPT_PREVIEW_CHARS)
        if system_prompt and user_prompt:
            break

    out: Dict[str, str] = {}
    if system_prompt:
        out["system_prompt"] = system_prompt
    if user_prompt:
        out["user_prompt"] = user_prompt
    return out


def _find_artifact_path(artifact_paths: Dict[str, Any], suffix: str) -> Optional[str]:
    for key, value in artifact_paths.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if key.endswith(suffix):
            return value
    return None


def _load_json_artifact(*, run_dir: Path, artifact_path: str) -> Optional[Dict[str, Any]]:
    path = _resolve_artifact_path(run_dir=run_dir, artifact_path=artifact_path)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return data


def _load_text_artifact(*, run_dir: Path, artifact_path: str) -> Optional[str]:
    path = _resolve_artifact_path(run_dir=run_dir, artifact_path=artifact_path)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _resolve_artifact_path(*, run_dir: Path, artifact_path: str) -> Optional[Path]:
    path = Path(artifact_path)
    if not path.is_absolute():
        path = (run_dir / path).resolve()
    else:
        path = path.resolve()
    if path == run_dir or run_dir in path.parents:
        return path
    return None


def _normalize_usage_metadata(raw: Any) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None

    input_tokens = _read_non_negative_int(raw.get("input_tokens"), raw.get("prompt_tokens"))
    output_tokens = _read_non_negative_int(raw.get("output_tokens"), raw.get("completion_tokens"))
    total_tokens = _read_non_negative_int(raw.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _read_non_negative_int(*values: Any) -> int:
    for value in values:
        try:
            parsed = int(value)
        except Exception:  # noqa: BLE001
            continue
        if parsed >= 0:
            return parsed
    return 0
