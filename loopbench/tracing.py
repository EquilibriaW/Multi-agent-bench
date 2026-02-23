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
_MAX_COMMUNICATION_ITEMS = 12
_MAX_COMMUNICATION_ITEM_CHARS = 700
_MAX_REVIEW_DIFF_VIEW_CHARS = 1800
_MAX_REWRITE_PREVIEW_CHARS = 6000


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
                    inputs=_event_inputs(event_type=event_type, ts_ms=event.get("ts_ms"), payload=payload),
                    tags=["loopbench_event", event_type],
                )
                child.post()
                event_outputs: Dict[str, Any] = {"recorded": True}
                if event_type == "role_phase" and isinstance(payload, dict):
                    event_outputs["role"] = payload.get("role")
                    event_outputs["phase"] = payload.get("phase")
                    event_outputs["ok"] = payload.get("ok")
                child.end(outputs=event_outputs)
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
        role_io = _extract_role_phase_io(payload=payload, run_dir=self._run_dir)

        io_inputs = role_io.get("inputs", {}) if role_io else {}
        io_outputs = role_io.get("outputs", {}) if role_io else {}

        # Use full messages array from OpenRouter request for LangSmith chat view
        messages = io_inputs.get("messages", [])
        llm_inputs: Dict[str, Any] = {"messages": messages}
        for key in ("coordination_messages_preview", "review_diff_tool_preview"):
            if key in io_inputs:
                llm_inputs[key] = io_inputs[key]

        # Build OpenAI-compatible choices array for LangSmith chat view
        response_text = io_outputs.get("openrouter_response_preview", "")
        llm_outputs: Dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    }
                }
            ]
        }
        for key in ("summary", "notes", "rewrite_preview", "review_diff_views_preview"):
            if key in io_outputs:
                llm_outputs[key] = io_outputs[key]

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

        # Move all debug/context data into metadata so chat view stays clean
        span_metadata: Dict[str, Any] = {
            "ls_provider": "openrouter",
            "ls_model_name": usage_payload["model"],
            "role": usage_payload["role"],
            "phase": usage_payload["phase"],
            "model": usage_payload["model"],
            "attempt_count": usage_payload["attempt_count"],
            "structured_valid": usage_payload["structured_valid"],
            "raw_chars": usage_payload["raw_chars"],
        }
        for key in (
            "artifact_paths",
            "openrouter_request_preview",
            "role_context_preview",
            "coordination_messages_preview",
            "review_diff_tool_preview",
        ):
            if key in io_inputs:
                span_metadata[key] = io_inputs[key]
        for key in (
            "artifact_paths",
            "openrouter_attempts_preview",
            "applied_paths",
            "commands_run",
            "summary",
            "notes",
            "rewrite_preview",
            "review_diff_views_preview",
        ):
            if key in io_outputs:
                span_metadata[key] = io_outputs[key]

        llm_child.add_metadata(span_metadata)
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


def _event_inputs(*, event_type: str, ts_ms: Any, payload: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"event_type": event_type, "ts_ms": ts_ms}
    payload_json = _to_json(payload)
    if len(payload_json) <= _MAX_EVENT_PAYLOAD_CHARS:
        try:
            out["payload"] = json.loads(payload_json)
            return out
        except Exception:  # noqa: BLE001
            pass
    out["payload_json"] = _truncate_text(payload_json, _MAX_EVENT_PAYLOAD_CHARS)
    return out


def _to_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return json.dumps(repr(payload), ensure_ascii=False)


def _to_pretty_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    except Exception:  # noqa: BLE001
        return _to_json(payload)


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
        usage = _normalize_usage_metadata(output.get("usage"))
    if usage is None:
        usage = _zero_usage_metadata()

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
            coordination_messages = _summarize_coordination_messages(context_json.get("implementation_messages"))
            if coordination_messages:
                out["inputs"]["coordination_messages_preview"] = coordination_messages
            review_diff_tool_preview = _summarize_review_diff_tool(context_json.get("review_diff_tool"))
            if review_diff_tool_preview:
                out["inputs"]["review_diff_tool_preview"] = review_diff_tool_preview

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
        review_diff_views = _summarize_review_diff_views(commands_run)
        if review_diff_views:
            out["outputs"]["review_diff_views_preview"] = review_diff_views
    rewrite_preview = _extract_rewrite_preview(output)
    if rewrite_preview:
        out["outputs"]["rewrite_preview"] = rewrite_preview
    summary = output.get("summary")
    if isinstance(summary, str) and summary.strip():
        out["outputs"]["summary"] = summary.strip()
    notes = output.get("notes")
    if isinstance(notes, str) and notes.strip():
        out["outputs"]["notes"] = notes.strip()

    return out


def _extract_chat_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract all chat messages from an OpenRouter request payload.

    Returns ``{"messages": [{"role": ..., "content": ...}, ...]}`` with each
    message's content individually truncated.  An empty dict is returned when
    no messages are found.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}

    out_messages: list[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        if not role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            value = content
        else:
            value = _to_pretty_json(content)
        out_messages.append({
            "role": role,
            "content": _truncate_text(value, _MAX_PROMPT_PREVIEW_CHARS),
        })

    if not out_messages:
        return {}
    return {"messages": out_messages}


def _summarize_coordination_messages(raw_messages: Any) -> list[str]:
    if not isinstance(raw_messages, list):
        return []
    lines: list[str] = []
    for message in raw_messages[-_MAX_COMMUNICATION_ITEMS:]:
        if not isinstance(message, dict):
            continue
        from_role = str(message.get("from_role") or "unknown")
        to_role = str(message.get("to_role") or "*")
        kind = str(message.get("kind") or "message")
        body_preview = _message_body_preview(message.get("body"))
        line = f"{from_role} -> {to_role} [{kind}]"
        if body_preview:
            line = f"{line}: {body_preview}"
        lines.append(_truncate_text(line, _MAX_COMMUNICATION_ITEM_CHARS))
    return lines


def _message_body_preview(body: Any) -> str:
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, dict):
        for key in (
            "summary",
            "planner_feedback",
            "error",
            "message",
            "status",
            "title",
            "task_id",
            "reason",
        ):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return f"{key}={value.strip()}"
        return _truncate_text(_to_json(body), _MAX_COMMUNICATION_ITEM_CHARS)
    if body is None:
        return ""
    return _truncate_text(str(body), _MAX_COMMUNICATION_ITEM_CHARS)


def _summarize_review_diff_tool(raw_tool: Any) -> Dict[str, str]:
    if not isinstance(raw_tool, dict):
        return {}
    commands = raw_tool.get("commands")
    if not isinstance(commands, dict):
        return {}
    out: Dict[str, str] = {}
    for key in ("list", "show", "files"):
        value = commands.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = _truncate_text(value.strip(), _MAX_COMMUNICATION_ITEM_CHARS)
    return out


def _summarize_review_diff_views(commands_run: list[Any]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for item in commands_run[:20]:
        if not isinstance(item, dict):
            continue
        cmd = str(item.get("cmd") or "").strip()
        if "review_diff_tool.py" not in cmd:
            continue
        preview = ""
        stdout_tail = item.get("stdout_tail")
        stderr_tail = item.get("stderr_tail")
        if isinstance(stdout_tail, str) and stdout_tail.strip():
            preview = stdout_tail.strip()
        elif isinstance(stderr_tail, str) and stderr_tail.strip():
            preview = stderr_tail.strip()
        out.append(
            {
                "cmd": _truncate_text(cmd, _MAX_COMMUNICATION_ITEM_CHARS),
                "ok": bool(item.get("ok")),
                "exit_code": item.get("exit_code"),
                "preview": _truncate_text(preview, _MAX_REVIEW_DIFF_VIEW_CHARS),
            }
        )
    return out


def _extract_rewrite_preview(output: Dict[str, Any]) -> str:
    for key in ("intent_patch", "patch"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate_text(value.strip(), _MAX_REWRITE_PREVIEW_CHARS)

    previews: list[str] = []
    for key in ("intent_file_updates", "file_updates"):
        updates = output.get(key)
        if not isinstance(updates, list):
            continue
        for update in updates[:3]:
            if not isinstance(update, dict):
                continue
            path = str(update.get("path") or "").strip()
            content = update.get("content")
            if not isinstance(content, str):
                continue
            header = f"--- {path}" if path else "--- file update"
            previews.append(f"{header}\n{_truncate_text(content.strip(), 1200)}")
    if previews:
        return _truncate_text("\n\n".join(previews), _MAX_REWRITE_PREVIEW_CHARS)
    notes = output.get("notes")
    if isinstance(notes, str) and notes.strip():
        notes_text = notes.strip()
        if "```diff" in notes_text or "diff --git" in notes_text:
            return _truncate_text(notes_text, _MAX_REWRITE_PREVIEW_CHARS)
    return ""


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


def _zero_usage_metadata() -> Dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def _normalize_usage_metadata(raw: Any) -> Optional[Dict[str, int]]:
    if not isinstance(raw, dict):
        return None

    input_tokens = _read_non_negative_int(raw.get("input_tokens"), raw.get("prompt_tokens"))
    output_tokens = _read_non_negative_int(raw.get("output_tokens"), raw.get("completion_tokens"))
    total_tokens = _read_non_negative_int(raw.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = input_tokens + output_tokens
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
