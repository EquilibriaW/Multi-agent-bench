"""
loopbench.hodoscope_export

Export benchmark run trajectories to hodoscope format and optionally
run LLM-driven summarization + embedding analysis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 8000


class TrajectoryExtractor:
    """Extracts hodoscope-format trajectories from a single benchmark run."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self._manifest: Dict[str, Any] = {}
        self._events: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        events_path = self.run_dir / "events.jsonl"
        if events_path.exists():
            with events_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            self._events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

    def extract(self) -> List[Dict[str, Any]]:
        """Extract one trajectory dict per role."""
        roles = self._manifest.get("roles", [])
        if not roles:
            return []

        trajectories = []
        for role in roles:
            traj = self._build_trajectory(role)
            if traj and traj.get("messages"):
                trajectories.append(traj)
        return trajectories

    def _build_trajectory(self, role: str) -> Dict[str, Any]:
        run_id = self._manifest.get("run_id", "unknown")
        task_id = self._manifest.get("task_id", "unknown")
        metrics = self._manifest.get("metrics", {})

        messages = self._build_messages_for_role(role)

        tokens_by_role = metrics.get("llm_tokens_by_role", {})
        role_tokens = tokens_by_role.get(role, {})

        metadata = {
            "model": self._detect_model(role),
            "score": 1.0 if self._manifest.get("hidden_pass") else 0.0,
            "instance_id": task_id,
            "run_id": run_id,
            "role": role,
            "input_tokens": role_tokens.get("input_tokens", 0),
            "output_tokens": role_tokens.get("output_tokens", 0),
            "total_tokens": role_tokens.get("total_tokens", 0),
            "public_pass": bool(self._manifest.get("public_pass")),
            "hidden_pass": bool(self._manifest.get("hidden_pass")),
            "review_iterations": metrics.get("review_iterations", 0),
        }

        return {
            "id": f"{run_id}__{role}",
            "messages": messages,
            "metadata": metadata,
        }

    def _build_messages_for_role(self, role: str) -> List[Dict[str, str]]:
        """Construct messages array from request/response artifacts + tool events."""
        messages: List[Dict[str, str]] = []

        # Pre-scan: find system prompt from the first role_phase for this role
        for event in self._events:
            if event.get("type") != "role_phase":
                continue
            payload = event.get("payload") or {}
            if payload.get("role") != role:
                continue
            output = payload.get("output") or {}
            artifact_paths = output.get("artifact_paths") or {}
            req_messages = self._read_request_messages(artifact_paths)
            if req_messages.get("system"):
                messages.append({"role": "system", "content": req_messages["system"]})
                break

        # Main pass: build chronological messages
        last_tool_call_role: str | None = None

        for event in self._events:
            event_type = event.get("type", "")
            payload = event.get("payload") or {}

            if event_type == "role_phase":
                event_role = payload.get("role", "")
                if event_role != role:
                    continue

                phase = payload.get("phase", "")
                output = payload.get("output") or {}
                artifact_paths = output.get("artifact_paths") or {}

                req_messages = self._read_request_messages(artifact_paths)
                user_content = req_messages.get("user", "")
                if user_content:
                    messages.append({"role": "user", "content": f"[{phase}] {user_content}"})
                else:
                    messages.append({"role": "user", "content": f"[{phase}]"})

                response_text = self._read_response_text(artifact_paths)
                if response_text:
                    messages.append({"role": "assistant", "content": response_text})

            elif event_type == "tool_call":
                event_role = payload.get("role", "")
                last_tool_call_role = event_role
                if event_role != role:
                    continue

                tool = payload.get("tool", "unknown")
                args = payload.get("args") or {}
                args_summary = _summarize_args(args)
                messages.append({
                    "role": "assistant",
                    "content": f"Tool: {tool}({args_summary})",
                })

            elif event_type == "tool_result":
                if last_tool_call_role != role:
                    last_tool_call_role = None
                    continue
                last_tool_call_role = None

                ok = payload.get("ok", False)
                exit_code = payload.get("exit_code")
                stderr = payload.get("stderr", "")
                stderr_tail = stderr[-500:] if stderr else ""
                content = f"Result: ok={ok}, exit={exit_code}"
                if stderr_tail:
                    content += f"\n{stderr_tail}"
                messages.append({"role": "user", "content": content})

        return messages

    def _read_request_messages(self, artifact_paths: Dict[str, Any]) -> Dict[str, str]:
        """Extract system/user prompts from openrouter request JSON."""
        req_path = self._find_artifact(artifact_paths, "_openrouter_request.json")
        if not req_path:
            return {}

        try:
            data = json.loads(req_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        result: Dict[str, str] = {}
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list):
            return result

        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            msg_role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        parts.append(item)
                text = "\n".join(parts)
            else:
                text = str(content) if content else ""

            text = _truncate(text, _MAX_CONTENT_CHARS)

            if msg_role == "system" and "system" not in result:
                result["system"] = text
            elif msg_role == "user" and "user" not in result:
                result["user"] = text

            if "system" in result and "user" in result:
                break

        return result

    def _read_response_text(self, artifact_paths: Dict[str, Any]) -> str:
        """Read raw LLM response text."""
        resp_path = self._find_artifact(artifact_paths, "_openrouter_response.txt")
        if not resp_path:
            return ""
        try:
            text = resp_path.read_text(encoding="utf-8")
            return _truncate(text, _MAX_CONTENT_CHARS)
        except Exception:
            return ""

    def _find_artifact(self, artifact_paths: Dict[str, Any], suffix: str) -> Optional[Path]:
        """Find an artifact file by suffix."""
        for key, value in artifact_paths.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            if key.endswith(suffix):
                path = Path(value)
                if not path.is_absolute():
                    path = self.run_dir / path
                if path.exists():
                    return path
        return None

    def _detect_model(self, role: str) -> str:
        """Detect the model used by a role from events."""
        for event in self._events:
            if event.get("type") != "role_phase":
                continue
            payload = event.get("payload") or {}
            if payload.get("role") != role:
                continue
            output = payload.get("output") or {}
            model = output.get("model")
            if model:
                return str(model)
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_run(run_dir: str | Path) -> Path:
    """Export trajectories for a single run.

    Returns the trajectory output directory.
    """
    run_dir = Path(run_dir).resolve()
    extractor = TrajectoryExtractor(run_dir)
    trajectories = extractor.extract()

    out_dir = run_dir / "hodoscope" / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    for traj in trajectories:
        traj_id = traj.get("id", "unknown")
        safe_name = traj_id.replace("/", "_").replace("\\", "_")
        out_path = out_dir / f"{safe_name}.json"
        out_path.write_text(
            json.dumps(traj, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return out_dir


def export_and_analyze(
    run_dirs: List[str],
    *,
    out_dir: str | None = None,
    viz: bool = False,
    summarize_model: str | None = None,
    embedding_model: str | None = None,
) -> Dict[str, Any]:
    """Batch: export trajectories + run hodoscope analysis + optional viz.

    Used by the ``loopbench hodoscope`` CLI subcommand.
    """
    try:
        import hodoscope
    except ImportError:
        raise SystemExit(
            "hodoscope is not installed. Install with: pip install hodoscope"
        )

    all_trajectory_dirs: List[Path] = []
    for rd in run_dirs:
        traj_dir = export_run(rd)
        all_trajectory_dirs.append(traj_dir)
        logger.info("Exported trajectories: %s", traj_dir)

    # Determine output directory
    if out_dir:
        analysis_out = Path(out_dir).resolve()
    elif len(all_trajectory_dirs) == 1:
        analysis_out = all_trajectory_dirs[0].parent  # <run_dir>/hodoscope/
    else:
        analysis_out = Path.cwd() / "hodoscope_analysis"
    analysis_out.mkdir(parents=True, exist_ok=True)

    # Build config with env-based defaults + explicit overrides
    config_kwargs: Dict[str, Any] = {}
    if summarize_model:
        config_kwargs["summarize_model"] = summarize_model
    if embedding_model:
        config_kwargs["embedding_model"] = embedding_model
    config = hodoscope.Config.from_env(**config_kwargs)

    # Load and process all trajectories
    all_trajectories: List[Dict[str, Any]] = []
    for traj_dir in all_trajectory_dirs:
        trajectories, _fields = hodoscope.load_trajectory_dir(str(traj_dir))
        all_trajectories.extend(trajectories)

    if not all_trajectories:
        return {
            "status": "no_trajectories",
            "dirs": [str(d) for d in all_trajectory_dirs],
        }

    summaries = hodoscope.process_trajectories(all_trajectories, config=config)
    analysis_path = analysis_out / "analysis.hodoscope.json"

    fields: Dict[str, Any] = {}
    if all_trajectories:
        meta = all_trajectories[0].get("metadata", {})
        fields["model"] = meta.get("model", "unknown")

    hodoscope.write_analysis_json(
        str(analysis_path), summaries, fields, config=config
    )
    logger.info("Analysis written: %s", analysis_path)

    result: Dict[str, Any] = {
        "status": "ok",
        "analysis_path": str(analysis_path),
        "trajectory_count": len(all_trajectories),
        "summary_count": len(summaries),
    }

    if viz:
        analysis_docs = [hodoscope.read_analysis_json(str(analysis_path))]
        grouped = hodoscope.group_summaries(analysis_docs)
        viz_path = hodoscope.visualize_action_summaries(
            grouped,
            output_file=str(analysis_out / "visualization.html"),
        )
        result["viz_path"] = str(viz_path)
        logger.info("Visualization: %s", viz_path)

    return result


def auto_analyze(run_dir: str | Path) -> None:
    """Called from controller post-run: export + analyze if keys available.

    Never raises — all errors are logged and swallowed.
    """
    try:
        run_dir = Path(run_dir).resolve()
        traj_dir = export_run(run_dir)
        logger.info("Hodoscope trajectories exported: %s", traj_dir)

        try:
            import hodoscope
        except ImportError:
            logger.debug("hodoscope not installed, skipping analysis")
            return

        config = hodoscope.Config.from_env()
        trajectories, fields = hodoscope.load_trajectory_dir(str(traj_dir))
        if not trajectories:
            return

        summaries = hodoscope.process_trajectories(trajectories, config=config)
        analysis_path = run_dir / "hodoscope" / "analysis.hodoscope.json"
        hodoscope.write_analysis_json(
            str(analysis_path), summaries, fields, config=config
        )
        logger.info("Hodoscope analysis written: %s", analysis_path)

    except Exception as exc:  # noqa: BLE001
        logger.debug("Hodoscope auto-analyze skipped: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _summarize_args(args: Dict[str, Any], max_len: int = 200) -> str:
    """Produce a compact summary of tool call arguments."""
    if not args:
        return ""
    try:
        text = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(args)
    return _truncate(text, max_len)
