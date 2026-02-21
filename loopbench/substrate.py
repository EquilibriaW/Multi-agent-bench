"""
loopbench.substrate

Local subprocess-backed substrate implementation.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .observability import ObservabilitySettings, build_observability
from .schema import SubstrateSpec, ToolResult
from .shell import run_shell
from .time_utils import now_ms


class LocalSubstrate:
    def __init__(
        self,
        role: str,
        worktree_path: str | Path,
        spec: SubstrateSpec,
        run_artifacts_dir: str | Path,
        observability: ObservabilitySettings | None = None,
    ):
        self.role = role
        self.worktree_path = Path(worktree_path).resolve()
        self.spec = spec
        self.run_artifacts_dir = Path(run_artifacts_dir).resolve()
        self._last_status: Dict[str, Any] = {"state": "initialized"}
        self.observability = build_observability(
            settings=observability or ObservabilitySettings(),
            artifacts_dir=self.run_artifacts_dir / "artifacts",
        )

    def kind(self) -> str:
        return self.spec.kind

    def up(self) -> None:
        if not self.spec.up_cmd:
            self._last_status = {"state": "up_skipped", "reason": "no up_cmd configured"}
            return
        result = run_shell(self.spec.up_cmd, cwd=self.worktree_path, timeout_sec=1800)
        self._write_artifact("env_up", result.stdout, result.stderr)
        self._last_status = {
            "state": "up_ok" if result.ok else "up_failed",
            "exit_code": result.exit_code,
        }
        if not result.ok:
            raise RuntimeError(f"env up failed for {self.role}: {result.stderr.strip()}")

    def down(self) -> None:
        if not self.spec.down_cmd:
            self._last_status = {"state": "down_skipped", "reason": "no down_cmd configured"}
            return
        result = run_shell(self.spec.down_cmd, cwd=self.worktree_path, timeout_sec=1800)
        self._write_artifact("env_down", result.stdout, result.stderr)
        self._last_status = {
            "state": "down_ok" if result.ok else "down_failed",
            "exit_code": result.exit_code,
        }
        if not result.ok:
            raise RuntimeError(f"env down failed for {self.role}: {result.stderr.strip()}")

    def status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def run_public_validation(self) -> ToolResult:
        started = now_ms()
        if not self.spec.public_validate_cmd:
            return ToolResult(
                ts_ms=started,
                ok=True,
                tool="env.public_validate",
                stdout="public validation skipped: no command configured\n",
                exit_code=0,
            )

        result = run_shell(self.spec.public_validate_cmd, cwd=self.worktree_path, timeout_sec=1800)
        self._write_artifact("public_validate", result.stdout, result.stderr)
        return ToolResult(
            ts_ms=started,
            ok=result.ok,
            tool="env.public_validate",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            data={"elapsed_sec": result.elapsed_sec},
        )

    def logs_tail(self, service: str, lines: int = 200) -> str:
        if self.spec.kind != "compose":
            return "logs_tail unavailable: substrate is not compose"

        command = f"docker compose logs --tail {int(lines)} {service}"
        result = run_shell(command, cwd=self.worktree_path, timeout_sec=120)
        if result.ok:
            return result.stdout
        return result.stderr or result.stdout

    def logs_query(self, query: str) -> str:
        return self.observability.logs_query(query)

    def metrics_query(self, query: str) -> str:
        return self.observability.metrics_query(query)

    def http_request(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body_bytes: Optional[bytes] = None
        headers = {}
        if json_body is not None:
            body_bytes = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url=url, method=method.upper(), data=body_bytes, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8", errors="replace")
                return {
                    "status": resp.status,
                    "headers": dict(resp.headers.items()),
                    "body": data,
                }
        except urllib.error.HTTPError as exc:
            return {
                "status": exc.code,
                "headers": dict(exc.headers.items()) if exc.headers else {},
                "body": exc.read().decode("utf-8", errors="replace"),
                "error": "http_error",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": 0,
                "headers": {},
                "body": "",
                "error": str(exc),
            }

    def _write_artifact(self, stem: str, stdout: str, stderr: str) -> None:
        out_dir = self.run_artifacts_dir / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.role}_{stem}.log"
        combined = f"STDOUT\n{stdout}\n\nSTDERR\n{stderr}\n"
        out_path.write_text(combined, encoding="utf-8")
