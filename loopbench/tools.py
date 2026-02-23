"""
loopbench.tools

ToolRouter implementation with budget checks and structured event logging.
"""
from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List

from .budget import BudgetTracker
from .events import EventLogger
from .interfaces import ToolRouter
from .sandbox import LocalSandbox
from .schema import Budget, ToolCall, ToolResult
from .shell import run_command
from .substrate import LocalSubstrate
from .time_utils import now_ms


class DefaultToolRouter(ToolRouter):
    def __init__(
        self,
        run_id: str,
        sandboxes: Dict[str, LocalSandbox],
        substrates: Dict[str, LocalSubstrate],
        budget: Budget,
        event_logger: EventLogger,
    ):
        self.run_id = run_id
        self.sandboxes = sandboxes
        self.substrates = substrates
        self.event_logger = event_logger
        self.budget = BudgetTracker.from_budget(budget)

    def call(self, call: ToolCall) -> ToolResult:
        allowed, reason = self.budget.can_call(call.tool)
        if not allowed:
            result = self._error(tool=call.tool, stderr=reason, exit_code=1)
            self.event_logger.log("tool_denied", {"call": call.model_dump(), "result": result.model_dump()})
            return result

        self.event_logger.log("tool_call", call.model_dump())

        try:
            result = self._dispatch(call)
        except Exception as exc:  # noqa: BLE001
            result = self._error(tool=call.tool, stderr=str(exc), exit_code=1)
        finally:
            self.budget.consume(call.tool)

        self.event_logger.log("tool_result", result.model_dump())
        return result

    def remaining_budget(self) -> Budget:
        return self.budget.snapshot()

    def _dispatch(self, call: ToolCall) -> ToolResult:
        role = call.role
        if role not in self.sandboxes:
            raise ValueError(f"unknown role: {role}")

        sandbox = self.sandboxes[role]
        substrate = self.substrates[role]

        if call.tool.startswith("repo."):
            return self._dispatch_repo(tool=call.tool, args=call.args, sandbox=sandbox)
        if call.tool.startswith("env."):
            return self._dispatch_env(tool=call.tool, args=call.args, substrate=substrate)
        if call.tool.startswith("meta."):
            return self._dispatch_meta(tool=call.tool)
        raise ValueError(f"unknown tool: {call.tool}")

    def _dispatch_repo(self, *, tool: str, args: Dict[str, Any], sandbox: LocalSandbox) -> ToolResult:
        if tool == "repo.exec":
            cmd_arg = args.get("cmd")
            if isinstance(cmd_arg, str):
                cmd = ["bash", "-lc", cmd_arg]
            elif isinstance(cmd_arg, list) and all(isinstance(x, str) for x in cmd_arg):
                cmd = cmd_arg
            else:
                raise ValueError("repo.exec requires cmd (string or list[string])")
            cwd = args.get("cwd")
            timeout_sec = int(args.get("timeout_sec", 600))
            return self._with_tool(sandbox.exec(cmd=cmd, cwd=cwd, timeout_sec=timeout_sec), tool)

        if tool == "repo.read_file":
            path = str(args["path"])
            content = sandbox.read_file(path)
            return self._ok(tool=tool, stdout=content, data={"path": path})

        if tool == "repo.write_file":
            path = str(args["path"])
            content = str(args.get("content", ""))
            sandbox.write_file(path, content)
            return self._ok(tool=tool, data={"path": path, "bytes": len(content.encode("utf-8"))})

        if tool == "repo.apply_patch":
            patch_text = str(args["patch_text"])
            if patch_text and not patch_text.endswith("\n"):
                patch_text = f"{patch_text}\n"
            sandbox.apply_patch(patch_text)
            return self._ok(tool=tool, data={"bytes": len(patch_text.encode("utf-8"))})

        if tool == "repo.git_status":
            return self._with_tool(sandbox.exec(["git", "status", "--short"], timeout_sec=30), tool)

        if tool == "repo.git_diff":
            base_ref = args.get("base_ref")
            cmd = ["git", "diff", "--binary"]
            if base_ref:
                cmd.append(str(base_ref))
            return self._with_tool(sandbox.exec(cmd, timeout_sec=60), tool)

        if tool == "repo.git_add_commit":
            message = str(args.get("message") or "loopbench commit")
            return self._with_tool(self._git_add_commit_local(sandbox=sandbox, message=message), tool)

        if tool == "repo.git_log":
            limit = int(args.get("limit", 20))
            return self._with_tool(
                sandbox.exec(["git", "log", f"--max-count={limit}", "--oneline"], timeout_sec=30),
                tool,
            )

        if tool == "repo.branch_head":
            branch = args.get("branch")
            cmd = ["git", "rev-parse", "HEAD"] if not branch else ["git", "rev-parse", str(branch)]
            return self._with_tool(sandbox.exec(cmd, timeout_sec=30), tool)

        if tool == "repo.cherry_pick":
            commit = str(args["commit"])
            return self._with_tool(sandbox.exec(["git", "cherry-pick", commit], timeout_sec=120), tool)

        if tool == "repo.list_files":
            return self._with_tool(sandbox.exec(["git", "ls-files"], timeout_sec=30), tool)

        raise ValueError(f"unknown repo tool: {tool}")

    def _dispatch_env(self, *, tool: str, args: Dict[str, Any], substrate: LocalSubstrate) -> ToolResult:
        if tool == "env.up":
            substrate.up()
            return self._ok(tool=tool)

        if tool == "env.down":
            substrate.down()
            return self._ok(tool=tool)

        if tool == "env.status":
            return self._ok(tool=tool, data=substrate.status())

        if tool == "env.public_validate":
            return self._with_tool(substrate.run_public_validation(), tool)

        if tool == "env.logs_tail":
            service = str(args.get("service") or "")
            lines = int(args.get("lines", 200))
            output = substrate.logs_tail(service=service, lines=lines)
            return self._ok(tool=tool, stdout=output)

        if tool == "env.logs_query":
            query = str(args.get("query") or "")
            output = substrate.logs_query(query=query)
            return self._ok(tool=tool, stdout=output)

        if tool == "env.metrics_query":
            query = str(args.get("query") or "")
            output = substrate.metrics_query(query=query)
            return self._ok(tool=tool, stdout=output)

        if tool == "env.http_request":
            method = str(args.get("method") or "GET")
            url = str(args["url"])
            json_body = args.get("json_body")
            response = substrate.http_request(method=method, url=url, json_body=json_body)
            return self._ok(tool=tool, stdout=json.dumps(response), data=response)

        raise ValueError(f"unknown env tool: {tool}")

    def _dispatch_meta(self, *, tool: str) -> ToolResult:
        if tool == "meta.remaining_budget":
            return self._ok(tool=tool, data=self.remaining_budget().model_dump())

        if tool == "meta.run_info":
            return self._ok(
                tool=tool,
                data={"run_id": self.run_id, "roles": sorted(self.sandboxes.keys())},
            )

        raise ValueError(f"unknown meta tool: {tool}")

    def _ok(
        self,
        *,
        tool: str,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
        data: Dict[str, Any] | None = None,
    ) -> ToolResult:
        return ToolResult(
            ts_ms=now_ms(),
            ok=True,
            tool=tool,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            data=data or {},
        )

    def _error(self, *, tool: str, stderr: str, exit_code: int = 1) -> ToolResult:
        return ToolResult(
            ts_ms=now_ms(),
            ok=False,
            tool=tool,
            stderr=stderr,
            exit_code=exit_code,
        )

    def _with_tool(self, result: ToolResult, tool: str) -> ToolResult:
        result.tool = tool
        return result

    def _git_add_commit_local(self, *, sandbox: LocalSandbox, message: str) -> ToolResult:
        quoted_message = shlex.quote(message)
        commit_script = (
            "set -euo pipefail\n"
            "if ! git status --porcelain | grep -q .; then\n"
            "  exit 0\n"
            "fi\n"
            "git add -A\n"
            "git reset -q HEAD -- .loopbench || true\n"
            "staged_before=\"$(git diff --cached --name-only)\"\n"
            "if [ -z \"$staged_before\" ]; then\n"
            "  exit 0\n"
            "fi\n"
            "while IFS= read -r staged_path; do\n"
            "  [ -n \"$staged_path\" ] || continue\n"
            "  case \"$staged_path\" in\n"
            "    *.orig|*.rej|*.pyc|.coverage|.coverage.*|"
            "__pycache__|__pycache__/*|*/__pycache__/*|"
            ".pytest_cache|.pytest_cache/*|*/.pytest_cache/*|"
            ".mypy_cache|.mypy_cache/*|*/.mypy_cache/*|"
            ".ruff_cache|.ruff_cache/*|*/.ruff_cache/*|"
            ".hypothesis|.hypothesis/*|*/.hypothesis/*)\n"
            "      git reset -q HEAD -- \"$staged_path\" || true\n"
            "      ;;\n"
            "  esac\n"
            "done <<EOF\n"
            "$staged_before\n"
            "EOF\n"
            "if ! git diff --cached --name-only | grep -q .; then\n"
            "  exit 0\n"
            "fi\n"
            f"git commit -m {quoted_message} >/dev/null\n"
            "git rev-parse HEAD\n"
        )
        result = run_command(["bash", "-lc", commit_script], cwd=sandbox.root, timeout_sec=90)
        return ToolResult(
            ts_ms=now_ms(),
            ok=result.ok,
            tool="repo.git_add_commit",
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            data={"cwd": str(sandbox.root), "elapsed_sec": result.elapsed_sec},
        )
