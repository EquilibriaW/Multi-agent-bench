"""
loopbench.budget

Budget accounting and enforcement for tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Budget


@dataclass
class BudgetTracker:
    remaining: Budget

    @classmethod
    def from_budget(cls, budget: Budget) -> "BudgetTracker":
        return cls(remaining=Budget.model_validate(budget.model_dump()))

    def snapshot(self) -> Budget:
        return Budget.model_validate(self.remaining.model_dump())

    def can_call(self, tool_name: str) -> tuple[bool, str]:
        if self.remaining.tool_calls <= 0:
            return False, "tool_calls budget exhausted"

        if tool_name in {"env.up", "env.down"} and self.remaining.env_cycles <= 0:
            return False, "env_cycles budget exhausted"
        if tool_name == "env.logs_query" and self.remaining.log_queries <= 0:
            return False, "log_queries budget exhausted"
        if tool_name == "env.metrics_query" and self.remaining.metric_queries <= 0:
            return False, "metric_queries budget exhausted"
        if tool_name == "env.http_request" and self.remaining.http_requests <= 0:
            return False, "http_requests budget exhausted"

        return True, ""

    def consume(self, tool_name: str) -> None:
        self.remaining.tool_calls -= 1

        if tool_name in {"env.up", "env.down"}:
            self.remaining.env_cycles -= 1
        elif tool_name == "env.logs_query":
            self.remaining.log_queries -= 1
        elif tool_name == "env.metrics_query":
            self.remaining.metric_queries -= 1
        elif tool_name == "env.http_request":
            self.remaining.http_requests -= 1

        # Keep non-negative to simplify reporting.
        self.remaining.tool_calls = max(self.remaining.tool_calls, 0)
        self.remaining.env_cycles = max(self.remaining.env_cycles, 0)
        self.remaining.log_queries = max(self.remaining.log_queries, 0)
        self.remaining.metric_queries = max(self.remaining.metric_queries, 0)
        self.remaining.http_requests = max(self.remaining.http_requests, 0)
