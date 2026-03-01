#!/usr/bin/env python3
"""
OpenRouter-backed role driver for LoopBench team experiments (multi-turn agentic mode).

Uses OpenRouter tool calling to run an agentic loop: the LLM receives tool schemas,
invokes tools via function calls, receives results, and iterates until it calls submit().

Expected env from harness:
- LB_ROLE
- LB_PHASE
- LB_WORKTREE
- LB_CONTEXT_JSON
- LB_OUTPUT_JSON
- LB_MODEL

Optional env:
- OPENROUTER_API_KEY / OPEN_ROUTER_API_KEY
- OPENROUTER_API_KEY_ENV (env var name containing the API key)
- OPENROUTER_BASE_URL (default: https://openrouter.ai/api/v1)
- OPENROUTER_HTTP_REFERER
- OPENROUTER_APP_TITLE
- OPENROUTER_TEMPERATURE (default: 0.2)
- OPENROUTER_MAX_TOKENS (default: 4096)
- OPENROUTER_REASONING_ENABLED (optional bool)
- OPENROUTER_REASONING_EFFORT (optional string, e.g. none/low/medium/high)
- OPENROUTER_REASONING_MAX_TOKENS (optional int >= 0)
- OPENROUTER_REASONING_EXCLUDE (optional bool)
- OPENROUTER_HTTP_TIMEOUT_SEC (default: 90)
- OPENROUTER_HTTP_RETRIES (default: 4)
- LOOPBENCH_SANDBOX_BACKEND (injected by harness; used for default policy)
- LOOPBENCH_MAX_COMMANDS (default: 12 in e2b, 8 otherwise)
- LOOPBENCH_COMMAND_TIMEOUT_SEC (default: 1200 in e2b, 180 otherwise)
- LOOPBENCH_PLANNER_NON_MUTATING_PHASES (optional comma-separated phases; e.g. review,finalize)
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_OPENROUTER_HTTP_RETRIES = 4
DEFAULT_OPENROUTER_HTTP_TIMEOUT_SEC = 90

OUTPUT_TRUNCATE_CHARS = 10000
MAX_TURNS_CODER = 50
MAX_TURNS_REVIEWER = 50
MAX_TURNS_BOOTSTRAP = 50
MAX_TURNS_REFLECT = 50
MAX_TURNS_FINALIZE = 50


# ---------------------------------------------------------------------------
# System prompt constants
# ---------------------------------------------------------------------------

TEAM_BASE_PROMPT = (
    "You are part of a 3-agent coding team in a benchmark harness. "
    "Prefer minimal, correct edits. Keep files syntactically valid. "
    "You have tools to explore the repository, modify files, and run commands. "
    "Use read_file to understand existing code before making changes. "
    "Call submit() when your work is complete."
)

BOOTSTRAP_SYSTEM = (
    "You are a technical planner. Read the task description, explore the repository, "
    "and decompose the work into subtasks for your 2 coders. "
    "Use lookup_docs() to check parameter formats before calling submit(). "
    "Call submit() with plan_markdown and subtasks. "
    "Split work so coders can work in parallel (different files when possible)."
)

REVIEW_SYSTEM = (
    "You are a code reviewer. Complete diffs for each coder are provided inline in the user message — "
    "read them carefully before doing anything else. "
    "Evaluate each diff against the task requirements. You may use read_file or exec to explore "
    "the repo for additional context if needed. "
    "Use lookup_docs('submit') to check parameter formats before calling submit(). "
    "When you have formed your judgment, call submit() with: "
    "merge_commits={\"coder_a\": [\"<full_sha>\"], ...} for correct implementations, "
    "request_rework=true and coder_feedback={\"coder_a\": \"what to fix\"} for implementations that need work. "
    "You MUST include at least one SHA in merge_commits if any diff is correct."
)

FINALIZE_SYSTEM = (
    "You are doing final integration checks. Verify the code works end-to-end. "
    "Use read_file and exec to check. Call submit() when satisfied."
)

GENERIC_CODER_ROLE_PROMPT = (
    "Role: coder (applies equally to coder_a and coder_b). "
    "You execute only assigned work from planner-reviewer messages/assignment payload. "
    "Do not invent a new work split. If assignment is missing/ambiguous, report blocker in notes. "
    "Use read_file to understand existing code before making changes. "
    "Use write_file or apply_patch to modify files. "
    "Use exec to run tests after changes. "
    "Use lookup_docs() to check parameter formats before calling submit() or other tools. "
    "Call submit(summary, commit_message) when your work is complete."
)

REFLECTION_PROMPT = (
    "Role: reflection analyst. You analyze the execution trace of the most recent review round "
    "and produce structured knowledge for the next round. Your output is DIRECTIVE — it tells "
    "agents exactly what to do differently, not what happened. "
    "Read the context provided, then call submit() with your directive and knowledge surfaces. "
    "Rules: "
    "1. Write a concise directive (200-500 chars) with embedded fix instructions. "
    "2. OVERWRITE each knowledge surface completely. Do not append to previous content. "
    "3. Explicitly list which previous insights are superseded in the 'superseded' array. "
    "4. Focus on actionable patterns, not raw data. What should change, not what happened. "
    "5. Keep task_understanding updated with refined understanding of what the task requires. "
    "6. In failure_patterns, only keep CURRENT failures. Remove resolved ones. "
    "7. In workflow_insights, capture strategic observations about what's working vs not."
)


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenRouter function calling format)
# ---------------------------------------------------------------------------

COMMON_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List directory contents with file sizes",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to repo root (default: '.')"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file content. Output truncated to 10KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "offset": {"type": "integer", "description": "Line offset to start reading from (0-based)"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search file contents with regex. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in, relative to repo root (default: '.')"},
                    "include": {"type": "string", "description": "Glob pattern to filter files (e.g. '*.py', '*.java')"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command in the repo working directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout_sec": {"type": "integer", "description": "Timeout in seconds (default: per policy)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show git working tree status",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show diff vs HEAD or a specific ref",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Git ref to diff against (default: HEAD)"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report_ambiguity",
            "description": "Report an ambiguity in the task, tools, or acceptance criteria. Use this when you encounter something unclear rather than guessing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["task_spec", "tool_affordance", "acceptance_criteria", "other"],
                        "description": "Category of ambiguity",
                    },
                    "description": {"type": "string", "description": "What is ambiguous"},
                    "attempted_resolution": {"type": "string", "description": "What you did instead of getting clarification"},
                },
                "required": ["category", "description"],
            },
        },
    },
]

CODER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write full content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to repo root"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff patch. Hunk counts are auto-corrected.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch_text": {"type": "string", "description": "Unified diff patch text"}
                },
                "required": ["patch_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finalize your work: stage, commit changes, and end the session",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of what was done"},
                    "commit_message": {"type": "string", "description": "Git commit message"},
                },
                "required": ["summary", "commit_message"],
            },
        },
    },
]

REVIEWER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show commit history",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "Git ref (default: HEAD)"},
                    "max_count": {"type": "integer", "description": "Max commits to show (default: 20)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Finalize review: submit merge decisions and feedback",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Review summary"},
                    "merge_commits": {
                        "type": "object",
                        "description": "Dict of role -> list of commit SHAs to merge",
                    },
                    "request_rework": {
                        "type": "boolean",
                        "description": "Whether to request rework from coders",
                    },
                    "coder_feedback": {
                        "type": "object",
                        "description": "Dict of role -> feedback string",
                    },
                },
                "required": ["summary", "merge_commits"],
            },
        },
    },
]

BOOTSTRAP_SUBMIT_TOOL: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the bootstrap plan and subtask decomposition",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_markdown": {"type": "string", "description": "Full plan in markdown format"},
                    "subtasks": {
                        "type": "array",
                        "description": "List of subtask objects with id, role, title, paths, acceptance",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string", "enum": ["coder_a", "coder_b"]},
                                "title": {"type": "string"},
                                "paths": {"type": "array", "items": {"type": "string"}},
                                "acceptance": {"type": "string"},
                            },
                        },
                    },
                    "summary": {"type": "string", "description": "Short summary of the plan"},
                },
                "required": ["plan_markdown", "subtasks"],
            },
        },
    },
]

REFLECT_SUBMIT_TOOL: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit reflection knowledge surfaces",
            "parameters": {
                "type": "object",
                "properties": {
                    "directive": {
                        "type": "string",
                        "description": "Concise directive for next round (200-500 chars)",
                    },
                    "task_understanding": {"type": "string"},
                    "failure_patterns": {"type": "string"},
                    "workflow_insights": {"type": "string"},
                    "superseded": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["directive"],
            },
        },
    },
]

FINALIZE_SUBMIT_TOOL: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit finalization summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Final summary"},
                    "commit_message": {"type": "string", "description": "Optional commit message"},
                },
                "required": ["summary"],
            },
        },
    },
]

DOCS_TOOL: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_docs",
            "description": (
                "Look up usage documentation for a tool or topic before using it. "
                "Call this when unsure about parameters, valid values, or best practices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Tool name or topic to look up (e.g. 'submit', 'merge_commits', 'dockerfile_templates')",
                    }
                },
                "required": ["topic"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Documentation registry (progressive disclosure)
# ---------------------------------------------------------------------------

_DOCS_REGISTRY: Dict[str, str] = {
    "submit": (
        "submit() -- Reviewer variant\n"
        "Parameters:\n"
        "  summary: str -- Review summary.\n"
        "  merge_commits: Dict mapping coder role names to lists of full commit SHAs.\n"
        "    Valid role keys: \"coder_a\", \"coder_b\" (these are the ONLY valid keys).\n"
        "    Example: {\"coder_a\": [\"abc123def456...\"], \"coder_b\": [\"789xyz...\"]}\n"
        "    Do NOT use keys like \"reference\", \"main\", or any other names.\n"
        "  request_rework: boolean, set to true if implementations need changes.\n"
        "  coder_feedback: Dict mapping role names to feedback strings.\n"
        "    Example: {\"coder_a\": \"Fix the return type on line 42\"}\n"
        "\n"
        "See also: submit_coder, submit_bootstrap"
    ),
    "submit_review": (
        "submit() -- Reviewer variant\n"
        "Parameters:\n"
        "  summary: str -- Review summary.\n"
        "  merge_commits: Dict mapping coder role names to lists of full commit SHAs.\n"
        "    Valid role keys: \"coder_a\", \"coder_b\" (these are the ONLY valid keys).\n"
        "    Example: {\"coder_a\": [\"abc123def456...\"], \"coder_b\": [\"789xyz...\"]}\n"
        "    Do NOT use keys like \"reference\", \"main\", or any other names.\n"
        "  request_rework: boolean, set to true if implementations need changes.\n"
        "  coder_feedback: Dict mapping role names to feedback strings.\n"
        "    Example: {\"coder_a\": \"Fix the return type on line 42\"}\n"
    ),
    "merge_commits": (
        "merge_commits parameter (used in reviewer submit())\n"
        "Type: Dict[str, List[str]]\n"
        "Maps coder role names to lists of full commit SHAs to merge.\n"
        "Valid role keys: \"coder_a\", \"coder_b\" -- no other keys are accepted.\n"
        "Example: {\"coder_a\": [\"abc123def456...\"], \"coder_b\": [\"789xyz...\"]}\n"
        "Do NOT use keys like \"reference\", \"main\", or branch names.\n"
    ),
    "submit_coder": (
        "submit() -- Coder variant\n"
        "Parameters:\n"
        "  summary: str -- Summary of what was done.\n"
        "  commit_message: str -- Git commit message for the changes.\n"
        "Call submit() after you have made and tested your changes.\n"
    ),
    "submit_bootstrap": (
        "submit() -- Bootstrap/planner variant\n"
        "Parameters:\n"
        "  plan_markdown: str -- Full implementation plan in markdown.\n"
        "  subtasks: List[Dict] -- Subtask objects with fields:\n"
        "    id: str -- Unique subtask identifier (e.g. \"1\", \"2\").\n"
        "    role: str -- Must be \"coder_a\" or \"coder_b\".\n"
        "    title: str -- Short description of the subtask.\n"
        "    paths: List[str] -- Files this subtask will touch.\n"
        "    acceptance: str -- Acceptance criteria for completion.\n"
        "  summary: str -- Short summary of the plan.\n"
        "Valid role values for subtasks: \"coder_a\", \"coder_b\" only.\n"
    ),
    "dockerfile": (
        "Dockerfile base image recommendations:\n"
        "  Java 17+: eclipse-temurin:17-jdk (NOT openjdk, which is deprecated on Docker Hub)\n"
        "  Java 11: eclipse-temurin:11-jdk\n"
        "  Node.js: node:18-bookworm (use -bookworm not -alpine if native deps needed)\n"
        "  Python: python:3.11-slim-bookworm\n"
        "  Ruby: ruby:3.2-bookworm (include build-essential, git, libyaml-dev for native gems)\n"
        "  Go: golang:1.22-bookworm\n"
        "  .NET 6+: mcr.microsoft.com/dotnet/sdk:8.0\n"
        "  .NET 3.1: mcr.microsoft.com/dotnet/sdk:3.1 (EOL but still available)\n"
        "  Rust: rust:1.77-bookworm\n"
        "  PHP: php:8.2-cli (or php:8.2-apache for web)\n"
        "\n"
        "Tips:\n"
        "  - Prefer -bookworm or -bullseye over -alpine for apps with native dependencies.\n"
        "  - Always include build-essential and git for languages with native extensions.\n"
        "  - Check the repo's version requirements before choosing\n"
        "    (e.g. java.sourceCompatibility in build.gradle).\n"
    ),
    "dockerfile_templates": (
        "Dockerfile base image recommendations:\n"
        "  Java 17+: eclipse-temurin:17-jdk (NOT openjdk, which is deprecated on Docker Hub)\n"
        "  Java 11: eclipse-temurin:11-jdk\n"
        "  Node.js: node:18-bookworm (use -bookworm not -alpine if native deps needed)\n"
        "  Python: python:3.11-slim-bookworm\n"
        "  Ruby: ruby:3.2-bookworm (include build-essential, git, libyaml-dev for native gems)\n"
        "  Go: golang:1.22-bookworm\n"
        "  .NET 6+: mcr.microsoft.com/dotnet/sdk:8.0\n"
        "  .NET 3.1: mcr.microsoft.com/dotnet/sdk:3.1 (EOL but still available)\n"
        "  Rust: rust:1.77-bookworm\n"
        "  PHP: php:8.2-cli (or php:8.2-apache for web)\n"
        "\n"
        "Tips:\n"
        "  - Prefer -bookworm or -bullseye over -alpine for apps with native dependencies.\n"
        "  - Always include build-essential and git for languages with native extensions.\n"
        "  - Check the repo's version requirements before choosing\n"
        "    (e.g. java.sourceCompatibility in build.gradle).\n"
    ),
    "exec": (
        "exec tool -- Run a shell command in the repo working directory.\n"
        "Parameters:\n"
        "  command: str -- Shell command to execute.\n"
        "  timeout_sec: int -- Timeout in seconds (capped by sandbox policy).\n"
        "\n"
        "Tips:\n"
        "  - Commands run with shell=True, so pipes and redirects work.\n"
        "  - Use 'cd' sparingly; paths are relative to the repo root.\n"
        "  - For long-running builds, increase timeout_sec.\n"
        "  - Check exit codes: non-zero is appended as [exit code: N].\n"
        "  - Output is truncated to 10KB; pipe through tail/head for large outputs.\n"
    ),
    "apply_patch": (
        "apply_patch tool -- Apply a unified diff patch.\n"
        "Parameters:\n"
        "  patch_text: str -- Unified diff text.\n"
        "\n"
        "Format:\n"
        "  --- a/path/to/file\n"
        "  +++ b/path/to/file\n"
        "  @@ -start,count +start,count @@\n"
        "   context line (space prefix)\n"
        "  -removed line\n"
        "  +added line\n"
        "\n"
        "Tips:\n"
        "  - Hunk counts are auto-corrected, so don't worry about exact counts.\n"
        "  - Context lines MUST start with a space character.\n"
        "  - Include enough context (3 lines) for unambiguous matching.\n"
        "  - Paths are relative to repo root.\n"
        "  - For large changes, prefer write_file over complex patches.\n"
    ),
}


def _lookup_docs(topic: str) -> str:
    """Look up documentation for a tool or topic, with fuzzy matching."""
    topic = topic.lower().strip()
    if not topic:
        available = sorted(_DOCS_REGISTRY.keys())
        return "Please specify a topic. Available topics: " + ", ".join(available)

    # Exact match
    if topic in _DOCS_REGISTRY:
        return _DOCS_REGISTRY[topic]

    # Substring match: find all topics containing the query or vice versa
    matches = [key for key in _DOCS_REGISTRY if topic in key or key in topic]
    if len(matches) == 1:
        return _DOCS_REGISTRY[matches[0]]
    if matches:
        return (
            f"Multiple topics match '{topic}': {', '.join(sorted(matches))}.\n"
            "Please specify one of these exactly."
        )

    # No match at all
    available = sorted(_DOCS_REGISTRY.keys())
    return f"No documentation found for '{topic}'. Available topics: " + ", ".join(available)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> int:
    role = _required_env("LB_ROLE")
    phase = _required_env("LB_PHASE")
    worktree = Path(_required_env("LB_WORKTREE")).resolve()
    context_path = Path(_required_env("LB_CONTEXT_JSON")).resolve()
    output_path = Path(_required_env("LB_OUTPUT_JSON")).resolve()
    model = os.environ.get("LB_MODEL") or DEFAULT_MODEL

    context = _read_json_file(context_path)
    api_key = _resolve_openrouter_api_key()
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is missing. Set OPENROUTER_API_KEY or OPEN_ROUTER_API_KEY, "
            "or set OPENROUTER_API_KEY_ENV to the env-var name."
        )

    repo_ctx = _collect_repo_context(worktree=worktree, context=context)
    payload = _build_payload(
        role=role,
        phase=phase,
        model=model,
        context=context,
        repo_ctx=repo_ctx,
    )
    _write_json_file(_role_trace_path(output_path, "openrouter_request", ".json"), payload)

    tools_schema = _tools_for_phase(role=role, phase=phase)
    max_turns = _max_turns_for_phase(role=role, phase=phase)

    result = _run_agentic_loop(
        payload=payload,
        api_key=api_key,
        tools_schema=tools_schema,
        phase=phase,
        role=role,
        max_turns=max_turns,
        worktree=worktree,
        context=context,
        output_path=output_path,
        model=model,
    )
    _write_json_file(output_path, result)
    return 0


# ---------------------------------------------------------------------------
# Payload / prompt construction
# ---------------------------------------------------------------------------

def _build_payload(
    *,
    role: str,
    phase: str,
    model: str,
    context: Dict[str, Any],
    repo_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    system_prompt = _build_system_prompt(role=role, phase=phase, context=context)
    user_prompt = _build_user_prompt(role=role, phase=phase, context=context, repo_ctx=repo_ctx)

    temperature = float(os.environ.get("OPENROUTER_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "4096"))

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    reasoning = _reasoning_payload_from_env()
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return payload


def _build_user_prompt(
    *,
    role: str,
    phase: str,
    context: Dict[str, Any],
    repo_ctx: Dict[str, Any],
) -> str:
    """Dispatch to phase-specific prompt builders producing readable markdown."""
    if phase == "bootstrap":
        return _build_bootstrap_prompt(context, repo_ctx)
    if phase == "review":
        if role == "planner_reviewer":
            return _build_review_prompt(context, repo_ctx)
    if phase == "review_verify":
        if role == "planner_reviewer":
            return _build_review_verify_prompt(context, repo_ctx)
    if phase == "reflect":
        return _build_reflect_prompt(context, repo_ctx)
    if phase == "finalize":
        return _build_finalize_prompt(context, repo_ctx)
    # Coder phases (implementation, rework)
    return _build_coder_prompt(context, repo_ctx)


def _build_bootstrap_prompt(context: Dict[str, Any], repo_ctx: Dict[str, Any]) -> str:
    task_id = context.get("task_id", "unknown")
    readme = repo_ctx.get("readme_task_md") or "(no task description available)"
    tracked = repo_ctx.get("tracked_files", [])
    tree = "\n".join(tracked[:200]) if tracked else "(no tracked files)"

    return (
        f"# Task: {task_id}\n\n"
        + _time_budget_line(context)
        + f"{readme}\n\n"
        f"## Repository Structure\n```\n{tree}\n```\n\n"
        "## Your Job\n"
        "Create an implementation plan and split it into subtasks for your 2 coders.\n"
        "Call submit() with plan_markdown and subtasks.\n"
        "Split work so coders can work in parallel (different files when possible).\n"
    )


def _build_review_prompt(context: Dict[str, Any], repo_ctx: Dict[str, Any]) -> str:
    round_num = context.get("round", "?")
    readme = repo_ctx.get("readme_task_md") or "(no task description available)"

    parts: List[str] = [f"# Review Round {round_num}\n\n" + _time_budget_line(context) + f"## Task Requirements\n{readme}\n"]

    directive = context.get("reflection_directive")
    if directive:
        parts.append(f"## Reflection Directive\n{directive}\n")

    # Inline diffs
    inline_diffs = context.get("inline_diffs", {})
    if inline_diffs:
        for coder, entries in inline_diffs.items():
            for entry in entries:
                sha = entry.get("sha", "?")
                subject = entry.get("subject", "")
                files = entry.get("files_changed", [])
                diff = entry.get("diff_content", "")
                files_str = ", ".join(files[:20]) if files else "(unknown)"
                parts.append(
                    f"## {coder} — SHA: {sha}\n"
                    f"Subject: {subject}\n"
                    f"Files changed: {files_str}\n"
                    f"```diff\n{diff}\n```\n"
                )
    else:
        # Fallback: show candidate_merge_commits metadata
        candidates = context.get("candidate_merge_commits", {})
        if candidates:
            for coder, entries in candidates.items():
                for entry in entries:
                    sha = entry.get("sha", "?")[:8]
                    subject = entry.get("subject", "")
                    files = entry.get("files_changed", [])
                    files_str = ", ".join(files[:20]) if files else "(unknown)"
                    parts.append(
                        f"## {coder} — {subject} ({sha})\n"
                        f"Files changed: {files_str}\n"
                        "(diff not available — use git_show to inspect)\n"
                    )

    # Coder command failures (build/test errors from coder runs)
    latest_outputs = context.get("latest_coder_outputs", {})
    if isinstance(latest_outputs, dict):
        for coder_name, outputs in latest_outputs.items():
            if not isinstance(outputs, list):
                continue
            for out_entry in outputs:
                failed = out_entry.get("failed_commands", []) if isinstance(out_entry, dict) else []
                if failed:
                    parts.append(f"### {coder_name} Command Failures\n")
                    for f in failed:
                        stderr_snippet = str(f.get("stderr_tail", ""))[:300]
                        parts.append(
                            f"- `{f.get('command', '')}` exit={f.get('exit_code', '?')}"
                            + (f": {stderr_snippet}" if stderr_snippet else "")
                            + "\n"
                        )

    # Previous test results
    last_val = context.get("last_public_validation", {})
    if isinstance(last_val, dict):
        stdout = last_val.get("stdout", "")
        stderr = last_val.get("stderr", "")
        val_text = ""
        if stdout:
            val_text += stdout[-2000:]
        if stderr:
            val_text += f"\n[stderr]\n{stderr[-2000:]}"
        if val_text.strip():
            parts.append(f"## Previous Test Results\n```\n{val_text.strip()}\n```\n")
        else:
            parts.append("## Previous Test Results\nNo tests run yet.\n")
    else:
        parts.append("## Previous Test Results\nNo tests run yet.\n")

    parts.append(
        "## Your Decision\n"
        "The diffs above contain the complete code changes from each coder. Review them against the task requirements.\n"
        "You may use read_file or exec for additional context if the diffs alone are insufficient.\n\n"
        "When ready, call submit() with:\n"
        "- merge_commits: include the full SHA for each coder whose diff is correct. "
        "Example: {\"coder_a\": [\"<full_sha>\"], \"coder_b\": [\"<full_sha>\"]}\n"
        "- request_rework: set to true if any coder needs to redo their work\n"
        "- coder_feedback: {\"coder_a\": \"specific feedback\"} for coders that need rework\n\n"
        "You MUST include SHAs in merge_commits for any correct implementation. Do not return empty merge_commits if a diff is good.\n"
    )

    return "\n".join(parts)


def _build_review_verify_prompt(context: Dict[str, Any], repo_ctx: Dict[str, Any]) -> str:
    round_num = context.get("round", "?")
    readme = repo_ctx.get("readme_task_md") or "(no task description available)"

    parts: List[str] = [
        f"# Integration Verification — Round {round_num}\n\n"
        + _time_budget_line(context)
        + f"## Task Requirements\n{readme}\n"
    ]

    harness_issues = context.get("harness_issues")
    if harness_issues:
        parts.append(
            "## Issues Detected\nThe following problems were detected after your review:\n"
            + "\n".join(f"- {issue}" for issue in harness_issues)
            + "\n\nYou MUST provide coder_feedback for each coder explaining what to fix.\n"
        )

    directive = context.get("reflection_directive")
    if directive:
        parts.append(f"## Reflection Directive\n{directive}\n")

    # Show test results (the main input for verify)
    last_val = context.get("last_public_validation", {})
    if isinstance(last_val, dict):
        stdout = last_val.get("stdout", "")
        stderr = last_val.get("stderr", "")
        val_text = ""
        if stdout:
            val_text += stdout[-3000:]
        if stderr:
            val_text += f"\n[stderr]\n{stderr[-2000:]}"
        if val_text.strip():
            parts.append(f"## Public Validation Results\n```\n{val_text.strip()}\n```\n")
        else:
            parts.append("## Public Validation Results\nNo test output available.\n")
    else:
        parts.append("## Public Validation Results\nNo test output available.\n")

    parts.append(
        "## Your Job\n"
        "The commits have already been merged. Verify the integrated result works correctly.\n"
        "Use read_file to inspect the merged code and exec to run tests.\n"
        "Call submit() with:\n"
        "- request_rework: true if the integration has problems\n"
        "- coder_feedback: {role: \"what to fix\"} if rework is needed\n"
        "- summary: your assessment\n"
    )

    return "\n".join(parts)


def _build_reflect_prompt(context: Dict[str, Any], _repo_ctx: Dict[str, Any]) -> str:
    round_num = context.get("round", "?")

    merged = context.get("merged_commits_this_round", {})
    merged_summary = ", ".join(f"{c}: {len(shas)}" for c, shas in merged.items()) if merged else "none"
    review_dec = context.get("review_decision_summary", {})
    rework = review_dec.get("request_rework", False)
    coder_fb = review_dec.get("coder_feedback", {})

    val = context.get("validation_result", {})
    val_ok = val.get("ok", False)
    val_stdout = val.get("stdout", "")[-2000:]
    val_stderr = val.get("stderr", "")[-2000:]

    knowledge = context.get("current_knowledge", {})

    return (
        f"# Reflection — Round {round_num}\n\n"
        + _time_budget_line(context)
        + f"## What Happened\n"
        f"- Merged: {merged_summary}\n"
        f"- Rework requested: {'yes' if rework else 'no'}\n"
        f"- Test result: {'ok' if val_ok else 'fail'}\n\n"
        f"## Test Output\n```\n{val_stdout}\n```\n"
        + (f"```\n[stderr]\n{val_stderr}\n```\n" if val_stderr.strip() else "")
        + f"\n## Coder Feedback Given\n{json.dumps(coder_fb, indent=2)}\n\n"
        f"## Current Knowledge Surfaces\n"
        f"- directive: {knowledge.get('directive', '(none)')}\n"
        f"- task_understanding: {knowledge.get('task_understanding', '(none)')}\n"
        f"- failure_patterns: {knowledge.get('failure_patterns', '(none)')}\n"
        f"- workflow_insights: {knowledge.get('workflow_insights', '(none)')}\n\n"
        "## Your Job\n"
        "Analyze what happened. Produce a directive for the next round.\n"
        "Call submit() with directive and updated knowledge surfaces.\n"
    )


def _build_coder_prompt(context: Dict[str, Any], repo_ctx: Dict[str, Any]) -> str:
    claimed = context.get("claimed_task", {})
    if isinstance(claimed, dict):
        title = claimed.get("title", "implementation task")
        acceptance = claimed.get("acceptance", "")
        paths = claimed.get("paths", [])
    else:
        title = "implementation task"
        acceptance = ""
        paths = []

    readme = repo_ctx.get("readme_task_md") or "(no task description available)"
    tracked = repo_ctx.get("tracked_files", [])
    tree = "\n".join(tracked[:200]) if tracked else "(no tracked files)"

    parts: List[str] = [f"# Assignment: {title}\n\n" + _time_budget_line(context)]

    parts.append(f"## Task Context\n{readme}\n")

    planner_summary = context.get("planner_summary", "")
    if planner_summary:
        phase = context.get("phase", "")
        plan_heading = "## Plan Reference" if phase == "rework" else "## Overall Plan"
        parts.append(f"{plan_heading}\n{planner_summary}\n")

    directive = context.get("reflection_directive")
    if directive:
        parts.append(f"## Reflection Directive\n{directive}\n")

    parts.append(f"## Your Assignment\n{title}\n")
    if acceptance:
        parts.append(f"Acceptance criteria: {acceptance}\n")
    if paths:
        parts.append(f"Assigned paths: {', '.join(paths)}\n")

    own_diff = context.get("own_diff", "")
    if own_diff:
        parts.append(f"## Your Current Changes\n```diff\n{own_diff}\n```\n")

    # Planner feedback
    feedback_by_role = context.get("planner_feedback_by_role", {})
    role = context.get("role", "")
    planner_feedback = feedback_by_role.get(role, "") if isinstance(feedback_by_role, dict) else ""
    if not planner_feedback:
        # Check for rework-specific context
        public_stdout = context.get("public_validate_stdout", "")
        public_stderr = context.get("public_validate_stderr", "")
        if public_stdout or public_stderr:
            parts.append("## Previous Validation Output\n")
            if public_stdout:
                parts.append(f"```\n{public_stdout[-2000:]}\n```\n")
            if public_stderr:
                parts.append(f"```\n[stderr] {public_stderr[-2000:]}\n```\n")
            parts.append("")
        parts.append("## Planner Feedback\nFirst implementation — no prior feedback.\n")
    else:
        parts.append(f"## Planner Feedback\n{planner_feedback}\n")
        public_stdout = context.get("public_validate_stdout", "")
        public_stderr = context.get("public_validate_stderr", "")
        if public_stdout or public_stderr:
            parts.append("## Previous Validation Output\n")
            if public_stdout:
                parts.append(f"```\n{public_stdout[-2000:]}\n```\n")
            if public_stderr:
                parts.append(f"```\n[stderr] {public_stderr[-2000:]}\n```\n")

    # Show the coder's own prior command failures (build/test errors from
    # previous rounds) so they can fix issues even when public_validate is off.
    prior_failures_by_role = context.get("prior_command_failures_by_role", {})
    prior_failures = prior_failures_by_role.get(role, []) if isinstance(prior_failures_by_role, dict) else []
    if prior_failures:
        parts.append("## Previous Command Failures\n")
        for fail in prior_failures:
            parts.append(f"Command: `{fail.get('command', '?')}`\n")
            parts.append(f"Exit code: {fail.get('exit_code', '?')}\n")
            stderr_tail = fail.get("stderr_tail", "")
            if stderr_tail:
                parts.append(f"```\n{stderr_tail}\n```\n")

    parts.append(f"## Repository Files\n```\n{tree}\n```\n")
    parts.append(
        "## Instructions\n"
        "Read the relevant files, implement the changes, run tests, then call submit().\n"
    )

    return "\n".join(parts)


def _build_finalize_prompt(context: Dict[str, Any], repo_ctx: Dict[str, Any]) -> str:
    readme = repo_ctx.get("readme_task_md") or "(no task description available)"
    metrics = context.get("metrics_so_far", {})
    review_rounds = metrics.get("review_round", 0)
    merge_conflicts = metrics.get("merge_conflicts", 0)

    return (
        "# Finalize\n\n"
        + _time_budget_line(context)
        + f"## Task Requirements\n{readme}\n\n"
        f"## Status\n"
        f"Review rounds: {review_rounds}, merge conflicts: {merge_conflicts}\n\n"
        "## Instructions\n"
        "Verify final coherence and ship readiness. Use read_file and exec to check.\n"
        "Call submit() when satisfied.\n"
    )


def _time_budget_line(context: Dict[str, Any]) -> str:
    deadline = context.get("run_deadline_epoch")
    if not deadline:
        return ""
    remaining = max(0, deadline - time.time())
    minutes = int(remaining // 60)
    if minutes >= 60:
        return f"> **Time budget: {minutes} minutes remaining.**\n\n"
    elif minutes >= 10:
        return f"> **Time budget: {minutes} minutes remaining. Be efficient.**\n\n"
    elif minutes > 0:
        return f"> **\u26a0 Time budget: {minutes} minutes remaining! Wrap up and submit soon.**\n\n"
    else:
        return f"> **\U0001f6a8 Time budget: EXPIRED. Submit immediately.**\n\n"


def _build_system_prompt(*, role: str, phase: str, context: Dict[str, Any]) -> str:
    if phase == "reflect":
        base = " ".join([TEAM_BASE_PROMPT, REFLECTION_PROMPT, f"Current phase: {phase}."])
    elif phase == "bootstrap":
        base = " ".join([TEAM_BASE_PROMPT, BOOTSTRAP_SYSTEM, f"Current phase: {phase}."])
    elif phase == "finalize":
        base = " ".join([TEAM_BASE_PROMPT, FINALIZE_SYSTEM, f"Current phase: {phase}."])
    elif role == "planner_reviewer" and phase in ("review", "review_verify"):
        base = " ".join([TEAM_BASE_PROMPT, REVIEW_SYSTEM, f"Current phase: {phase}."])
    else:
        # Coder phases
        role_prompt = GENERIC_CODER_ROLE_PROMPT
        phase_hint = f"Current phase: {phase}."
        base = " ".join([TEAM_BASE_PROMPT, role_prompt, phase_hint])

    deadline = context.get("run_deadline_epoch")
    if deadline:
        remaining = max(0, deadline - time.time())
        if remaining < 600:
            base += f" URGENT: Only {int(remaining // 60)} minutes left in the run. Finish and submit immediately."

    return base


def _collect_repo_context(*, worktree: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    task_readme = _safe_read_text(worktree / "public" / "README.task.md", max_chars=8000)
    if task_readme is None:
        task_readme = _safe_read_text(worktree / ".loopbench" / "public" / "README.task.md", max_chars=8000)
    tracked = _git_ls_files(worktree)
    return {
        "readme_task_md": task_readme,
        "tracked_files_count": len(tracked),
        "tracked_files": tracked[:200],
    }


# ---------------------------------------------------------------------------
# Phase / tool selection
# ---------------------------------------------------------------------------

def _tools_for_phase(*, role: str, phase: str) -> List[Dict[str, Any]]:
    if phase == "bootstrap":
        return COMMON_TOOLS + DOCS_TOOL + BOOTSTRAP_SUBMIT_TOOL
    if phase == "reflect":
        return COMMON_TOOLS + DOCS_TOOL + REFLECT_SUBMIT_TOOL
    if phase == "finalize":
        return COMMON_TOOLS + DOCS_TOOL + FINALIZE_SUBMIT_TOOL
    if role == "planner_reviewer":
        return COMMON_TOOLS + DOCS_TOOL + REVIEWER_TOOLS
    return COMMON_TOOLS + DOCS_TOOL + CODER_TOOLS


def _max_turns_for_phase(*, role: str, phase: str) -> int:
    if phase == "bootstrap":
        return MAX_TURNS_BOOTSTRAP
    if phase == "reflect":
        return MAX_TURNS_REFLECT
    if phase == "finalize":
        return MAX_TURNS_FINALIZE
    if role == "planner_reviewer":
        return MAX_TURNS_REVIEWER
    return MAX_TURNS_CODER


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def _run_agentic_loop(
    *,
    payload: Dict[str, Any],
    api_key: str,
    tools_schema: List[Dict[str, Any]],
    phase: str,
    role: str,
    max_turns: int,
    worktree: Path,
    context: Dict[str, Any],
    output_path: Path,
    model: str,
) -> Dict[str, Any]:
    messages = list(payload["messages"])
    all_tool_calls: List[Dict[str, Any]] = []
    inspected_commits: Dict[str, set] = {}
    total_usage = _zero_usage_metadata()
    conversation_turns: List[Dict[str, Any]] = []
    last_turn = max_turns
    last_response: Optional[Dict[str, Any]] = None

    for turn in range(1, max_turns + 1):
        last_turn = turn
        call_payload = {**payload, "messages": messages, "tools": tools_schema}
        call_payload.pop("response_format", None)  # not compatible with tools

        response = _call_openrouter(payload=call_payload, api_key=api_key)
        last_response = response
        _accumulate_usage(total_usage, response["usage"])

        choice = response["choice"]
        assistant_message = choice.get("message", {})
        if not isinstance(assistant_message, dict):
            assistant_message = {"role": "assistant", "content": response.get("reply_text", "")}

        messages.append(assistant_message)
        turn_tool_results: List[Dict[str, Any]] = []

        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            # No tool calls — model is done (implicit finish)
            break

        submit_result = None
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "").strip()
            try:
                fn_args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                fn_args = {}
            if not isinstance(fn_args, dict):
                fn_args = {}

            tc_id = tc.get("id", f"tc_{turn}_{fn_name}")

            if fn_name == "submit":
                submit_result = fn_args
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": "submit() accepted — session ending.",
                })
                all_tool_calls.append({"tool": fn_name, "args": fn_args, "result": "accepted"})
                turn_tool_results.append({"tool_call_id": tc_id, "name": "submit", "content": "accepted"})
                break

            result_text = _execute_tool(fn_name, fn_args, worktree, context, phase)

            if fn_name in ("git_show", "git_diff_files"):
                _track_inspection(inspected_commits, fn_args, context)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_text,
            })
            all_tool_calls.append({"tool": fn_name, "args": fn_args, "result": result_text})
            turn_tool_results.append({"tool_call_id": tc_id, "name": fn_name, "content": result_text})

        conversation_turns.append({
            "turn": turn,
            "assistant_message": assistant_message,
            "tool_results": turn_tool_results,
            "usage": response["usage"],
        })

        if submit_result is not None:
            try:
                _write_json_file(
                    _role_trace_path(output_path, "conversation", ".json"),
                    {
                        "role": role,
                        "phase": phase,
                        "model": model,
                        "continued_from_prior": bool(context.get("prior_conversation_messages")),
                        "system_message": messages[0] if messages and messages[0].get("role") == "system" else None,
                        "user_message": messages[1] if len(messages) > 1 and messages[1].get("role") == "user" else None,
                        "turns": conversation_turns,
                        "total_usage": total_usage,
                    },
                )
                _write_text_file(
                    _role_trace_path(output_path, "conversation", ".txt"),
                    _render_conversation_transcript(messages, conversation_turns, role, phase, model),
                )
            except Exception:  # noqa: BLE001
                pass
            # Write raw response text for tracing fallback
            try:
                _last_reply = last_response.get("reply_text", "") if last_response else ""
                _write_text_file(
                    _role_trace_path(output_path, "openrouter_response", ".txt"),
                    _last_reply,
                )
            except Exception:  # noqa: BLE001
                pass
            return _build_agentic_output(
                submit_args=submit_result,
                all_tool_calls=all_tool_calls,
                inspected_commits=inspected_commits,
                role=role,
                phase=phase,
                turn_count=turn,
                total_usage=total_usage,
                model=model,
                context=context,
            )

    # Budget exhausted or implicit finish
    try:
        _write_json_file(
            _role_trace_path(output_path, "conversation", ".json"),
            {
                "role": role,
                "phase": phase,
                "model": model,
                "continued_from_prior": bool(context.get("prior_conversation_messages")),
                "system_message": messages[0] if messages and messages[0].get("role") == "system" else None,
                "user_message": messages[1] if len(messages) > 1 and messages[1].get("role") == "user" else None,
                "turns": conversation_turns,
                "total_usage": total_usage,
            },
        )
        _write_text_file(
            _role_trace_path(output_path, "conversation", ".txt"),
            _render_conversation_transcript(messages, conversation_turns, role, phase, model),
        )
    except Exception:  # noqa: BLE001
        pass
    # Write raw response text for tracing fallback
    try:
        _last_reply = last_response.get("reply_text", "") if last_response else ""
        _write_text_file(
            _role_trace_path(output_path, "openrouter_response", ".txt"),
            _last_reply,
        )
    except Exception:  # noqa: BLE001
        pass
    return _build_fallback_output(
        messages=messages,
        all_tool_calls=all_tool_calls,
        inspected_commits=inspected_commits,
        role=role,
        phase=phase,
        turn_count=last_turn,
        total_usage=total_usage,
        model=model,
        context=context,
    )


def _build_agentic_output(
    *,
    submit_args: Dict[str, Any],
    all_tool_calls: List[Dict[str, Any]],
    inspected_commits: Dict[str, set],
    role: str,
    phase: str,
    turn_count: int,
    total_usage: Dict[str, int],
    model: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {
        "status": "completed",
        "role": role,
        "phase": phase,
        "model": model,
        "execution_mode": "agentic_tool_loop",
        "openrouter_structured_valid": True,
        "openrouter_turn_count": turn_count,
        "openrouter_usage": total_usage,
        "coordination_phase": context.get("coordination_phase"),
    }

    if phase == "bootstrap":
        output["plan_markdown"] = submit_args.get("plan_markdown", "")
        output["subtasks"] = submit_args.get("subtasks", [])
        output["summary"] = submit_args.get("summary", "bootstrap plan created")
        return output

    if phase == "reflect":
        output["directive"] = submit_args.get("directive", "")
        output["task_understanding"] = submit_args.get("task_understanding", "")
        output["failure_patterns"] = submit_args.get("failure_patterns", "")
        output["workflow_insights"] = submit_args.get("workflow_insights", "")
        output["superseded"] = submit_args.get("superseded", [])
        output["summary"] = submit_args.get("directive", "reflection complete")
        return output

    if role == "planner_reviewer" and phase in ("review", "review_verify"):
        merge_commits = submit_args.get("merge_commits", {})
        if not isinstance(merge_commits, dict):
            merge_commits = {}
        output["merge_commits"] = merge_commits
        output["request_rework"] = bool(submit_args.get("request_rework", False))
        coder_feedback = submit_args.get("coder_feedback", {})
        output["coder_feedback"] = coder_feedback if isinstance(coder_feedback, dict) else {}
        output["summary"] = submit_args.get("summary", f"review {phase} complete")
        # Pre-mark all inlined commits as inspected (diffs were in the prompt)
        if context.get("inline_diffs"):
            for coder_name, diff_entries in context["inline_diffs"].items():
                bucket = inspected_commits.setdefault(coder_name, set())
                for diff_entry in diff_entries:
                    sha = diff_entry.get("sha", "")
                    if sha and diff_entry.get("diff_content"):
                        bucket.add(sha)
        # Convert sets to sorted lists for JSON serialization
        output["inspected_commits"] = {
            r: sorted(commits) for r, commits in inspected_commits.items()
        }
        output["agentic_tool_calls"] = all_tool_calls
        output["intent_file_updates"] = []
        output["intent_patch"] = ""
        output["intent_run_commands"] = []
        return output

    # Coder or finalize phase
    output["summary"] = submit_args.get("summary", f"{role} {phase} complete")
    output["commit_message"] = submit_args.get("commit_message", f"{role}: {phase} update")
    output["intent_file_updates"] = []
    output["intent_patch"] = ""
    output["intent_run_commands"] = []
    output["changed"] = True
    output["agentic_tool_calls"] = all_tool_calls
    return output


def _build_fallback_output(
    *,
    messages: List[Dict[str, Any]],
    all_tool_calls: List[Dict[str, Any]],
    inspected_commits: Dict[str, set],
    role: str,
    phase: str,
    turn_count: int,
    total_usage: Dict[str, int],
    model: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    # Extract the last assistant text as a summary
    last_text = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                last_text = content.strip()
                break

    summary = last_text[:500] if last_text else f"{role} {phase} (budget exhausted)"

    # Synthesize submit_args based on phase
    if phase == "bootstrap":
        submit_args: Dict[str, Any] = {
            "plan_markdown": last_text or f"# Plan\n\nBudget exhausted before plan was submitted.",
            "subtasks": [],
            "summary": summary,
        }
    elif phase == "reflect":
        submit_args = {
            "directive": summary,
        }
    elif role == "planner_reviewer" and phase in ("review", "review_verify"):
        submit_args = {
            "summary": summary,
            "merge_commits": {},
            "request_rework": True,
            "coder_feedback": {},
        }
    else:
        # Coder or finalize
        submit_args = {
            "summary": summary,
            "commit_message": f"{role}: {phase} update (budget exhausted)",
        }

    return _build_agentic_output(
        submit_args=submit_args,
        all_tool_calls=all_tool_calls,
        inspected_commits=inspected_commits,
        role=role,
        phase=phase,
        turn_count=turn_count,
        total_usage=total_usage,
        model=model,
        context=context,
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _execute_tool(
    fn_name: str,
    fn_args: Dict[str, Any],
    worktree: Path,
    context: Dict[str, Any],
    phase: str,
) -> str:
    try:
        return _execute_tool_inner(fn_name, fn_args, worktree, context, phase)
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _git_available(worktree: Path) -> bool:
    """Check whether git commands can work in this worktree.

    E2B sandboxes use git worktrees whose .git file points to a host-side
    gitdir that doesn't exist inside the VM.  Detect this early so git
    tools can return a helpful message instead of wasting turns.
    """
    git_path = worktree / ".git"
    if git_path.is_dir():
        return True
    if git_path.is_file():
        try:
            content = git_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if content.startswith("gitdir:"):
            gitdir = content[len("gitdir:"):].strip()
            return Path(gitdir).exists()
    return False



def _execute_tool_inner(
    fn_name: str,
    fn_args: Dict[str, Any],
    worktree: Path,
    context: Dict[str, Any],
    phase: str,
) -> str:
    if fn_name == "lookup_docs":
        topic = fn_args.get("topic", "").lower().strip()
        return _lookup_docs(topic)

    if fn_name == "report_ambiguity":
        category = fn_args.get("category", "other")
        description = fn_args.get("description", "")
        resolution = fn_args.get("attempted_resolution", "")
        # Write to ambiguity reports directory
        import time as _time_mod
        report = {
            "ts": _time_mod.time(),
            "role": context.get("role", "unknown"),
            "phase": phase,
            "category": category,
            "description": description,
            "attempted_resolution": resolution,
        }
        reports_dir = worktree / ".loopbench" / "ambiguity_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"{int(_time_mod.time() * 1000)}.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        # Best-effort mirror to run artifacts (may be unreachable in sandboxed envs)
        run_dir_str = os.environ.get("LB_RUN_DIR", "")
        if run_dir_str:
            try:
                run_reports = Path(run_dir_str) / "ambiguity_reports"
                run_reports.mkdir(parents=True, exist_ok=True)
                (run_reports / report_file.name).write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
            except OSError:
                pass  # host path not mounted in sandbox
        return f"Ambiguity reported: [{category}] {description[:100]}"

    policy = _get_command_policy()
    default_timeout = policy["command_timeout_sec"]

    if fn_name == "list_files":
        raw_path = fn_args.get("path", ".")
        target = _resolve_within_worktree(worktree, raw_path)
        if not target.is_dir():
            return f"error: not a directory: {raw_path}"
        entries: List[str] = []
        try:
            for item in sorted(target.iterdir()):
                if item.name == ".git":
                    continue
                try:
                    rel = item.relative_to(worktree).as_posix()
                except ValueError:
                    rel = item.name
                if item.is_dir():
                    entries.append(f"{rel}/")
                else:
                    try:
                        size = item.stat().st_size
                    except OSError:
                        size = 0
                    entries.append(f"{rel}  ({size} bytes)")
        except OSError as exc:
            return f"error: {exc}"
        return _truncate_output("\n".join(entries))

    if fn_name == "read_file":
        raw_path = fn_args.get("path", "")
        if not raw_path:
            return "error: path is required"
        target = _resolve_within_worktree(worktree, raw_path)
        if not target.is_file():
            return f"error: file not found: {raw_path}"
        try:
            all_lines = target.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except OSError as exc:
            return f"error: {exc}"
        offset = int(fn_args.get("offset", 0))
        limit = fn_args.get("limit")
        if offset > 0:
            all_lines = all_lines[offset:]
        if limit is not None:
            all_lines = all_lines[: int(limit)]
        content = "".join(all_lines)
        return _truncate_output(content)

    if fn_name == "search_files":
        pattern = fn_args.get("pattern", "")
        if not pattern:
            return "error: pattern is required"
        raw_path = fn_args.get("path", ".")
        target = _resolve_within_worktree(worktree, raw_path)
        if not target.is_dir():
            return f"error: not a directory: {raw_path}"
        cmd = ["grep", "-rn", "--include=*", "-E", pattern, str(target)]
        include = fn_args.get("include")
        if include:
            cmd = ["grep", "-rn", f"--include={include}", "-E", pattern, str(target)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "error: search timed out"
        if proc.returncode == 2:
            return f"error: invalid pattern: {proc.stderr.strip()}"
        output = proc.stdout
        if not output.strip():
            return "(no matches)"
        # Make paths relative to worktree
        worktree_prefix = str(worktree) + "/"
        if worktree_prefix in output:
            output = output.replace(worktree_prefix, "")
        # Limit output
        lines = output.splitlines()
        if len(lines) > 100:
            output = "\n".join(lines[:100]) + f"\n\n... [{len(lines) - 100} more matches truncated]"
        return _truncate_output(output)

    if fn_name == "write_file":
        raw_path = fn_args.get("path", "")
        file_content = fn_args.get("content", "")
        if not raw_path:
            return "error: path is required"
        target = _resolve_within_worktree(worktree, raw_path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file_content, encoding="utf-8")
        except OSError as exc:
            return f"error: {exc}"
        return f"wrote {len(file_content)} chars to {raw_path}"

    if fn_name == "apply_patch":
        patch_text = fn_args.get("patch_text", "")
        if not patch_text:
            return "error: patch_text is required"
        if not patch_text.endswith("\n"):
            patch_text += "\n"
        patch_text = _fix_patch_hunk_counts(patch_text)
        try:
            proc = subprocess.run(
                ["git", "apply", "--allow-empty", "-"],
                input=patch_text,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "error: git apply timed out"
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            return f"error: git apply failed (exit {proc.returncode}): {stderr}"
        return "patch applied successfully"

    if fn_name == "exec":
        command = fn_args.get("command", "")
        if not command:
            return "error: command is required"
        timeout_sec = int(fn_args.get("timeout_sec", default_timeout))
        timeout_sec = max(5, min(timeout_sec, default_timeout))
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout_sec}s"
        parts: List[str] = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(f"[stderr]\n{proc.stderr}")
        if proc.returncode != 0:
            parts.append(f"[exit code: {proc.returncode}]")
        output = "\n".join(parts) if parts else "(no output)"
        return _truncate_output(output)

    if fn_name in ("git_status", "git_diff", "git_log", "git_show", "git_diff_files"):
        if not _git_available(worktree):
            return "error: git is not available in this sandbox."

    if fn_name == "git_status":
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "error: git status timed out"
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return _truncate_output(output) if output.strip() else "(clean working tree)"

    if fn_name == "git_diff":
        ref = fn_args.get("ref", "HEAD")
        cmd = ["git", "diff", ref]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "error: git diff timed out"
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return _truncate_output(output) if output.strip() else "(no diff)"

    if fn_name == "git_log":
        ref = fn_args.get("ref", "HEAD")
        max_count = int(fn_args.get("max_count", 20))
        max_count = max(1, min(max_count, 100))
        cmd = ["git", "log", f"--max-count={max_count}", "--oneline", ref]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "error: git log timed out"
        output = proc.stdout
        if proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return _truncate_output(output) if output.strip() else "(no commits)"

    if fn_name == "git_show":
        commit_sha = fn_args.get("commit_sha", "")
        if not commit_sha:
            return "error: commit_sha is required"
        cmd = ["git", "show", "--stat", "--patch", commit_sha]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(worktree), timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "error: git show timed out"
        if proc.returncode == 0:
            return _truncate_output(proc.stdout) if proc.stdout.strip() else f"(no output for {commit_sha})"
        return f"error: git show failed for {commit_sha}: {proc.stderr.strip()}"

    if fn_name == "git_diff_files":
        commit_sha = fn_args.get("commit_sha", "")
        if not commit_sha:
            return "error: commit_sha is required"
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(worktree), timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "error: git diff-tree timed out"
        if proc.returncode == 0:
            return _truncate_output(proc.stdout) if proc.stdout.strip() else f"(no files changed in {commit_sha})"
        return f"error: git diff-tree failed for {commit_sha}: {proc.stderr.strip()}"

    return f"error: unknown tool '{fn_name}'"


def _resolve_within_worktree(worktree: Path, raw_path: str) -> Path:
    root = worktree.resolve()
    path = Path(raw_path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"path escapes worktree root: {raw_path}")
    return resolved


# ---------------------------------------------------------------------------
# Patch hunk count fixer (copied from loopbench/tools.py)
# ---------------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)")


def _fix_patch_hunk_counts(patch_text: str) -> str:
    """Recalculate hunk header line counts in a unified diff.

    LLMs frequently produce correct patch content but wrong ``@@`` line counts,
    causing ``git apply`` to reject the patch as "corrupt".  This function
    re-parses each hunk, counts actual old/new lines, and rewrites the header.
    """
    trailing_nl = patch_text.endswith("\n")
    text = patch_text.rstrip("\n")
    lines = text.split("\n")
    out: list[str] = []
    hunk_start_idx: int | None = None
    hunk_lines: list[str] = []
    old_start = new_start = 0
    hunk_tail = ""

    def _flush_hunk() -> None:
        nonlocal hunk_start_idx, hunk_lines, old_start, new_start, hunk_tail
        if hunk_start_idx is None:
            return
        old_count = 0
        new_count = 0
        for hl in hunk_lines:
            if hl.startswith("-"):
                old_count += 1
            elif hl.startswith("+"):
                new_count += 1
            else:
                # context line (space prefix) or empty-string representing blank context
                old_count += 1
                new_count += 1
        header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{hunk_tail}"
        out.append(header)
        out.extend(hunk_lines)
        hunk_start_idx = None
        hunk_lines = []

    for idx, line in enumerate(lines):
        m = _HUNK_RE.match(line)
        if m:
            _flush_hunk()
            old_start = int(m.group(1))
            new_start = int(m.group(3))
            hunk_tail = m.group(5) or ""
            hunk_start_idx = idx
            continue
        if hunk_start_idx is not None:
            if line.startswith(("---", "+++")):
                _flush_hunk()
                out.append(line)
            else:
                hunk_lines.append(line)
            continue
        out.append(line)

    _flush_hunk()
    result = "\n".join(out)
    if trailing_nl:
        result += "\n"
    return result


def _truncate_output(text: str, *, max_chars: int = OUTPUT_TRUNCATE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [{len(text) - max_chars} chars truncated] ...\n\n" + text[-half:]


def _track_inspection(
    inspected_commits: Dict[str, set],
    fn_args: Dict[str, Any],
    context: Dict[str, Any],
) -> None:
    commit_sha = fn_args.get("commit_sha", "")
    if not commit_sha:
        return
    # Determine which role this commit belongs to by checking coder_commits
    coder_commits = context.get("coder_commits", {})
    for role_name, commits in coder_commits.items():
        if isinstance(commits, list) and commit_sha in commits:
            bucket = inspected_commits.setdefault(role_name, set())
            bucket.add(commit_sha)
            return
    # Also check candidate_merge_commits (may be structured differently)
    candidates = context.get("candidate_merge_commits", {})
    if isinstance(candidates, dict):
        for role_name, commit_list in candidates.items():
            if not isinstance(commit_list, list):
                continue
            shas: set = set()
            for item in commit_list:
                if isinstance(item, str):
                    shas.add(item)
                elif isinstance(item, dict):
                    sha = item.get("sha", item.get("commit", ""))
                    if isinstance(sha, str):
                        shas.add(sha)
            if commit_sha in shas:
                bucket = inspected_commits.setdefault(role_name, set())
                bucket.add(commit_sha)
                return


# ---------------------------------------------------------------------------
# Kept utility functions
# ---------------------------------------------------------------------------


def _reasoning_payload_from_env() -> Dict[str, Any] | None:
    payload: Dict[str, Any] = {}
    for name, key in (
        ("OPENROUTER_REASONING_ENABLED", "enabled"),
        ("OPENROUTER_REASONING_EXCLUDE", "exclude"),
    ):
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            payload[key] = True
        elif value in {"0", "false", "no", "off"}:
            payload[key] = False

    effort = (os.environ.get("OPENROUTER_REASONING_EFFORT") or "").strip()
    if effort:
        payload["effort"] = effort

    raw_max_tokens = (os.environ.get("OPENROUTER_REASONING_MAX_TOKENS") or "").strip()
    if raw_max_tokens:
        try:
            payload["max_tokens"] = max(0, min(65536, int(raw_max_tokens)))
        except ValueError:
            pass

    return payload or None


def _get_command_policy() -> Dict[str, Any]:
    backend = os.environ.get("LOOPBENCH_SANDBOX_BACKEND", "").strip().lower()
    if backend == "e2b_firecracker":
        default_max_commands = 12
        default_timeout_sec = 1200
    else:
        default_max_commands = 8
        default_timeout_sec = 180

    max_commands = _read_int_env(
        name="LOOPBENCH_MAX_COMMANDS",
        default=default_max_commands,
        min_value=1,
        max_value=20,
    )
    command_timeout_sec = _read_int_env(
        name="LOOPBENCH_COMMAND_TIMEOUT_SEC",
        default=default_timeout_sec,
        min_value=30,
        max_value=3600,
    )

    return {
        "max_commands": max_commands,
        "command_timeout_sec": command_timeout_sec,
    }


def _read_int_env(*, name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def _resolve_openrouter_api_key() -> str:
    key_env = os.environ.get("OPENROUTER_API_KEY_ENV") or os.environ.get("OPEN_ROUTER_API_KEY_ENV")
    if key_env:
        value = os.environ.get(key_env)
        if value:
            return value

    for name in ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _call_openrouter(*, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    http_retries = _read_int_env(
        name="OPENROUTER_HTTP_RETRIES",
        default=DEFAULT_OPENROUTER_HTTP_RETRIES,
        min_value=1,
        max_value=10,
    )
    http_timeout_sec = _read_int_env(
        name="OPENROUTER_HTTP_TIMEOUT_SEC",
        default=DEFAULT_OPENROUTER_HTTP_TIMEOUT_SEC,
        min_value=10,
        max_value=600,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    app_title = os.environ.get("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title

    request_payload = _serialize_payload_for_openrouter(payload)
    body = json.dumps(request_payload).encode("utf-8")
    err_text = ""

    for attempt in range(1, http_retries + 1):
        req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=http_timeout_sec) as resp:
                response = json.loads(resp.read().decode("utf-8", errors="replace"))
                choice = _first_choice(response)
                return {
                    "reply_text": _extract_openrouter_reply_text(response),
                    "choice": choice,
                    "usage": _normalize_usage_metadata(response.get("usage")),
                    "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
                    "native_finish_reason": (
                        choice.get("native_finish_reason") if isinstance(choice, dict) else None
                    ),
                    "response_shape": _summarize_openrouter_response_shape(response),
                }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            err_text = f"http {exc.code}: {raw}"
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after:
                    try:
                        wait = min(float(retry_after), 120.0)
                    except (ValueError, TypeError):
                        wait = min(2 ** attempt, 60) + random.uniform(0, 2)
                else:
                    wait = min(2 ** attempt, 60) + random.uniform(0, 2)
                time.sleep(wait)
                continue
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
        # Exponential backoff with jitter to avoid thundering herd.
        time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))

    raise RuntimeError(f"openrouter request failed after retries ({http_retries}): {err_text}")


def _serialize_payload_for_openrouter(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep artifact payloads structured while preserving provider compatibility:
    OpenRouter expects message content as text, so non-string content is JSON-encoded.
    """
    request_payload = json.loads(json.dumps(payload))
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return request_payload

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            continue
        try:
            message["content"] = json.dumps(content, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            message["content"] = str(content)
    return request_payload


def _extract_openrouter_reply_text(response: Dict[str, Any]) -> str:
    choice = _first_choice(response)
    if not isinstance(choice, dict):
        return ""

    message = choice.get("message")
    if isinstance(message, dict):
        parsed = message.get("parsed")
        if isinstance(parsed, dict) and parsed:
            return json.dumps(parsed, ensure_ascii=False)

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, dict):
            for key in ("text", "content", "output_text", "value"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                for key in ("text", "content", "output_text", "value"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
            if parts:
                return "\n".join(parts).strip()

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                fn = item.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str) and args.strip():
                    return args.strip()

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            return refusal.strip()

        fallback = json.dumps(message, ensure_ascii=False)
        return fallback.strip()

    text = choice.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    return ""


def _first_choice(response: Dict[str, Any]) -> Dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    return first if isinstance(first, dict) else {}


def _summarize_openrouter_response_shape(response: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    choice = _first_choice(response)
    if not choice:
        return summary

    summary["finish_reason"] = choice.get("finish_reason")
    summary["native_finish_reason"] = choice.get("native_finish_reason")

    message = choice.get("message")
    if isinstance(message, dict):
        summary["message_keys"] = sorted(str(key) for key in message.keys())
        content = message.get("content")
        summary["content_type"] = type(content).__name__
        if isinstance(content, list):
            summary["content_len"] = len(content)
            summary["content_item_types"] = [
                (
                    str(item.get("type"))
                    if isinstance(item, dict) and item.get("type") is not None
                    else ("dict" if isinstance(item, dict) else type(item).__name__)
                )
                for item in content[:12]
            ]
        elif isinstance(content, str):
            summary["content_len"] = len(content)
        summary["has_tool_calls"] = isinstance(message.get("tool_calls"), list)

    return summary


def _zero_usage_metadata() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _normalize_usage_metadata(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, dict):
        return _zero_usage_metadata()

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


def _accumulate_usage(target: Dict[str, int], delta: Dict[str, int]) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        target[key] = int(target.get(key, 0)) + int(delta.get(key, 0))


def _read_non_negative_int(*values: Any) -> int:
    for value in values:
        try:
            parsed = int(value)
        except Exception:  # noqa: BLE001
            continue
        if parsed >= 0:
            return parsed
    return 0





def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _read_json_file(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"context JSON must be an object: {path}")
    return data


def _render_conversation_transcript(
    messages: List[Dict[str, Any]],
    conversation_turns: List[Dict[str, Any]],
    role: str,
    phase: str,
    model: str,
) -> str:
    """Render a human-readable transcript of the agentic conversation."""
    lines: List[str] = []
    lines.append(f"{role} / {phase}  ({model})")
    lines.append("")

    # System and user prompts (truncated)
    for msg in messages[:2]:
        msg_role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if len(content) > 3000:
            content = content[:3000] + "\n... (truncated)"
        lines.append(f"[{msg_role}]")
        lines.append(content)
        lines.append("")

    lines.append("---")
    lines.append("")

    for turn_data in conversation_turns:
        turn_num = turn_data.get("turn", 0)
        assistant_msg = turn_data.get("assistant_message", {})
        tool_results = turn_data.get("tool_results", [])
        usage = turn_data.get("usage", {})

        tokens = usage.get("total_tokens", 0)
        lines.append(f"[turn {turn_num}]  ({tokens} tok)")

        # Assistant reasoning text
        content = ""
        if isinstance(assistant_msg, dict):
            content = assistant_msg.get("content") or ""
        if content:
            lines.append(content.strip())
            lines.append("")

        # Tool calls and results
        tool_calls = assistant_msg.get("tool_calls", []) if isinstance(assistant_msg, dict) else []
        result_map = {tr.get("tool_call_id"): tr for tr in tool_results}

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "?")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                args = raw_args

            # Compact arg display
            if isinstance(args, dict):
                if fn_name in ("read_file", "list_files"):
                    arg_str = args.get("path", str(args))
                elif fn_name == "write_file":
                    path = args.get("path", "?")
                    content_len = len(args.get("content", ""))
                    arg_str = f"{path}  ({content_len} chars)"
                elif fn_name == "exec":
                    arg_str = args.get("command", str(args))
                elif fn_name == "submit":
                    arg_str = (args.get("summary") or "")[:120]
                elif fn_name == "apply_patch":
                    patch = args.get("patch_text", "")
                    arg_str = f"({len(patch)} chars)"
                elif fn_name == "lookup_docs":
                    arg_str = args.get("topic", str(args))
                else:
                    arg_str = json.dumps(args, ensure_ascii=False)
                    if len(arg_str) > 200:
                        arg_str = arg_str[:200] + "..."
            else:
                arg_str = str(args)[:200]

            lines.append(f"  > {fn_name}({arg_str})")

            # Result
            tc_id = tc.get("id", "")
            tr = result_map.get(tc_id, {})
            result_content = tr.get("content", "")
            if result_content:
                # Truncate long results but keep enough to be useful
                if len(result_content) > 800:
                    result_content = result_content[:800] + "\n    ... (truncated)"
                for rline in result_content.split("\n"):
                    lines.append(f"    {rline}")
            lines.append("")

        lines.append("")

    return "\n".join(lines)


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _role_trace_path(output_path: Path, suffix: str, extension: str) -> Path:
    stem = output_path.stem
    if stem.endswith("_output"):
        stem = stem[:-7]
    return output_path.with_name(f"{stem}_{suffix}{extension}")



def _safe_read_text(path: Path, *, max_chars: int) -> str | None:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    if len(data) > max_chars:
        return data[:max_chars] + "\n...[truncated]..."
    return data


def _git_ls_files(worktree: Path) -> List[str]:
    out: List[str] = []
    for path in sorted(worktree.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        try:
            rel = path.relative_to(worktree).as_posix()
        except Exception:  # noqa: BLE001
            continue
        if rel.startswith(".loopbench/"):
            continue
        out.append(rel)
    return out


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        lb_output = os.environ.get("LB_OUTPUT_JSON")
        if lb_output:
            payload = {
                "status": "error",
                "error": str(exc),
                "role": os.environ.get("LB_ROLE"),
                "phase": os.environ.get("LB_PHASE"),
            }
            Path(lb_output).parent.mkdir(parents=True, exist_ok=True)
            Path(lb_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
