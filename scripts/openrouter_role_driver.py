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
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "moonshotai/kimi-k2.5"
DEFAULT_OPENROUTER_HTTP_RETRIES = 4
DEFAULT_OPENROUTER_HTTP_TIMEOUT_SEC = 90

OUTPUT_TRUNCATE_CHARS = 10000
MAX_TURNS_CODER = 30
MAX_TURNS_REVIEWER = 25
MAX_TURNS_BOOTSTRAP = 5
MAX_TURNS_REFLECT = 5
MAX_TURNS_FINALIZE = 10


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

PLANNER_ROLE_PROMPT = (
    "Role: planner_reviewer. You own decomposition, review guidance, and final coherence. "
    "In bootstrap, read the task README and repo structure, then call submit() with your plan and subtasks. "
    "In review (review_stage=select), use git_show and git_diff_files to inspect each coder commit before nominating merges. "
    "You MUST inspect every commit you intend to merge — uninspected nominations will be dropped. "
    "Use exec to run tests on the codebase. "
    "Call submit() with merge_commits, request_rework, and coder_feedback. "
    "Use commit SHAs from coder_commits in the prompt; do not use symbolic tokens like HEAD. "
    "In review_verify (review_stage=verify), run tests on the integrated candidate and decide whether rework is needed. "
    "In finalize, focus on final coherence and ship readiness."
)

GENERIC_CODER_ROLE_PROMPT = (
    "Role: coder (applies equally to coder_a and coder_b). "
    "You execute only assigned work from planner-reviewer messages/assignment payload. "
    "Do not invent a new work split. If assignment is missing/ambiguous, report blocker in notes. "
    "Use read_file to understand existing code before making changes. "
    "Use write_file or apply_patch to modify files. "
    "Use exec to run tests after changes. "
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
            "name": "git_show",
            "description": "Show the full diff for a specific commit SHA",
            "parameters": {
                "type": "object",
                "properties": {
                    "commit_sha": {"type": "string", "description": "Commit SHA to inspect"}
                },
                "required": ["commit_sha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff_files",
            "description": "List files changed in a specific commit",
            "parameters": {
                "type": "object",
                "properties": {
                    "commit_sha": {"type": "string", "description": "Commit SHA to inspect"}
                },
                "required": ["commit_sha"],
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
                                "role": {"type": "string"},
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
    system_prompt = _build_system_prompt(role=role, phase=phase)

    prompt_keys = (
        "reflection_directive",
        "review_stage",
        "task_id",
        "round",
        "coordination_phase",
        "assignment",
        "claimed_task",
        "public_validate_stderr",
        "public_validate_stdout",
        "planner_summary",
        "coder_commits",
        "candidate_merge_commits",
        "latest_coder_outputs",
        "implementation_messages",
        "last_public_validation",
        "coordination_summary",
        "review_diff_tool",
        "knowledge_tool",
        "validation_result",
        "review_decision_summary",
        "coder_output_summaries",
        "merged_commits_this_round",
        "current_knowledge",
    )
    user_prompt = {
        "role": role,
        "phase": phase,
        **{key: context.get(key) for key in prompt_keys},
        "repo_context": repo_ctx,
    }

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
    # Do NOT set response_format — tools and json_object are mutually exclusive
    reasoning = _reasoning_payload_from_env()
    if reasoning is not None:
        payload["reasoning"] = reasoning
    return payload


def _build_system_prompt(*, role: str, phase: str) -> str:
    if phase == "reflect":
        return " ".join([TEAM_BASE_PROMPT, REFLECTION_PROMPT, f"Current phase: {phase}."])
    role_prompt = PLANNER_ROLE_PROMPT if role == "planner_reviewer" else GENERIC_CODER_ROLE_PROMPT
    phase_hint = f"Current phase: {phase}."
    return " ".join([TEAM_BASE_PROMPT, role_prompt, phase_hint])


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
        return COMMON_TOOLS + BOOTSTRAP_SUBMIT_TOOL
    if phase == "reflect":
        return COMMON_TOOLS + REFLECT_SUBMIT_TOOL
    if phase == "finalize":
        return COMMON_TOOLS + FINALIZE_SUBMIT_TOOL
    if role == "planner_reviewer":
        return COMMON_TOOLS + REVIEWER_TOOLS
    return COMMON_TOOLS + CODER_TOOLS


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
    last_turn = max_turns

    for turn in range(1, max_turns + 1):
        last_turn = turn
        call_payload = {**payload, "messages": messages, "tools": tools_schema}
        call_payload.pop("response_format", None)  # not compatible with tools

        response = _call_openrouter(payload=call_payload, api_key=api_key)
        _accumulate_usage(total_usage, response["usage"])

        choice = response["choice"]
        assistant_message = choice.get("message", {})
        if not isinstance(assistant_message, dict):
            assistant_message = {"role": "assistant", "content": response.get("reply_text", "")}

        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            # No tool calls — model is done (implicit finish)
            break

        submit_result = None
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
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
                break

            result_text = _execute_tool(fn_name, fn_args, worktree, context, phase)

            if fn_name in ("git_show", "git_diff_files"):
                _track_inspection(inspected_commits, fn_name, fn_args, context)

            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_text,
            })
            all_tool_calls.append({"tool": fn_name, "args": fn_args, "result": result_text})

        if submit_result is not None:
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


_GIT_UNAVAILABLE_HINT = (
    "error: git is not available in this sandbox (worktree gitdir points to host filesystem). "
    "Use exec() with review_diff_tool.py to inspect commits instead. Example:\n"
    "  exec({\"command\": \"python .loopbench/artifacts/review_diffs/review_diff_tool.py "
    "--manifest .loopbench/artifacts/review_diffs/round_N/manifest.json list\"})\n"
    "  exec({\"command\": \"python .loopbench/artifacts/review_diffs/review_diff_tool.py "
    "--manifest .loopbench/artifacts/review_diffs/round_N/manifest.json show "
    "--coder CODER --sha COMMIT_SHA\"})\n"
    "  exec({\"command\": \"python .loopbench/artifacts/review_diffs/review_diff_tool.py "
    "--manifest .loopbench/artifacts/review_diffs/round_N/manifest.json files "
    "--coder CODER --sha COMMIT_SHA\"})\n"
    "Replace round_N with the current review round number from your context."
)


def _execute_tool_inner(
    fn_name: str,
    fn_args: Dict[str, Any],
    worktree: Path,
    context: Dict[str, Any],
    phase: str,
) -> str:
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
            return _GIT_UNAVAILABLE_HINT

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
                cmd,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return "error: git show timed out"
        output = proc.stdout
        if proc.returncode != 0 and proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return _truncate_output(output) if output.strip() else f"(no output for {commit_sha})"

    if fn_name == "git_diff_files":
        commit_sha = fn_args.get("commit_sha", "")
        if not commit_sha:
            return "error: commit_sha is required"
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(worktree),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "error: git diff-tree timed out"
        output = proc.stdout
        if proc.returncode != 0 and proc.stderr:
            output += f"\n[stderr]\n{proc.stderr}"
        return _truncate_output(output) if output.strip() else f"(no files changed in {commit_sha})"

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
    fn_name: str,
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

def _is_planner_review_phase(phase: str) -> bool:
    phase_name = str(phase or "").strip().lower()
    return phase_name in {"review", "review_verify"}


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


def _parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    candidates = [text]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return {}


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


def _safe_text(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


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
