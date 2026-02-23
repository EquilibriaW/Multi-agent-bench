"""
loopbench.review_logic

Shared review-phase parsing and commit-selection helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import shlex
from typing import Any, Callable, Dict, List, Optional, Set


ResolveSymbolicCommit = Callable[[str, str], Optional[str]]


class PublicValidationState(str, Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE_NOOP = "unavailable_noop"
    SKIPPED_POLICY_OFF = "skipped_policy_off"
    UNAVAILABLE_NO_COMMAND = "unavailable_no_command"


@dataclass(frozen=True)
class PublicValidationRecord:
    policy: str
    state: PublicValidationState
    ok: bool = False
    stdout: str = ""
    stderr: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "state": self.state.value,
            "policy": self.policy,
        }

    @property
    def noop(self) -> bool:
        return is_noop_public_validation(self.stdout)


@dataclass(frozen=True)
class PublicValidationSummary:
    passed: bool
    available: bool
    noop: bool
    last_validation: PublicValidationRecord


@dataclass
class ReviewDecision:
    merge_commits_by_role: Dict[str, List[str]] = field(default_factory=dict)
    unresolved_merge_tokens_by_role: Dict[str, List[str]] = field(default_factory=dict)
    uninspected_nominated_commits_by_role: Dict[str, List[str]] = field(default_factory=dict)
    invalid_merge_roles: List[str] = field(default_factory=list)
    coder_feedback: Dict[str, str] = field(default_factory=dict)
    merge_commits_provided: bool = False
    merge_commits_valid: bool = True
    request_rework: bool = False
    dynamic_checks_ran: bool = False


def extract_review_decision(
    *,
    planner_output: Dict[str, Any],
    available_commits_by_role: Dict[str, List[str]],
    coder_roles: Set[str],
    resolve_symbolic_commit: ResolveSymbolicCommit | None = None,
) -> ReviewDecision:
    decision = ReviewDecision()
    if not isinstance(planner_output, dict):
        return decision

    decision.dynamic_checks_ran = planner_ran_dynamic_checks(planner_output)
    decision.request_rework = bool(planner_output.get("request_rework"))
    decision.coder_feedback = normalize_coder_feedback(
        planner_output.get("coder_feedback"),
        allowed_roles=coder_roles,
    )

    raw_merge = planner_output.get("merge_commits")
    if raw_merge is None:
        return decision
    decision.merge_commits_provided = True
    if not isinstance(raw_merge, dict):
        decision.merge_commits_valid = False
        return decision

    for role, raw_commits in raw_merge.items():
        role_name = str(role or "").strip()
        if role_name not in coder_roles:
            decision.merge_commits_valid = False
            if role_name and role_name not in decision.invalid_merge_roles:
                decision.invalid_merge_roles.append(role_name)
            continue
        commits = available_commits_by_role.get(role_name, [])
        if not commits:
            continue
        selected, unresolved = resolve_commit_selection(
            raw_commits,
            commits,
            role=role_name,
            resolve_symbolic_commit=resolve_symbolic_commit,
        )
        if selected:
            decision.merge_commits_by_role[role_name] = selected
        if unresolved:
            decision.unresolved_merge_tokens_by_role[role_name] = unresolved
    return decision


def planner_ran_dynamic_checks(output: Dict[str, Any]) -> bool:
    commands_run = output.get("commands_run")
    if isinstance(commands_run, list):
        parseable_commands = 0
        for item in commands_run:
            if not isinstance(item, dict):
                continue
            cmd = item.get("cmd")
            if not isinstance(cmd, str):
                continue
            parseable_commands += 1
            if is_review_diff_tool_command(cmd):
                continue
            return True
        if parseable_commands > 0:
            return False
    attempts = output.get("run_commands_attempted")
    return isinstance(attempts, int) and attempts > 0


def is_review_diff_tool_command(command: str) -> bool:
    command_text = command.strip()
    if not command_text:
        return False
    if "review_diff_tool.py" not in command_text:
        return False
    return " show " in f" {command_text} " or " files " in f" {command_text} "


def parse_review_diff_command(command: str, *, coder_roles: Set[str]) -> Optional[Dict[str, str]]:
    if "review_diff_tool.py" not in command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None

    if "show" not in parts and "files" not in parts:
        return None

    coder = ""
    sha = ""
    for idx, part in enumerate(parts):
        if part == "--coder" and idx + 1 < len(parts):
            coder = parts[idx + 1].strip()
        if part == "--sha" and idx + 1 < len(parts):
            sha = parts[idx + 1].strip()
    if not coder or not sha:
        return None
    if coder not in coder_roles:
        return None
    return {"coder": coder, "sha": sha}


def inspected_review_commits_by_role(
    *,
    planner_output: Dict[str, Any],
    available_commits_by_role: Dict[str, List[str]],
    coder_roles: Set[str],
    resolve_symbolic_commit: ResolveSymbolicCommit | None = None,
) -> Dict[str, Set[str]]:
    commands_run = planner_output.get("commands_run")
    if not isinstance(commands_run, list):
        commands_run = []

    inspected_by_role: Dict[str, Set[str]] = {}
    for item in commands_run:
        if not isinstance(item, dict):
            continue
        if not bool(item.get("ok")):
            continue
        parsed = parse_review_diff_command(
            str(item.get("cmd") or ""),
            coder_roles=coder_roles,
        )
        if parsed is None:
            continue
        role_name = parsed["coder"]
        available = available_commits_by_role.get(role_name, [])
        if not available:
            continue
        selected, _ = resolve_commit_selection(
            [parsed["sha"]],
            available,
            role=role_name,
            resolve_symbolic_commit=resolve_symbolic_commit,
        )
        if not selected:
            continue
        bucket = inspected_by_role.setdefault(role_name, set())
        bucket.update(selected)
    return inspected_by_role


def nominated_commits_without_review_inspection(
    *,
    inspected_commits_by_role: Dict[str, Set[str]],
    nominated_commits_by_role: Dict[str, List[str]],
    already_merged_commits_by_role: Dict[str, Set[str]],
) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for role_name, nominated in nominated_commits_by_role.items():
        if not nominated:
            continue
        already_merged = already_merged_commits_by_role.get(role_name, set())
        inspected = inspected_commits_by_role.get(role_name, set())
        unresolved = [sha for sha in nominated if sha not in already_merged and sha not in inspected]
        if unresolved:
            missing[role_name] = unresolved
    return missing


def verification_command_stats(planner_output: Dict[str, Any]) -> Dict[str, int]:
    commands_run = planner_output.get("commands_run")
    if not isinstance(commands_run, list):
        return {"attempted": 0, "succeeded": 0, "failed": 0}

    attempted = 0
    succeeded = 0
    failed = 0
    for item in commands_run:
        if not isinstance(item, dict):
            continue
        cmd = item.get("cmd")
        if not isinstance(cmd, str):
            continue
        if is_review_diff_tool_command(cmd):
            continue
        attempted += 1
        if bool(item.get("ok")):
            succeeded += 1
        else:
            failed += 1
    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
    }


def normalize_coder_feedback(raw: Any, *, allowed_roles: Set[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(raw, dict):
        for role, msg in raw.items():
            role_name = str(role or "").strip()
            if role_name not in allowed_roles:
                continue
            if not isinstance(msg, str):
                continue
            text = msg.strip()
            if text:
                out[role_name] = text
        return out

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            role_name = str(item.get("role") or "").strip()
            if role_name not in allowed_roles:
                continue
            msg = item.get("feedback")
            if not isinstance(msg, str):
                continue
            text = msg.strip()
            if text:
                out[role_name] = text
    return out


def resolve_commit_selection(
    raw_commits: Any,
    available_commits: List[str],
    *,
    role: str,
    resolve_symbolic_commit: ResolveSymbolicCommit | None = None,
) -> tuple[List[str], List[str]]:
    if not isinstance(raw_commits, list):
        return [], []
    selected: List[str] = []
    unresolved: List[str] = []
    available_set = set(available_commits)
    for raw in raw_commits:
        if not isinstance(raw, str):
            continue
        token = raw.strip()
        if not token:
            continue
        matches = [sha for sha in available_commits if sha == token or sha.startswith(token)]
        if len(matches) == 1:
            sha = matches[0]
            if sha not in selected:
                selected.append(sha)
            continue
        if len(matches) > 1:
            unresolved.append(token)
            continue

        resolved_sha = (
            resolve_symbolic_commit(role, token) if resolve_symbolic_commit is not None else None
        )
        if resolved_sha and resolved_sha in available_set:
            if resolved_sha not in selected:
                selected.append(resolved_sha)
            continue
        unresolved.append(token)
    return selected, unresolved


def planner_mutated_round(planner_output: Dict[str, Any]) -> bool:
    if bool(planner_output.get("changed")):
        return True
    applied_paths = planner_output.get("applied_paths")
    return isinstance(applied_paths, list) and len(applied_paths) > 0


def review_round_has_accepted_work(*, planner_output: Dict[str, Any], merged_commits_added: bool) -> bool:
    if merged_commits_added:
        return True
    return planner_mutated_round(planner_output)


def public_validate_policy(raw_policy: Any) -> str:
    policy = str(raw_policy or "advisory").strip().lower()
    if policy in {"off", "advisory", "required"}:
        return policy
    return "advisory"


def public_validation_state(
    *,
    policy: str,
    state: str | PublicValidationState,
    ok: bool = False,
    stdout: str = "",
    stderr: str = "",
) -> PublicValidationRecord:
    if isinstance(state, PublicValidationState):
        normalized_state = state
    else:
        try:
            normalized_state = PublicValidationState(str(state))
        except ValueError:
            normalized_state = PublicValidationState.UNAVAILABLE_NOOP
    return PublicValidationRecord(
        policy=public_validate_policy(policy),
        state=normalized_state,
        ok=bool(ok),
        stdout=stdout,
        stderr=stderr,
    )


def evaluate_public_validation(*, policy: str, ok: bool, stdout: str, stderr: str) -> PublicValidationSummary:
    noop = is_noop_public_validation(stdout)
    available = not noop
    passed = bool(ok) and available
    state = PublicValidationState.PASSED if passed else (
        PublicValidationState.FAILED if available else PublicValidationState.UNAVAILABLE_NOOP
    )
    return PublicValidationSummary(
        passed=passed,
        available=available,
        noop=noop,
        last_validation=public_validation_state(
            policy=policy, state=state, ok=bool(ok), stdout=stdout, stderr=stderr
        ),
    )


def is_noop_public_validation(stdout: str) -> bool:
    text = (stdout or "").lower()
    return ("public smoke: no-op" in text) or ("public validation skipped: no command configured" in text)
