"""
loopbench.workspace

Provision run-scoped workspaces and git worktrees.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .repo_materializer import configure_git_identity, materialize_repo
from .schema import TaskPack
from .shell import ensure_success, run_command

DEFAULT_ROLES = ["planner_reviewer", "coder_a", "coder_b"]


@dataclass(frozen=True)
class RunPaths:
    workspace_root: Path
    runs_root: Path
    run_dir: Path
    role_paths: Dict[str, Path]
    base_commit: str


class WorkspaceManager:
    def __init__(self, project_root: str | Path, workspace_root: str, runs_root: str):
        self.project_root = Path(project_root).resolve()
        self.workspace_root_base = self.project_root / workspace_root
        self.runs_root_base = self.project_root / runs_root

    def provision(self, run_id: str, task: TaskPack, roles: Iterable[str] = DEFAULT_ROLES) -> RunPaths:
        role_list = list(roles)
        if not role_list:
            raise ValueError("at least one role is required")

        workspace_root = self.workspace_root_base / run_id
        run_dir = self.runs_root_base / run_id
        runs_root = self.runs_root_base

        if workspace_root.exists() or run_dir.exists():
            raise FileExistsError(
                f"run_id '{run_id}' already exists (workspace={workspace_root}, run_dir={run_dir})"
            )

        base_repo_path = workspace_root / "base"
        worktrees_dir = workspace_root / "worktrees"

        workspace_root.mkdir(parents=True, exist_ok=False)
        run_dir.mkdir(parents=True, exist_ok=False)
        worktrees_dir.mkdir(parents=True, exist_ok=False)

        materialize_repo(
            source=Path(task.root_dir) / task.workspace.base_repo,
            target=base_repo_path,
            git_user_name="loopbench",
            git_user_email="loopbench@local",
        )
        configure_git_identity(
            repo=base_repo_path,
            user_name="loopbench",
            user_email="loopbench@local",
        )
        base_commit = self._git(base_repo_path, ["rev-parse", "HEAD"]).strip()

        role_paths: Dict[str, Path] = {}
        for role in role_list:
            branch = f"lb/{run_id}/{role}"
            role_path = worktrees_dir / role
            ensure_success(
                run_command(
                    [
                        "git",
                        "-C",
                        str(base_repo_path),
                        "worktree",
                        "add",
                        "-b",
                        branch,
                        str(role_path),
                        base_commit,
                    ]
                ),
                f"git worktree add {role}",
            )
            self._stage_public_assets(task, role_path)
            role_paths[role] = role_path

        self._init_run_docs(run_dir)

        run_paths = RunPaths(
            workspace_root=workspace_root,
            runs_root=runs_root,
            run_dir=run_dir,
            role_paths=role_paths,
            base_commit=base_commit,
        )
        return run_paths

    def _git(self, repo: Path, args: List[str]) -> str:
        result = run_command(["git", "-C", str(repo), *args])
        ensure_success(result, f"git {' '.join(args)}")
        return result.stdout

    def _init_run_docs(self, run_dir: Path) -> None:
        (run_dir / "plans").mkdir(parents=True, exist_ok=True)
        (run_dir / "role_summaries").mkdir(parents=True, exist_ok=True)
        (run_dir / "role_stdio").mkdir(parents=True, exist_ok=True)
        (run_dir / "role_runtime").mkdir(parents=True, exist_ok=True)
        (run_dir / "public_validate").mkdir(parents=True, exist_ok=True)
        (run_dir / "hidden_validate").mkdir(parents=True, exist_ok=True)
        (run_dir / "repo_state").mkdir(parents=True, exist_ok=True)
        (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        (run_dir / "status.md").write_text("# Run Status\n\n- initialized\n", encoding="utf-8")
        (run_dir / "decisions.md").write_text("# Decisions\n\n", encoding="utf-8")
        (run_dir / "open_questions.md").write_text("# Open Questions\n\n", encoding="utf-8")

    def _stage_public_assets(self, task: TaskPack, role_path: Path) -> None:
        source_public = Path(task.root_dir) / task.public_dir
        source_task_yaml = Path(task.task_yaml)

        loopbench_dir = role_path / ".loopbench"
        target_public = loopbench_dir / "public"
        loopbench_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_public, target_public, dirs_exist_ok=True)
        shutil.copy2(source_task_yaml, loopbench_dir / "task.yaml")

        public_link = role_path / "public"
        if not public_link.exists():
            # Relative symlink keeps the worktree self-contained and avoids
            # leaking absolute host paths into container/E2B mounts.
            public_link.symlink_to(Path(".loopbench", "public"), target_is_directory=True)
