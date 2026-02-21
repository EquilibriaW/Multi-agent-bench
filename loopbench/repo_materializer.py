"""
loopbench.repo_materializer

Shared repository materialization and git identity configuration.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .shell import ensure_success, run_command


def materialize_repo(
    *,
    source: str | Path,
    target: str | Path,
    git_user_name: str,
    git_user_email: str,
    initial_commit_message: str = "Initial task template",
) -> None:
    source_path = Path(source)
    target_path = Path(target)

    if source_path.is_file():
        ensure_success(
            run_command(["git", "clone", str(source_path), str(target_path)]),
            f"git clone {source_path}",
        )
        return

    if source_path.is_dir() and (source_path / ".git").exists():
        ensure_success(
            run_command(["git", "clone", "--no-hardlinks", str(source_path), str(target_path)]),
            f"git clone {source_path}",
        )
        return

    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
        ensure_success(run_command(["git", "-C", str(target_path), "init"]), "git init")
        configure_git_identity(
            repo=target_path,
            user_name=git_user_name,
            user_email=git_user_email,
        )
        ensure_success(run_command(["git", "-C", str(target_path), "add", "-A"]), "git add")
        ensure_success(
            run_command(["git", "-C", str(target_path), "commit", "-m", initial_commit_message]),
            "git commit initial",
        )
        return

    raise FileNotFoundError(f"Unsupported source repo: {source_path}")


def configure_git_identity(*, repo: str | Path, user_name: str, user_email: str) -> None:
    repo_path = Path(repo)
    ensure_success(
        run_command(["git", "-C", str(repo_path), "config", "user.name", user_name]),
        "git config user.name",
    )
    ensure_success(
        run_command(["git", "-C", str(repo_path), "config", "user.email", user_email]),
        "git config user.email",
    )
