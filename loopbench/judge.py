"""
loopbench.judge

Hidden validation runner using a fresh repo checkout.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import JudgeRuntimeConfig
from .docker_runtime import docker_info, judge_docker_env
from .path_utils import resolve_within_root
from .repo_materializer import configure_git_identity, materialize_repo
from .schema import TaskPack, ToolResult
from .shell import run_command, run_shell
from .time_utils import now_ms


class LocalJudge:
    """
    Host-process judge runner.

    When require_docker=True, hidden validation fails fast unless `docker info`
    succeeds on the judge host. This keeps agent sandboxes unprivileged while
    still allowing hidden test realism for Docker-based ABC tasks.
    """

    def __init__(
        self,
        task: TaskPack,
        run_dir: str | Path,
        require_docker: bool = False,
        docker_env: dict[str, str] | None = None,
    ):
        self.task = task
        self.run_dir = Path(run_dir).resolve()
        self.require_docker = require_docker
        self.docker_env = docker_env or {}

    def run_hidden_validation(self, final_patch_path: str) -> ToolResult:
        ts_ms = now_ms()
        docker_diag: str | None = None
        if self.require_docker:
            docker_check = docker_info(env=self.docker_env, timeout_sec=30)
            if not docker_check.ok:
                docker_diag = (
                    "docker runtime unavailable for judge before hidden validation. "
                    "Start docker daemon on judge host.\n\n"
                    f"{docker_check.stderr}"
                )

        judge_root = self.run_dir / "hidden_validate" / "judge_workspace"
        repo_dir = judge_root / "repo"

        if judge_root.exists():
            shutil.rmtree(judge_root)
        judge_root.mkdir(parents=True, exist_ok=True)

        source_root = Path(self.task.root_dir)
        source_repo = source_root / self.task.workspace.base_repo
        materialize_repo(
            source=source_repo,
            target=repo_dir,
            git_user_name="loopbench-judge",
            git_user_email="loopbench-judge@local",
        )
        configure_git_identity(
            repo=repo_dir,
            user_name="loopbench-judge",
            user_email="loopbench-judge@local",
        )

        hidden_src = source_root / self.task.hidden_dir
        public_src = source_root / self.task.public_dir
        if hidden_src.exists():
            shutil.copytree(hidden_src, judge_root / "hidden")
        if public_src.exists():
            shutil.copytree(public_src, judge_root / "public")

        patch_path = resolve_within_root(root=self.run_dir, raw_path=final_patch_path)
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        if patch_text.strip():
            patch_result = run_command(
                ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", str(patch_path)]
            )
            if not patch_result.ok:
                return ToolResult(
                    ts_ms=ts_ms,
                    ok=False,
                    tool="judge.hidden_validate",
                    stdout=patch_result.stdout,
                    stderr=patch_result.stderr,
                    exit_code=patch_result.exit_code,
                    data={"stage": "apply_patch"},
                )

        env = {
            "LOOPBENCH_REPO_DIR": str(repo_dir),
            "LOOPBENCH_TASK_DIR": str(source_root),
            "LOOPBENCH_HIDDEN_DIR": str(judge_root / "hidden"),
            "LOOPBENCH_PUBLIC_DIR": str(judge_root / "public"),
        }
        run_env = {**env, **self.docker_env}
        result = run_shell(
            self.task.judge.hidden_validate_cmd,
            cwd=judge_root,
            timeout_sec=self.task.judge.timeout_sec,
            env=run_env,
        )
        stderr = result.stderr
        infra_error = bool(docker_diag and not result.ok)
        if docker_diag and not result.ok:
            stderr = f"{docker_diag}\n\n{stderr}".strip()

        out_dir = self.run_dir / "hidden_validate"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "result.log"
        out_path.write_text(
            f"CMD: {self.task.judge.hidden_validate_cmd}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{stderr}\n",
            encoding="utf-8",
        )

        return ToolResult(
            ts_ms=ts_ms,
            ok=result.ok,
            tool="judge.hidden_validate",
            stdout=result.stdout,
            stderr=stderr,
            exit_code=result.exit_code,
            data={
                "result_log": str(out_path),
                "infra_error": infra_error,
            },
        )


def build_judge(*, task: TaskPack, run_dir: str | Path, judge_cfg: JudgeRuntimeConfig) -> LocalJudge:
    docker_env = judge_docker_env(judge_cfg)
    if judge_cfg.kind == "docker_container":
        return LocalJudge(task=task, run_dir=run_dir, require_docker=True, docker_env=docker_env)
    return LocalJudge(task=task, run_dir=run_dir, require_docker=False, docker_env=docker_env)
