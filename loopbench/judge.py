"""
loopbench.judge

Hidden validation runner using a fresh repo checkout.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
import fcntl
import re
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

        # Place the judge workspace outside the per-run directory tree so that
        # agents using a local-process sandbox cannot discover hidden tests by
        # walking up from their worktree into runs/<run_id>/hidden_validate/.
        judge_root = self.run_dir.parent / ".judge_workspaces" / self.run_dir.name
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
        lock_ctx = (
            _judge_docker_runtime_lock(runs_root=self.run_dir.parent, docker_env=self.docker_env)
            if self.require_docker
            else nullcontext(None)
        )
        with lock_ctx as lock_path:
            result = run_shell(
                self.task.judge.hidden_validate_cmd,
                cwd=judge_root,
                timeout_sec=self.task.judge.timeout_sec,
                env=run_env,
            )
        stderr = result.stderr
        infra_error = bool(
            (docker_diag and not result.ok)
            or (not result.ok and _is_likely_docker_infra_failure(stderr=result.stderr, stdout=result.stdout))
        )
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
                "judge_workspace": str(judge_root),
                "infra_error": infra_error,
                "docker_lock_path": str(lock_path) if lock_path else None,
            },
        )


def build_judge(*, task: TaskPack, run_dir: str | Path, judge_cfg: JudgeRuntimeConfig) -> LocalJudge:
    docker_env = judge_docker_env(judge_cfg)
    if judge_cfg.kind == "docker_container":
        return LocalJudge(task=task, run_dir=run_dir, require_docker=True, docker_env=docker_env)
    return LocalJudge(task=task, run_dir=run_dir, require_docker=False, docker_env=docker_env)


_INFRA_ERROR_SUBSTRINGS = (
    "cannot connect to the docker daemon",
    "error during connect",
    "permission denied while trying to connect to the docker daemon socket",
    "failed to update builder last activity time",
    "apt-get: command not found",
    "two workspace members are both named",
)


def _is_likely_docker_infra_failure(*, stderr: str, stdout: str) -> bool:
    text = f"{stderr}\n{stdout}".lower()
    if any(marker in text for marker in _INFRA_ERROR_SUBSTRINGS):
        return True
    if "/.docker/buildx/activity" in text and "operation not permitted" in text:
        return True
    if "no solution found when resolving dependencies" in text and "pytest==8.4.1 depends on python>=3.9" in text:
        return True
    return False


@contextmanager
def _judge_docker_runtime_lock(*, runs_root: Path, docker_env: dict[str, str]) -> Iterator[Path]:
    lock_root = runs_root / ".locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    host_token = _docker_host_lock_token(docker_env.get("DOCKER_HOST"))
    lock_path = lock_root / f"judge_docker_{host_token}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _docker_host_lock_token(raw_host: str | None) -> str:
    text = (raw_host or "local").strip().lower()
    if not text:
        text = "local"
    token = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return token or "local"
