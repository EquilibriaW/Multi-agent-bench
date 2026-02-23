#!/usr/bin/env python3
"""
Normalize ABC-Bench task folders into LoopBench TaskPack layout.

Expected source layout (after extracting tasks.tar.gz):
  tasks/task_<repo>__<scenario>/
    <repo_source>/
    task.yaml
    run-tests.sh
    docker-compose.yaml (optional)
    Dockerfile (optional)
    tests/ (optional)

Output layout per task:
  <out_root>/<task_id>/
    task.yaml
    repo.bundle
    public/
      README.task.md
      smoke/smoke.sh
      workloads/
      env/
    hidden/
      validate.sh
      run-tests.sh (if present)
      tests/ (if present)
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

KNOWN_NON_REPO_NAMES = {
    "task.yaml",
    "run-tests.sh",
    "docker-compose.yaml",
    "docker-compose.yml",
    "docker-compose.override.yml",
    "Dockerfile",
    "solution.sh",
    "tests",
    "README.md",
    "readme.md",
    "public",
    "hidden",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abc-root", required=True, help="Directory containing task_* folders")
    parser.add_argument("--out-root", required=True, help="Destination LoopBench tasks root")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of imported tasks")
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Import only specific task folder names (repeatable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output task folders if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = Path(args.abc_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not src_root.exists():
        raise SystemExit(f"source root does not exist: {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    task_dirs = [d for d in sorted(src_root.iterdir()) if d.is_dir() and d.name.startswith("task_")]
    if args.task_id:
        include = set(args.task_id)
        task_dirs = [d for d in task_dirs if d.name in include]

    if args.limit > 0:
        task_dirs = task_dirs[: args.limit]

    if not task_dirs:
        raise SystemExit("no matching task directories found")

    imported = 0
    skipped = 0
    for task_dir in task_dirs:
        out_dir = out_root / task_dir.name
        if out_dir.exists():
            if args.overwrite:
                shutil.rmtree(out_dir)
            else:
                print(f"skip (exists): {out_dir}")
                skipped += 1
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        import_one_task(task_dir=task_dir, out_dir=out_dir)
        imported += 1
        print(f"imported: {task_dir.name}")

    print(f"done: imported={imported}, skipped={skipped}, out_root={out_root}")


def import_one_task(task_dir: Path, out_dir: Path) -> None:
    raw_task_yaml = read_yaml(task_dir / "task.yaml")
    instruction = str(raw_task_yaml.get("instruction") or "").strip()
    language = detect_language(raw_task_yaml)
    difficulty = str(raw_task_yaml.get("difficulty") or "medium")

    repo_src = detect_repo_source(task_dir)
    if repo_src is None:
        raise RuntimeError(f"could not detect repository source folder in {task_dir}")

    create_git_bundle(repo_src, out_dir / "repo.bundle")

    public_dir = out_dir / "public"
    hidden_dir = out_dir / "hidden"
    (public_dir / "smoke").mkdir(parents=True, exist_ok=True)
    (public_dir / "workloads").mkdir(parents=True, exist_ok=True)
    (public_dir / "env").mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    (public_dir / "README.task.md").write_text(
        render_task_readme(task_id=task_dir.name, instruction=instruction),
        encoding="utf-8",
    )

    smoke_staged = stage_public_smoke(task_dir=task_dir, smoke_dir=public_dir / "smoke")
    public_validate_cmd = detect_public_smoke_command(smoke_dir=public_dir / "smoke") if smoke_staged else None

    if (task_dir / "run-tests.sh").exists():
        shutil.copy2(task_dir / "run-tests.sh", hidden_dir / "run-tests.sh")
        make_executable(hidden_dir / "run-tests.sh")

    if (task_dir / "tests").exists() and (task_dir / "tests").is_dir():
        shutil.copytree(task_dir / "tests", hidden_dir / "tests")

    compose_src = find_first_existing(task_dir, ["docker-compose.yaml", "docker-compose.yml", "docker-compose.override.yml"])
    if compose_src:
        shutil.copy2(compose_src, public_dir / "env" / "docker-compose.yaml")

    if (task_dir / "Dockerfile").exists():
        public_dockerfile = public_dir / "env" / "Dockerfile"
        shutil.copy2(task_dir / "Dockerfile", public_dockerfile)
        rewrite_public_env_dockerfile(public_dockerfile, repo_dir_name=repo_src.name)

    (public_dir / "original_task.yaml").write_text(
        yaml.safe_dump(raw_task_yaml, sort_keys=False),
        encoding="utf-8",
    )

    hidden_validate = render_hidden_validate(has_run_tests=(hidden_dir / "run-tests.sh").exists())
    (hidden_dir / "validate.sh").write_text(hidden_validate, encoding="utf-8")
    make_executable(hidden_dir / "validate.sh")

    task_yaml = {
        "task_id": task_dir.name,
        "kind": "abcbench",
        "difficulty": difficulty,
        "language": language,
        "workspace": {
            "base_repo": "repo.bundle",
            "entrypoint": "README.task.md",
        },
        "substrate": {
            "kind": "compose" if compose_src else "none",
            "worktree_isolation": True,
            "up_cmd": "docker compose -f public/env/docker-compose.yaml up -d --build" if compose_src else None,
            "down_cmd": "docker compose -f public/env/docker-compose.yaml down --remove-orphans" if compose_src else None,
            "public_validate_cmd": public_validate_cmd,
            "public_validate_policy": "advisory" if public_validate_cmd else "off",
        },
        "feedback_surfaces": {
            "logs": True,
            "metrics": False,
            "db": False,
            "browser": False,
        },
        "judge": {
            "hidden_validate_cmd": "bash hidden/validate.sh",
            "timeout_sec": 1800,
        },
        "budgets": {
            "wall_clock_sec": 3600,
            "tool_calls": 800,
            "env_cycles": 30,
            "log_queries": 200,
            "metric_queries": 100,
            "http_requests": 300,
        },
    }

    task_yaml["substrate"] = {k: v for k, v in task_yaml["substrate"].items() if v is not None}
    (out_dir / "task.yaml").write_text(yaml.safe_dump(task_yaml, sort_keys=False), encoding="utf-8")


def read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping yaml: {path}")
    return data


def detect_repo_source(task_dir: Path) -> Path | None:
    candidates = []
    for child in task_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in {"tests", "public", "hidden", ".git"}:
            continue
        if child.name in KNOWN_NON_REPO_NAMES:
            continue
        candidates.append(child)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    for cand in candidates:
        if (cand / ".git").exists():
            return cand

    scored = sorted(candidates, key=lambda p: count_files(p), reverse=True)
    return scored[0]


def count_files(root: Path) -> int:
    total = 0
    for _ in root.rglob("*"):
        total += 1
    return total


def create_git_bundle(repo_src: Path, out_bundle: Path) -> None:
    if (repo_src / ".git").exists():
        run(["git", "-C", str(repo_src), "bundle", "create", str(out_bundle), "--all"])
        return

    with tempfile.TemporaryDirectory(prefix="loopbench_abc_repo_") as tmp:
        tmp_repo = Path(tmp) / "repo"
        shutil.copytree(repo_src, tmp_repo)
        run(["git", "-C", str(tmp_repo), "init"])
        run(["git", "-C", str(tmp_repo), "config", "user.name", "loopbench-importer"])
        run(["git", "-C", str(tmp_repo), "config", "user.email", "loopbench-importer@local"])
        run(["git", "-C", str(tmp_repo), "add", "-A"])
        run(["git", "-C", str(tmp_repo), "commit", "-m", "Initial import snapshot"])
        run(["git", "-C", str(tmp_repo), "bundle", "create", str(out_bundle), "--all"])


def detect_language(task_yaml: Dict) -> str:
    tags = task_yaml.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    lowered = [str(t).lower() for t in tags]
    for token in lowered:
        if token in {"python", "java", "javascript", "typescript", "go", "ruby", "php", "rust", "c#", "csharp"}:
            return token
    return "unknown"


def render_task_readme(task_id: str, instruction: str) -> str:
    body = instruction if instruction else "No instruction found in source task.yaml"
    return f"# {task_id}\n\n{body}\n"


def render_hidden_validate(has_run_tests: bool) -> str:
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "if [[ -z \"${LOOPBENCH_REPO_DIR:-}\" ]]; then",
        "  echo \"LOOPBENCH_REPO_DIR is required\" >&2",
        "  exit 1",
        "fi",
        "",
        "cd \"${LOOPBENCH_REPO_DIR}\"",
        "",
        "# Use a writable per-run HOME so hidden scripts can install local tools (uv, etc.).",
        "export HOME=\"${LOOPBENCH_HIDDEN_DIR}/.loopbench_home\"",
        "export UV_PYTHON=\"${UV_PYTHON:-3.11}\"",
        "export UV_CACHE_DIR=\"${HOME}/.cache/uv\"",
        "mkdir -p \"${HOME}\" \"${HOME}/.local/bin\" \"${UV_CACHE_DIR}\"",
        "",
        "# ABC hidden scripts assume Debian-like hosts. Provide a safe shim when absent.",
        "if ! command -v apt-get >/dev/null 2>&1; then",
        "  shim_dir=\"${LOOPBENCH_HIDDEN_DIR}/.loopbench_shims\"",
        "  mkdir -p \"${shim_dir}\"",
        "  cat > \"${shim_dir}/apt-get\" <<'SHIM'",
        "#!/usr/bin/env bash",
        "echo \"loopbench: apt-get unavailable; skipping apt-get $*\" >&2",
        "exit 0",
        "SHIM",
        "  chmod +x \"${shim_dir}/apt-get\"",
        "  export PATH=\"${shim_dir}:${PATH}\"",
        "fi",
    ]

    if has_run_tests:
        script.extend(
            [
                "if [[ -f \"${LOOPBENCH_HIDDEN_DIR}/run-tests.sh\" ]]; then",
                "  # Many ABC task runners start with `cd <repo_name>` where <repo_name>",
                "  # was the original top-level extracted folder. Our normalized checkout",
                "  # uses LOOPBENCH_REPO_DIR as root, so create a compatibility symlink when needed.",
                "  first_cd=$(awk '/^[[:space:]]*cd[[:space:]]+/ {print $2; exit}' \"${LOOPBENCH_HIDDEN_DIR}/run-tests.sh\" || true)",
                "  if [[ -n \"${first_cd}\" ]]; then",
                "    first_cd=${first_cd%\"\\\"\"}",
                "    first_cd=${first_cd#\"\\\"\"}",
                "    first_cd=${first_cd%\\'}",
                "    first_cd=${first_cd#\\'}",
                "    if [[ ! -e \"${first_cd}\" ]]; then",
                "      ln -s . \"${first_cd}\"",
                "    fi",
                "  fi",
                "  runner=\"${LOOPBENCH_HIDDEN_DIR}/run-tests.sh\"",
                "  patched=\"${LOOPBENCH_HIDDEN_DIR}/run-tests.loopbench.sh\"",
                "  # Keep uv project bootstrap isolated; never register transient judge repos",
                "  # as members in the parent LoopBench workspace.",
                "  sed -E 's/uv[[:space:]]+init/uv init --no-workspace/g' \"${runner}\" > \"${patched}\"",
                "  chmod +x \"${patched}\"",
                "  runner=\"${patched}\"",
                "  (",
                "    unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_SHLVL VIRTUAL_ENV PYTHONHOME PYTHONPATH",
                "    bash \"${runner}\"",
                "  )",
                "else",
                "  echo \"hidden run-tests.sh missing\" >&2",
                "  exit 1",
                "fi",
            ]
        )
    else:
        script.extend(
            [
                "echo \"No hidden run-tests.sh available\" >&2",
                "exit 1",
            ]
        )

    return "\n".join(script) + "\n"


def rewrite_public_env_dockerfile(dockerfile_path: Path, repo_dir_name: str) -> None:
    """
    ABC task Dockerfiles often use `COPY ./<repo_dir> ...` from the original extracted layout.
    After normalization, repo root is flattened, so rewrite that source to `.`.
    """
    if not repo_dir_name.strip():
        return

    text = dockerfile_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^(\s*COPY\s+)(?:\./)?{re.escape(repo_dir_name)}(?:/)?(\s+.+)$",
        flags=re.MULTILINE,
    )
    rewritten = pattern.sub(r"\1.\2", text)
    if rewritten != text:
        dockerfile_path.write_text(rewritten, encoding="utf-8")


def stage_public_smoke(*, task_dir: Path, smoke_dir: Path) -> bool:
    copied = False
    for candidate in (task_dir / "public" / "smoke", task_dir / "smoke"):
        if not candidate.is_dir():
            continue
        for child in candidate.iterdir():
            target = smoke_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
                copied = True
                continue
            shutil.copy2(child, target)
            copied = True
    if not copied:
        return False
    for script in smoke_dir.rglob("*.sh"):
        if script.is_file():
            make_executable(script)
    return True


def detect_public_smoke_command(*, smoke_dir: Path) -> str | None:
    preferred = smoke_dir / "smoke.sh"
    if preferred.is_file():
        return "bash public/smoke/smoke.sh"

    scripts = sorted(path for path in smoke_dir.rglob("*.sh") if path.is_file())
    if not scripts:
        return None
    rel = scripts[0].relative_to(smoke_dir).as_posix()
    return f"bash public/smoke/{rel}"


def find_first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return
    joined = " ".join(cmd)
    raise RuntimeError(
        f"command failed: {joined}\nexit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


if __name__ == "__main__":
    main()
