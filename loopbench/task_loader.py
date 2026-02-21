"""
loopbench.task_loader

Load and normalize task directories into TaskPack objects.
"""
from __future__ import annotations

from pathlib import Path

from .io_utils import read_yaml_mapping
from .schema import (
    Budget,
    FeedbackSurfaces,
    JudgeSpec,
    SubstrateSpec,
    TaskPack,
    WorkspaceSpec,
)


def load_task_pack(task_dir: str | Path) -> TaskPack:
    root = Path(task_dir).resolve()
    task_yaml = root / "task.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"Missing task.yaml at {task_yaml}")

    raw = read_yaml_mapping(task_yaml, label="task yaml")
    task_id = raw.get("task_id") or root.name
    kind = raw.get("kind") or "repo_patch"

    workspace = WorkspaceSpec.model_validate(raw.get("workspace") or {})
    substrate = SubstrateSpec.model_validate(raw.get("substrate") or {})
    feedback = FeedbackSurfaces.model_validate(raw.get("feedback_surfaces") or {})
    judge = JudgeSpec.model_validate(raw.get("judge") or {})
    budget = Budget.model_validate(raw.get("budgets") or raw.get("budget") or {})

    task = TaskPack(
        task_id=task_id,
        kind=kind,
        difficulty=raw.get("difficulty"),
        language=raw.get("language"),
        workspace=workspace,
        substrate=substrate,
        feedback_surfaces=feedback,
        judge=judge,
        budget=budget,
        root_dir=str(root),
        task_yaml=str(task_yaml),
        public_dir="public",
        hidden_dir="hidden",
    )

    _validate_task_paths(task)
    return task


def _validate_task_paths(task: TaskPack) -> None:
    root = Path(task.root_dir)
    base_path = root / task.workspace.base_repo
    if not base_path.exists():
        raise FileNotFoundError(f"Task base repo path not found: {base_path}")

    public_path = root / task.public_dir
    hidden_path = root / task.hidden_dir

    if not public_path.exists():
        raise FileNotFoundError(f"Task public directory not found: {public_path}")
    if not hidden_path.exists():
        raise FileNotFoundError(f"Task hidden directory not found: {hidden_path}")
