"""
loopbench.cli

CLI entrypoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from rich.console import Console
except ModuleNotFoundError:
    class Console:  # type: ignore[override]
        def print(self, message: str) -> None:
            print(message)

from .config import load_runtime_config
from .controller import pack_task, replay_events, run_benchmark, run_judge_only
from .experiments import run_experiment
from .preflight import run_preflight

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run agents on a task (public loop + hidden judge)")
    run_p.add_argument("--task-dir", required=True)
    run_p.add_argument("--run-id", required=True)
    run_p.add_argument("--config", default="configs/runtime.yaml")
    run_p.add_argument("--agents-config", default="configs/agents.yaml")
    run_p.add_argument(
        "--allow-noop",
        action="store_true",
        help="allow noop drivers (only for harness plumbing/smoke tests)",
    )

    judge_p = sub.add_parser("judge", help="run hidden validation on an existing final patch")
    judge_p.add_argument("--task-dir", required=True)
    judge_p.add_argument("--run-dir", required=True)
    judge_p.add_argument("--config", default="configs/runtime.yaml")

    pack_p = sub.add_parser("pack", help="pack a task into a distributable tar.gz")
    pack_p.add_argument("--task-dir", required=True)
    pack_p.add_argument("--out", required=True)

    replay_p = sub.add_parser("replay", help="replay event logs and print aggregate counts")
    replay_p.add_argument("--run-dir", required=True)

    exp_p = sub.add_parser("experiment", help="run multi-rollout experiment matrix")
    exp_p.add_argument("--spec", required=True, help="experiment yaml spec path")
    exp_p.add_argument("--tasks-root", default=None, help="optional tasks root for relative task entries")
    exp_p.add_argument("--allow-noop", action="store_true")

    pre_p = sub.add_parser("preflight", help="run infra readiness checks")
    pre_p.add_argument("--config", default="configs/runtime.yaml")

    insp_p = sub.add_parser("inspect", help="render run artifacts as a cohesive narrative")
    insp_p.add_argument("run_dir", help="path to run directory (must contain manifest.json)")
    insp_p.add_argument("--format", choices=["rich", "markdown"], default="rich", dest="output_format")
    insp_p.add_argument("--diff-lines", type=int, default=60, help="max diff lines shown")

    hodo_p = sub.add_parser("hodoscope", help="export trajectories and run hodoscope analysis")
    hodo_p.add_argument("run_dirs", nargs="+", help="run directories to analyze")
    hodo_p.add_argument("--viz", action="store_true", help="generate interactive HTML visualization")
    hodo_p.add_argument("--out", default=None, help="output directory for analysis artifacts")
    hodo_p.add_argument("--summarize-model", default=None, help="LLM model for action summarization")
    hodo_p.add_argument("--embedding-model", default=None, help="model for embedding summaries")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        manifest = run_benchmark(
            task_dir=args.task_dir,
            run_id=args.run_id,
            runtime_config_path=args.config,
            agents_config_path=args.agents_config,
            project_root=Path.cwd(),
            allow_noop=args.allow_noop,
        )
        console.print("[green]run complete[/green]")
        console.print(f"run_id: {manifest.run_id}")
        console.print(f"task_id: {manifest.task_id}")
        console.print(f"public_pass: {manifest.public_pass}")
        console.print(f"hidden_pass: {manifest.hidden_pass}")
        console.print(f"manifest: runs/{manifest.run_id}/manifest.json")
        return

    if args.cmd == "judge":
        result = run_judge_only(
            task_dir=args.task_dir,
            run_dir=args.run_dir,
            runtime_config_path=args.config,
        )
        console.print("[green]judge complete[/green]")
        console.print(f"ok: {result.ok}")
        console.print(f"exit_code: {result.exit_code}")
        if result.stderr.strip():
            console.print("stderr:")
            console.print(result.stderr[-2000:])
        return

    if args.cmd == "pack":
        out = pack_task(task_dir=args.task_dir, out_path=args.out)
        console.print(f"[green]packed[/green] -> {out}")
        return

    if args.cmd == "replay":
        summary = replay_events(run_dir=args.run_dir)
        console.print(json.dumps(summary, indent=2))
        return

    if args.cmd == "experiment":
        result = run_experiment(
            spec_path=args.spec,
            project_root=Path.cwd(),
            tasks_root=args.tasks_root,
            allow_noop=args.allow_noop,
        )
        console.print("[green]experiment complete[/green]")
        console.print(json.dumps(result, indent=2), highlight=False, markup=False)
        return

    if args.cmd == "preflight":
        runtime_cfg = load_runtime_config(args.config)
        result = run_preflight(runtime_cfg)
        console.print(json.dumps(result, indent=2))
        return

    if args.cmd == "inspect":
        from .inspect_run import run_inspect

        run_inspect(run_dir=args.run_dir, output_format=args.output_format, diff_lines=args.diff_lines)
        return

    if args.cmd == "hodoscope":
        from .hodoscope_export import export_and_analyze

        result = export_and_analyze(
            run_dirs=args.run_dirs,
            out_dir=args.out,
            viz=args.viz,
            summarize_model=args.summarize_model,
            embedding_model=args.embedding_model,
        )
        console.print("[green]hodoscope analysis complete[/green]")
        console.print(json.dumps(result, indent=2))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()
