#!/usr/bin/env python3
"""Generate experiment config YAML for full evaluation runs.

Scans .tmp/abc_out/task_* for all task directories and writes a YAML config
with the specified lineups.
"""

import os
import sys
from pathlib import Path

TASKS_DIR = Path(".tmp/abc_out")
OUTPUT_PATH = Path("configs/experiment.full-eval-2model-224x1-20260226.yaml")


def main():
    if not TASKS_DIR.is_dir():
        print(f"ERROR: tasks directory not found: {TASKS_DIR}", file=sys.stderr)
        sys.exit(1)

    task_dirs = sorted(
        d.name for d in TASKS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("task_")
    )

    if not task_dirs:
        print("ERROR: no task directories found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(task_dirs)} tasks")

    tasks_block = "\n".join(f"  - {TASKS_DIR}/{t}" for t in task_dirs)

    yaml_content = f"""\
experiment_id: full-eval-2model-224x1-20260226

tasks:
{tasks_block}

lineups:
  - name: kimi-k2.5
    planner_model: moonshotai/kimi-k2.5
    coder_a_model: moonshotai/kimi-k2.5
    coder_b_model: moonshotai/kimi-k2.5
    role_command: scripts/dspy_role_driver.py
    role_env:
      OPENROUTER_API_KEY_ENV: OPEN_ROUTER_API_KEY
      OPENROUTER_APP_TITLE: loopbench-team
      OPENROUTER_HTTP_REFERER: https://localhost/loopbench
      OPENROUTER_TEMPERATURE: "0.2"
      OPENROUTER_MAX_TOKENS: "16384"
      OPENROUTER_HTTP_TIMEOUT_SEC: "90"
      OPENROUTER_HTTP_RETRIES: "4"


  - name: deepseek-v3.2
    planner_model: deepseek/deepseek-v3.2
    coder_a_model: deepseek/deepseek-v3.2
    coder_b_model: deepseek/deepseek-v3.2
    role_command: scripts/dspy_role_driver.py
    role_env:
      OPENROUTER_API_KEY_ENV: OPEN_ROUTER_API_KEY
      OPENROUTER_APP_TITLE: loopbench-team
      OPENROUTER_HTTP_REFERER: https://localhost/loopbench
      OPENROUTER_TEMPERATURE: "0.2"
      OPENROUTER_MAX_TOKENS: "16384"
      OPENROUTER_HTTP_TIMEOUT_SEC: "90"
      OPENROUTER_HTTP_RETRIES: "4"
      OPENROUTER_REASONING_ENABLED: "false"


rollouts_per_task: 1
max_parallel_rollouts: 30
available_agent_sandboxes: 100
runtime_config: configs/runtime.e2b.openrouter.longrun.yaml
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(yaml_content)
    print(f"Wrote config to {OUTPUT_PATH}")
    print(f"  Tasks: {len(task_dirs)}")
    print(f"  Lineups: 2 (kimi-k2.5, deepseek-v3.2)")
    print(f"  Total jobs: {len(task_dirs) * 2}")


if __name__ == "__main__":
    main()
