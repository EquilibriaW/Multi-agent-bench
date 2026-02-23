# Roles & Multi-Agent Protocol

LoopBench standardizes a collaboration pattern that resembles real coding teams:

- 1 × **planner_reviewer**
- 2 × **coder** agents: `coder_a`, `coder_b`

The goal is NOT to find the optimal collaboration strategy.
The goal is to **make collaboration measurable and reproducible**.

## 1) Worktree layout

Each role owns a worktree:

```
worktrees/
  planner_reviewer/
  coder_a/
  coder_b/
```

Rules:
- Only coders commit to their own branches.
- Planner/reviewer cherry-picks commits into a merge candidate (its own worktree).
- The final patch is produced from planner/reviewer worktree.

## 2) Run phases

### Phase 0 — Bootstrap (planner_reviewer)
Inputs:
- `public/README.task.md`
- repo code
Outputs:
- `runs/<run_id>/plans/plan.md`
- `runs/<run_id>/plans/subtasks.yaml` (machine-readable assignment)

### Phase 1 — Parallel implementation (coder_a + coder_b)
Each coder:
- reads plan + assigned files
- implements a bounded change set
- runs relevant public validation tools
- produces:
  - commits on their branch
  - `runs/<run_id>/role_summaries/coder_*.md` (what changed + why)
  - `runs/<run_id>/repo_state/<role>/diff_from_base.patch` (exact delta)

Prompt policy:
- `coder_a` and `coder_b` use the same generic coder-role prompt.
- task/work split instructions are authored by `planner_reviewer` (plan + DB assignments), not by per-coder prompt variants.

### Phase 2 — Review & iteration (planner_reviewer ↔ coders)
Planner/reviewer:
- reviews diffs
- runs dynamic checks (`run_commands`) during review
- explicitly nominates which coder commits are eligible to merge
- requests changes
- optionally writes public regression tests
- repeats until:
  - public validation policy is satisfied (`required`) or advisory checks are exhausted (`advisory`/`off`)
  - merge candidate is coherent

Harness enforcement:
- merge only includes commits listed by planner/reviewer nomination (`merge_commits`), not all unseen coder commits.
- planner/reviewer can force rework even when public validation passes (`request_rework=true` + `coder_feedback`).
- planner/reviewer output in review must include structured `merge_commits` (role -> commit list) and `request_rework`.
- planner/reviewer review context includes coder commit candidates (`coder_commits`, `candidate_merge_commits`) and recent coder outputs to support explicit merge decisions.
- planner/reviewer also receives a diff-inspection command (`review_diff_tool`) to list commits and view full patch content by coder + commit sha.
- after nominated merges are materialized, planner/reviewer runs a `review_verify` phase on the integrated candidate and can request rework.
- a review round is only acceptance-eligible when it is actionable:
  - nominates at least one coder commit for merge, or
  - requests coder rework, or
  - applies planner-owned direct edits with rationale.

### Phase 3 — Finalization (planner_reviewer)
Outputs:
- `final.patch` (unified diff)
- `manifest.json` with public results

Judge then executes hidden validation.

## 3) Tool discipline

- Use ToolRouter methods; do not rely on direct shell networking.
- Use `env.*` tools for deployment/feedback.
- Use `repo.*` tools for changes and git operations.

## 4) Progress tracking (Codex-friendly)

The benchmark harness should keep a small set of **append-only** files that agents update:

- `runs/<run_id>/status.md` — high-level progress, blockers
- `runs/<run_id>/decisions.md` — architectural decisions (short)
- `runs/<run_id>/open_questions.md` — TODOs and uncertainties

The controller can enforce structure by linting these files.

## 5) Team Coordination Protocol (DB-backed)

Coordination is persisted in `runs/<run_id>/coordination/coordination.db` (SQLite), not only in prompt/context.

- `tasks` table: planner assignments, role claims, completion/failure state.
- `messages` table: top-down directives and coder->planner progress updates.
- `claims` table: append-only task claim/complete/fail audit trail.

Required behavior:
- planner seeds implementation/rework tasks in DB.
- coders claim only their assigned pending tasks (transactional claim).
- coders publish completion/failure messages back to planner.
- harness records coordination counters into `manifest.json` metrics.

## 6) Driver policy

- `configs/agents.yaml` is production-oriented and uses `driver: shell`.
- `configs/agents.smoke.yaml` is for plumbing-only smoke tests and uses `driver: noop`.
- `loopbench run` rejects any `noop` role by default.
- Use `--allow-noop` only for harness verification, never for reported benchmark results.
- `scripts/openrouter_role_driver.py` can be used as a shell driver command for OpenRouter-hosted models.
  Set `OPENROUTER_API_KEY_ENV` (e.g., `OPEN_ROUTER_API_KEY`) in role env instead of committing keys.
- Shell role drivers are executed via the role sandbox (not directly on host shell).
