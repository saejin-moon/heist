# FUTURE PLANS & WORKFLOW RULES

Status snapshot: all 9 revisions in `revisions.md` are implemented and
committed (see `REVISION_PLAN.md` for per-revision status + gates).

Post-implementation fix: `train_ippo.py` had a latent shape bug in the
truncation bootstrap path (line ~302) where `terminal_observation` was
packed with a leading env dim of 1 but `role_id` was not stripped;
triggered any time a rollout contained a truncation.  Fixed by indexing
`[0]` on the packed arrays.  Stale pre-REV-7 checkpoints moved to
`deleted/checkpoints/` (regenerable, invalidated by dim changes).

---

## ⚠️ WORKFLOW RULE: Test at small scale BEFORE training

**Before actually training any model on a real budget:**

1. If there is an error or a feature being built, that error/feature MUST be
   tested at a much smaller scale FIRST (a few hundred to a few thousand
   steps, a minimal map, a synthetic trace, a unit test).
2. Never validate a fix or a feature by re-training the full model and
   reading the model's results.  Training is the slowest, noisiest feedback
   loop in the repo; a bug can silently corrupt a multi-hour run.
3. Keep the fast loops in front of the slow loop:
   - lint/format (`uv run ruff check`, `uv run ruff format --check`, seconds)
   - unit/shape smoke tests (`uv run pytest`, ~20s)
   - 2048-step trainer smoke (seconds to a minute)
   - PettingZoo `parallel_api_test` via `src.dummy:make_env` + `dummy.py`
     (contract regressions; also run in CI)
   - scripted-curriculum revalidation (mechanics regressions, ~1 min)

Only after all of those pass, launch a real training run.

> History: the `train_comm.py` GAE indexing bug (`next_values[ai]` vs
> `next_values[:, ai]`) passed the 2048-step smoke (num_envs=4) and only
> crashed at num_envs=8 mid-rollout.  Lesson: vary `num_envs`/`num_steps`
> in smokes so shape-sensitive paths (truncation bootstrapping) are
> exercised with more than one env per batch.

Enforcement: `.github/workflows/ci.yml` runs `ruff check`, `ruff format
--check`, `pytest`, and the PettingZoo parallel-API smoke on every push to
main and on PRs.  `uv sync --locked` in CI guarantees the lockfile matches
`pyproject.toml`, so a stale `uv.lock` fails the build instead of silently
installing a different tree.

---

## Next steps (in priority order)

### 1. The real research experiment (GPU)
**NOT STARTED — SETUP ONLY.**  The full 300k-step campaign
(`experiment_stage0_300k.sh`) must run on a **different machine** and is
**not scheduled here**.  Do NOT launch it on this box yet.

Status while setting up:
- All trainers smoke-tested at 2048 steps and at the real batch shape
  (8 envs × 256 steps); `train_comm.py` GAE truncation bug fixed and
  re-validated (both centralized and default paths).
- `experiment_stage0_300k.sh` is the resumable driver (skips runs whose
  `checkpoints/<algo>_s<seed>/` already exist).  README documents the
  exact setup + run commands.
- An early partial campaign on this machine was **aborted**; its
  checkpoints/event files were quarantined to `deleted/` (see
  `log/experiment_stage0_300k.log` for the record).  Nothing from that
  run should be treated as a result.

To launch the real campaign: follow README → "Full research experiment
(300k, stage-0)".  When it finishes, resolve G4 "comm >= Phase A floor"
and refresh the results tables in `PLAN.md` / `results/README.md`.

### 2. CAI on the new contract
Run `run_eval.py` / `summarize()` on trained models to measure Credit
Attribution Index and counterfactual importance WITHOUT the `global_state`
crutch.  This tests the core PLAN.md claim (Causal Credit Dilution, shared-
critic collapse) under the new observation contract.

### 3. Curriculum scaling
The REV-7 obs change invalidated all old checkpoints.  Re-train at least
IPPO across the 5-stage curriculum (11x11 -> 50x50).  Scripted controller
proves stages solvable; learned baselines under the new contract are
unverified past stage-0 smokes.

### 4. Emergent-language deep-dive (Phase C)
Beyond correlation: cluster gated TarMAC messages by phase, check whether
gates discretize (-> 0/1), and whether attention edges match the causal
chain (e.g., hacker broadcasting "terminal hacked" to extractor).

### 5. Comm variants
- ✅ `train_comm.py --centralized`: state-aware rollout/GAE/minibatch updates
  with a centralized critic on `env.state()`; smoke-tested at 2048 steps and
  at the real batch shape (8 envs × 256 steps) on CPU, both centralized and
  default paths (the rollout-end bootstrap is guarded against empty
  truncation sets).
- 🔜 Add a QMIX-style comm variant (value decomposition + messages) and, if
  useful, a MAPPO-style comm trainer if `--centralized` is not enough.

### 6. Housekeeping
- ✅ Stale pre-REV-7 checkpoints/event files quarantined to `deleted/`
  (they are invalidated by the REV-7 dim change).
- ✅ `train_ippo.py` truncation-bootstrap shape bug fixed (commit `36d99c6`).
- ✅ `train_comm.py` GAE truncation bootstrap fixed + `--centralized` comm
  variant smoke-tested (both paths) at the real batch shape.
- ✅ Early partial campaign aborted by request; its checkpoints quarantined
  to `deleted/` (campaign itself moves to a different machine later).
- Keep the pytest suite green (`uv run pytest -q`).
