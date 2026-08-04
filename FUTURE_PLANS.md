# FUTURE PLANS & WORKFLOW RULES

Status snapshot: all 9 revisions in `revisions.md` are implemented and
committed (see `REVISION_PLAN.md` for per-revision status + gates).

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
Run IPPO (Phase A floor), MAPPO, QMIX, and `train_comm` at 300k–1M steps on
stage-0 (seeds 0–2).  Resolves the deferred G4 claim "comm ≥ Phase A floor".
GPU measured at only ~16 sps (env stepping dominates), so budget ~5h per
300k-step comm run; parallelize seeds across GPUs if available.

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
`CommAgent` supports `centralized=True`; add a MAPPO-style comm trainer
(centralized critic + messages) and optionally a QMIX-style comm variant.

### 6. Housekeeping
- Retrain on CUDA (all smokes so far were CPU).
- `checkpoints/mappo_s0/policy.pt` and old `runs/` event files are
  invalidated by the REV-7 dim change: regenerate or remove.
- Keep the pytest suite green (`uv run pytest -q`).
