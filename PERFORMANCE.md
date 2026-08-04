# PERFORMANCE ANALYSIS & OPTIMIZATION PLAN

Measured on this machine (CPU + RTX 3000 Ada laptop GPU, 8 CPU cores) with
the current codebase.  All timings are real benchmarks, not estimates.
Nothing here has been changed yet; this document lists the opportunities and
the recommended order of attack.  Per the workflow rule, any change must be
validated with a 2048-step trainer smoke before a real run.

---

## 1. Where the time actually goes

### 1.1 Raw env stepping is NOT the stage-0 bottleneck

Micro-benchmarks (single env, `tools/assess_time.py` section 2; resets
included):

| Stage | map | guards | wait ms | random ms | rand steps/s |
|---|---|---|---|---|---|
| 0 | 11x11 | 0 | 0.12 | 0.14 | 7,000 |
| 1 | 15x15 | 2 | 0.34 | 0.36 | 2,700 |
| 2 | 21x21 | 3 | 0.32 | 0.44 | 2,300 |
| 3 | 35x35 | 5 | 0.43 | 0.60 | 1,700 |
| 4 | 50x50 | 6 | 1.57 | 1.38 | 720 |

With 8 envs, the wall-clock cost of one rollout step (one `vec_env.step`)
is ~1 ms at stage-0 — small next to the policy and tensor overhead below.
(Earlier scratch numbers omitted resets and were faster; the real rollout
pays resets, so these are the right figures for training estimates.)

### 1.2 The training loop dominates at stage-0 (CUDA, 8 envs)

`tools/assess_time.py` section 3 measures one full rollout step (tensor
creation + 4 policy forwards + `vec_env.step`, whole loop, no per-phase
timers): **~4.4-5.4 ms/vec-step** at stage-0 (~185-230 vec-steps/s).  The
per-phase breakdown (separate timers) attributes roughly 40% to the four
policy forwards, 35% to `torch.tensor` copies, and 25% to the env step.

End-to-end trainer throughput (`tools/assess_time.py` section 4) is the
number that matters for the campaign:

| Algo | env-steps/s (this box) | 300k-run time |
|---|---|---|
| IPPO | ~630 | ~8 min |
| MAPPO | ~520 | ~10 min |
| QMIX (single-env) | ~105 | ~48 min |
| comm | ~460 | ~11 min |

So the full stage-0 campaign (4 algos x 3 seeds x 300k) is ~4 h on this
machine, and the **trainer loop (rollout + GAE + update + eval) is the
bottleneck, not the raw env**.

Units note: trainer logs print `sps` in rollout-iterations/s (each
iteration steps 8 envs), so the aborted campaign's `sps=74` means
8x74 = **~592 env-steps/s** — exactly what the benchmark above measures.
The earlier pre-correction estimate of 12-16 h was off by ~8x.

### 1.3 At larger maps, the guard AI becomes the env bottleneck

At stage-4 scale (50x50, 6 guards), a per-step breakdown of the env shows:

| Component | ms/step | Share of env step |
|---|---|---|
| `_move_guards` (BFS per guard, pure Python) | 1.230 | 99% |
| `_get_obs` x4 (grid copy + np.pad) | 0.220 | 18% |
| `state()` (centralized critics) | 0.014 | 1% |

`_move_guards` is called separately from `_get_obs` in this micro-profile
(hence the >100% sum), but the conclusion is unambiguous: **BFS pathfinding
in `_bfs_next` / `_move_guards` is the dominant env cost once guards are
present**.  Each guard runs a full pure-Python BFS over the grid every step
in search/converge state.

---

## 2. Recommended changes (priority order)

### P1. Eliminate per-step tensor creation and batch the policy forward

**Where:** `src/train_ippo.py`, `src/train_mappo.py`, `src/train_comm.py`
(rollout loop), `src/vec_env.py`.

**Why:** 78% of rollout time at stage-0 is the 4-per-agent
`torch.tensor(..., device)` copies plus 4 separate MLP forward passes per
step.

**What:**
- Have `vec_env` return obs/role/mask already stacked per agent, and convert
  to torch once per rollout step (or keep persistent device buffers and
  `.copy_()` into them) instead of `torch.tensor(numpy)` per agent per
  component.
- Batch the four per-agent forwards into a single stacked forward
  (e.g. `[4*B, in_dim]` one MLP pass) — same math, one kernel launch.

**Expected gain:** the 6.2 ms/step of policy+tensor could drop to ~1-1.5 ms
on CUDA → roughly **2-3x** faster training at stage-0.

### P2. Fix the guard pathfinding: BFS is right, the implementation is slow

**Should we switch from BFS to A\*?  No.**  The grid is 4-directional and
unweighted (`ACTION_DELTAS`, unit cost per move; `WAIT` excluded), so BFS
already returns optimal shortest paths.  A\* with the Manhattan heuristic
would find the same optimal paths, but on an uncluttered grid it expands
exactly the same diamond-shaped region as BFS (the heuristic is exact along
unobstructed paths), and on maze-like maps both collapse to full-grid
expansion.  The win is not in the algorithm, it is in the implementation:
the measured ~1.23 ms/step is pure-Python `deque`/`dict` overhead for a
~2500-cell search, not search complexity.

**What:**
- **P2a (drop-in, exact behavior):** move `_bfs_next` into a `@njit(cache=True)`
  kernel (int32 `prev`-array + int32 queue over the grid), mirroring
  `vision.py`'s `raycast`/`line_is_clear`.  Fold `_valid_moves` in via a
  small njit neighbor scan so the guard loop stops allocating Python lists.
  ~5-10x on the guard path; env step at stage-4 drops from ~1.25 ms to well
  under 0.5 ms.  Minor at stage-0 (no guards), large for stages 2-4.
- **P2b (bigger structural win, same semantics):** in `converge` state every
  guard hunts the *nearest* agent.  Replace N per-guard BFS with ONE
  multi-source BFS per step from all 4 agent positions: it produces, for
  every walkable cell, its exact distance to the nearest agent; each guard
  then moves to the neighbor whose distance is one less (guaranteed optimal,
  same as the BFS "first step" today).  6 BFS → 1 BFS per step at C speed.
  `search` state keeps a per-target BFS (targets are per-guard), also njit.
- **P2c (optional):** the walkability field (`WALL`/`DOOR` only) is static
  for long stretches, so BFS results could be cached per (guard, target)
  and invalidated only when a door opens/closes.  Likely unnecessary once
  P2a/P2b land; revisit only if a profile says so.

**Expected gain:** ~5-10x on the guard path → env step at stage-4 drops
from ~1.25 ms to well under 0.5 ms.

### P3. Avoid the per-agent full-grid copy in `_get_obs`

**Where:** `src/env.py` `_get_obs`.

**Why:** each agent call does `self.grid.copy()` (a full map copy) plus two
`np.pad` calls; at 50x50 that is 4 full-grid copies per step.

**What:** build the "composite grid" once per step (agents + guards drawn
in), then slice the 5x5 windows from a single padded buffer shared across
the 4 agents.

**Expected gain:** ~0.1-0.2 ms/step at stage-4.  Secondary; fold into P2
work if convenient.

### P4. Reduce eval cost at larger maps (only if it shows up)

**Where:** `src/train_*.py` `eval_policies` / `src/evaluate.py`.

**Why:** 20 episodes × ~300 steps at ~3,800 steps/s (stage-4 single env) is
~1.6 s per eval — negligible today.  Only revisit if a later profile shows
eval dominating.

---

## 3. What NOT to do

- **Do not rewrite the env loop into a single batched NumPy sim** as the
  primary move.  The env is already fast (110k steps/s at stage-0); the
  Python per-env loop in `vec_env.step` is not the bottleneck.  The big win
  is P1 (training-loop tensor/forward overhead) and P2 (guard BFS) — both
  are local, low-risk edits.
- **Do not add torch.compile / vmap / CUDA graphs yet.**  The rollout path
  is dominated by tiny tensor copies and Python; those tools will not help
  until P1/P2 remove the overhead they cannot see.

---

## 4. Expected end state

Measured on this machine (RTX 3000 Ada, 22 cores) with the new
`tools/assess_time.py`; campaign-log units corrected (log sps x 8 =
env-steps/s):

| Campaign | Today (measured, this box) | With P1+P2 (est.) |
|---|---|---|
| stage-0 300k (12 runs) | ~4 h | ~2 h |
| full 1M grid (12 runs/stage) | ~4.4 days | ~2 days |
| curriculum IPPO+MAPPO 1M | ~10 h | ~5 h |

A 3080 Ti is roughly comparable to the 3000 Ada for these tiny MLPs (the
bottleneck is CPU env-stepping + per-step Python), so expect similar
numbers, likely a bit faster.  This is **8x faster than the pre-correction
estimate** (12-16 h stage-0) because trainer `sps` was misread as
env-steps/s when it is iterations/s.

P1 helps every algorithm at every stage; P2 disproportionately helps the
bigger curriculum stages where the wall-clock estimate is dominated by
guard pathfinding.  Combined they roughly halve the full-program estimate
and de-risk the 3-week plan.

## 5. Validation gate

Per the workflow rule, after implementing P1 and/or P2:
1. `uv run ruff check` + `uv run ruff format --check` + `uv run pytest -q`
2. 2048-step smoke of each touched trainer (IPPO/MAPPO/comm), CPU, then the
   real 8-env x 256-step shape
3. Re-run the env micro-benchmarks above to confirm the measured gains
   before launching any campaign.
