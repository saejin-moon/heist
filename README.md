# HEIST: Hierarchical Environment for Interdependent Sequential Tasks
HEIST is a PettingZoo-compliant Multi-Agent Reinforcement Learning (MARL) benchmark specifically engineered to stress-test cooperative credit assignment. By forcing agents to navigate parallel physical spaces while remaining strictly gated by sequential causal dependencies and dynamic partial observability, HEIST isolates and exposes the failure modes of standard value-decomposition algorithms (e.g., QMIX, MAPPO).

## Causal Credit Dilution
Standard cooperative MARL algorithms (such as QMIX and MAPPO) rely on shared reward functions, assuming agents contribute to the global state simultaneously. However, HEIST introduces Causal Credit Dilution. While agents navigate the environment in parallel, their objective interactions are bound by strict sequential dependencies (e.g., the Extractor cannot secure the loot until the Hacker disables the terminal). Consequently, if a downstream agent fails late in the episode, the shared negative reward propagates backward, diluting the credit upstream agents deserved for executing their prerequisites flawlessly.

## Environment Architecture
To preserve the Markov Property under strict causal gating, HEIST abandons standard flat observation arrays in favor of a multi-tensor `Dict` space for each agent:

1. **`observation` (5x5 matrix):** The agent's local physical view, restricted by a dynamic Bresenham-raycast Fog of War. 
2. **`action_mask` (6-element vector):** Mathematically enforces the sequential causal gates. Downstream actions (e.g., extracting loot) are dynamically masked out until upstream dependencies are resolved, preventing the policy network from wasting gradient updates on impossible actions.
3. **`role_id` (4-element one-hot):** Identifies the agent's role (Scout, Hacker, Muscle, Extractor) so shared policies can distinguish between roles without aliasing (REV-2).

REV-7 (see `REVISION_PLAN.md §6`) removed the former `global_state` broadcast
from the per-agent observation contract.  Agents must now communicate phase
status (e.g., "terminal hacked") through learned TarMAC-style messages; a
centralized critic (`env.state()`) remains available for MAPPO/QMIX value
estimation but is not part of the per-agent observation dict.

## Adversary and Loss Conditions
To prevent the environment from degrading into a trivial pathfinding task, HEIST introduces two opposing adversarial pressures. First, Rule-Based Adversaries (Guards) execute dynamic random-walk patrols. If a guard's Manhattan distance to any agent closes to $\le 1$, a global alarm triggers, terminating the episode with a catastrophic `-10.0` shared reward. Second, a constant time penalty (`-0.01` per step) acts as a baseline bleed. This dual-pressure system prevents policy collapse: agents cannot safely sprint blindly to objectives, nor can they exploit a "hide-and-wait" policy to avoid the guards entirely.

## Installation and Developer Tooling
`uv` must be installed before running this repository.
```bash
# clone
git clone https://github.com/saejin-moon/heist.git
cd heist
# create env and install pinned deps (numpy<2.5 is required for numba compatibility)
uv sync
```
`uv sync` installs the runtime deps declared in `pyproject.toml`
(`[project].dependencies`) plus the dev group (`pytest`, `ruff`) and is the
single entry point; `requirements.txt` is kept in sync for legacy pip users.

Always invoke commands through `uv run` (never `source activate`):
```bash
uv run python src/manual_control.py                              # human play
uv run python -m pettingzoo.test.parallel_test -e src.dummy:make_env  # API smoke
uv run pytest                                                     # 11 unit/mechanics smokes
uv run ruff check                                                 # lint (E/F/W/I/UP/B/SIM)
uv run ruff format --check                                        # formatting
```

## Full research experiment (300k, stage-0)

> **Status: SETUP ONLY. This campaign is not scheduled to run yet.** The
> intent is to launch it later on a dedicated machine. Do not run the
> end-to-end campaign on this box; only short trainer smokes are expected
> during setup. See `FUTURE_PLANS.md §1` and the banner at the top of
> `experiment_stage0_300k.sh`.

### 1. Set up the environment on the target machine
```bash
# requires: uv, a CUDA-capable GPU, ~4-8h of wall time for all 12 runs
# one-shot alternative (installs system pkgs + uv, clones, syncs, gates,
# smokes, then runs the campaign; --daemon backgrounds it):
#   zsh train-stage-0.zsh [--daemon] [--eval]
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked        # install pinned deps from uv.lock (fails if stale)
uv run pytest -q        # 11 unit/mechanics smokes must pass
uv run ruff check       # lint must pass
```
Verify the GPU is visible to torch:
```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

### 1b. Estimate wall-clock time on THIS machine first
Before launching anything, measure real throughput and get campaign
estimates (env speed per stage, rollout overhead, end-to-end trainer
steps/s for each algo, and extrapolated hours for the full 1M grid):
```bash
./assess-time.zsh                    # full assessment (~5-10 min)
./assess-time.zsh --trainer-steps 10240   # faster, less precise
```
Notes: the trainer logs `sps` in rollout-iterations/s (each iteration steps
8 envs) -- multiply by `num_envs` for env-steps/s.  This is why campaign
estimates here are ~8x smaller than the earlier pre-correction guesses.

### 2. Launch the campaign
```bash
# Sequential driver: IPPO, MAPPO, QMIX, comm at 300k steps on stage-0
# (seeds 0-2). Logs to log/experiment_stage0_300k.log. Resumable: reruns
# skip any <algo>_s<seed> that already has a checkpoint in checkpoints/.
mkdir -p log           # log/ is gitignored, not present on a fresh clone
nohup bash experiment_stage0_300k.sh > log/launch.out 2>&1 &

# watch progress
tail -f log/experiment_stage0_300k.log
```
Each trainer logs to TensorBoard under `runs/<algo>_s<seed>/`:
```bash
uv run tensorboard --logdir runs --port 6006
```

### 3. Evaluate after the campaign finishes
```bash
# G4 comparison: win rate + CAI/counterfactual (IPPO/MAPPO/QMIX) and
# message-outcome correlation (comm), 60 episodes, seed 555.
# Writes results/stage0_300k_comparison.json
uv run python src/eval_stage0_300k.py
```
Refresh `PLAN.md` "Baseline Validation Campaign" and `results/README.md`
with the resulting tables.

## Project Layout
```
PERFORMANCE.md      measured bottlenecks + prioritized optimization plan
train-stage-0.zsh   one-shot provision + run for the target machine
assess-time.zsh     throughput benchmark + wall-clock estimate wrapper
tools/assess_time.py  the benchmark itself (env, rollout, trainer sps, extrapolation)
src/constants.py    reward/alarm constants, ROLE_ACTIONS, tile palette
src/map_gen.py      procedural map generation (rooms, doors, cameras, spawns)
src/vision.py       numba-JIT Bresenham line-of-sight + camera exposure
src/env.py          HeistEnv, the PettingZoo ParallelEnv (causal gating)
src/vec_env.py      vectorized env wrapper for PPO-style rollouts
src/model.py        HeistAgent (PPO actor-critic), MappoAgent, QMixNet + mixer
src/train_ippo.py   independent PPO baseline (--shared for param sharing)
src/train_mappo.py  shared actor + centralized critic baseline
src/train_qmix.py   value-decomposition baseline (replay buffer + monotonic mixer)
src/train_comm.py   REV-7 learned-communication baseline (TarMAC messages)
src/evaluate.py     win rate, Credit Attribution Index, counterfactual importance
src/curriculum.py   5 staged configs from 11x11 (no security) to 50x50 (full)
src/dummy.py        random-policy smoke-test entrypoint
src/test_qmix_smoke.py  standalone QMIX logic test
```

## Training

All trainers are CleanRL-style scripts with TensorBoard logging, checkpointing
to `src/runs/<algo>_s<seed>/`, and `--env-config` taking a **JSON string**
(comma-separated key=value breaks on tuple values like map_size).

```bash
# Independent PPO, tiny stage-0 map, smoke test (~1 min on CUDA)
uv run python src/train_ippo.py --total-timesteps 2048 --num-envs 2 --num-steps 128 \
  --eval-every 2 --eval-episodes 2 \
  --env-config '{"map_size": [11, 11], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 60}'

# MAPPO (shared actor, centralized critic)
uv run python src/train_mappo.py --total-timesteps 200000 --num-envs 8 --num-steps 256 \
  --env-config '{"map_size": [15, 15], "guard_count": 2, "camera_count": 0, "max_steps": 120}'

# QMIX (off-policy; use --no-cuda for quick CPU smoke tests to skip triton JIT warmup)
uv run python src/train_qmix.py --total-steps 100000 --no-cuda \
  --env-config '{"map_size": [15, 15], "guard_count": 0, "camera_count": 0, "max_steps": 120}'

# Comm (REV-7): shared TarMAC policy with learned messages
uv run python src/train_comm.py --total-steps 100000 \
  --env-config '{"map_size": [11, 11], "guard_count": 0, "camera_count": 0, "max_steps": 120}'

# Curriculum: stage-0 config up to the full 50x50 benchmark
uv run python -c "from curriculum import CURRICULUM; [print(s) for s in CURRICULUM]"
```

Common flags: `--no-cuda`, `--no-save-model`, `--seed`, `--eval-every`,
`--eval-episodes`, `--anneal-lr`, and algorithm hyperparameters
(`--gamma`, `--gae-lambda`, `--clip-coef`, `--learning-rate`, ...).

## Evaluation

`evaluate.py` measures the benchmark's novelty hook, **Causal Credit
Dilution**, two ways:

* **Credit Attribution Index (CAI):** Pearson correlation between each
  agent's per-episode shaped credit (excluding the shared terminal reward)
  and the episode outcome. Upstream agents (scout/hacker) whose credit stays
  high while predicting wins are healthy; dilution shows as upstream credit
  decoupling from outcome.
* **Counterfactual importance:** win-rate drop when each agent is replaced
  by a no-op, compared to the baseline team.

```bash
uv run python -c "
from env import HeistEnv
from model import HeistAgent
from evaluate import summarize
env = HeistEnv({'map_size': (11,11), 'guard_count': 0, 'camera_count': 0, 'max_steps': 60})
summarize({a: HeistAgent() for a in env.agents}, env, episodes=30, seed=0, device='cpu')
"
```

## Known Issues

* **Baselines learn slowly at stage-0.** With 300k steps IPPO reaches a
  noisy 3-10% win rate; the causal chain completes 97%/87%/87%
  (terminal/loot/extraction) but the final convergence is the bottleneck.
  The env itself is solvable (a BFS+wait scripted controller wins 29/30).
  With `spawn_mode="role"` the early chain is nearly free (agents spawn
  beside terminal/loot) while the extract tile is far, so the decisive
  skill is navigation the small MLPs acquire slowly. Full campaign record
  in PLAN.md "Baseline Validation Campaign".
* **QMIX first GPU run** spends minutes in triton JIT compilation; either
  pass `--no-cuda` for short smoke tests or raise the timeout for the first
  real run.
* **Throughput** is currently ~40-70 SPS (CUDA) due to per-step Python env
  stepping plus numba JIT warmup; the numba kernels compile once per process.