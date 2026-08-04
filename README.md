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

## Novel Credit Assignment Algorithms
HEIST introduces two novel, theoretically grounded MARL credit-assignment algorithms designed to conquer Causal Credit Dilution:

1. **CIR (Causal Influence Routing):** Counterfactual message ablation in the TarMAC communication channel (`src/model.py:CommAgent.get_influence_matrix`). By measuring the absolute change in a receiver's value function when muting each sender, GAE advantage vectors are causally routed from receivers back to the senders who enabled them (`--cir-coef`).
2. **CAR (Counterfactual Affordance Reward):** Detects when an agent's `INTERACT` action flips a teammate's `action_mask` from 0 to 1 (`src/env.py:HeistEnv.step`). Unlocking a causal affordance awards an intrinsic reward proportional to the Centralized Critic's valuation of the new state (`--car-coef`).

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
uv run pytest                                                     # 18 unit/mechanics smokes
uv run ruff check                                                 # lint (E/F/W/I/UP/B/SIM)
uv run ruff format --check                                        # formatting
uv run python tools/status.py                                    # active training status
```

## Quick Start & Research Campaign (`train.zsh`)

The entrypoint script `train.zsh` provisions dependencies, verifies GPU CUDA availability, runs all validation gates, and executes multi-stage campaigns.

```bash
# 1. Run a 15-second sample test across all algorithm variants:
./train.zsh --sample

# 2. Run custom coefficient tuning (20,480 steps on Stage 0):
./train.zsh --steps 20480 --cir-coef 0.3 --car-coef 0.4

# 3. Launch full campaign in background (daemon mode):
./train.zsh --num-stages 5 --daemon

# 4. Monitor live training status & log tailing:
uv run python tools/status.py --watch
```

### Wall-Clock Assessment (`assess-time.zsh`)
Measure exact CUDA throughput and extrapolated wall-clock times for your hardware:
```bash
./assess-time.zsh
```
*Current benchmark throughput on CUDA (NVIDIA RTX 3000 Ada): MAPPO/Comm ~1080 steps/s, IPPO ~870 steps/s, QMIX ~500 steps/s. Full 5-stage campaign completes in ~32 hours.*

## Project Layout
```
train.zsh           one-shot provision, setup, and campaign runner
assess-time.zsh     throughput benchmark + wall-clock estimate wrapper
tools/assess_time.py  the benchmark itself (env, rollout, trainer sps)
tools/status.py     live training status monitor and log tailer
src/constants.py    reward/alarm constants, ROLE_ACTIONS, tile palette
src/map_gen.py      procedural map generation (rooms, doors, cameras, spawns)
src/vision.py       numba-JIT Bresenham line-of-sight + camera exposure
src/env.py          HeistEnv, PettingZoo ParallelEnv (causal gating & CAR tracking)
src/vec_env.py      vectorized env wrapper for PPO/QMIX rollouts
src/model.py        HeistAgent, CommAgent (CIR influence matrix), QMixNet + mixer
src/train_ippo.py   independent PPO baseline
src/train_mappo.py  shared actor + centralized critic + CAR intrinsic reward
src/train_qmix.py   value-decomposition baseline (batched GPU action selection)
src/train_comm.py   TarMAC communication baseline + CIR advantage routing
src/evaluate.py     win rate, Credit Attribution Index, counterfactual importance
src/curriculum.py   5 staged configs from 11x11 (no security) to 50x50 (full)
src/test_cir_smoke.py CIR influence matrix & routing unit test
src/test_car_smoke.py CAR affordance unlock & info pass-through unit test
```

## Training

All trainers are CleanRL-style scripts with TensorBoard logging, checkpointing
to `checkpoints/<run_name>/`, and `--env-config` taking a **JSON string**.

```bash
# Independent PPO, tiny stage-0 map
uv run python src/train_ippo.py --total-timesteps 20480 --num-envs 8 --num-steps 256

# MAPPO + CAR (Counterfactual Affordance Rewards)
uv run python src/train_mappo.py --total-timesteps 200000 --num-envs 8 --num-steps 256 --car-coef 0.5

# TarMAC Comm + CIR (Causal Influence Routing) + CAR
uv run python src/train_comm.py --total-steps 200000 --num-envs 8 --num-steps 256 --cir-coef 0.5 --car-coef 0.5

# QMIX (off-policy value-decomposition)
uv run python src/train_qmix.py --total-steps 200000 --train-freq 4
```

Common flags: `--no-cuda`, `--no-save-model`, `--seed`, `--eval-every`,
`--eval-episodes`, `--anneal-lr`, `--cir-coef`, `--car-coef`.

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