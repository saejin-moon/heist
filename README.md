# HEIST: Hierarchical Environment for Interdependent Sequential Tasks

HEIST is a PettingZoo multi-agent reinforcement learning benchmark. The environment evaluates cooperative credit assignment in multi-agent teams. 

Four specialized agents (Scout, Hacker, Muscle, Extractor) work together in a grid. Although agents move simultaneously, their objectives follow a strict causal order. The Scout tags the security terminal. Then the Hacker disables the security system. Next, the Extractor secures the loot. Finally, all four agents must reach the extraction tile.

```
Scout (Tag Terminal) -> Hacker (Disable Terminal) -> Extractor (Get Loot) -> Team (Extract)
```

## Causal Credit Dilution

Standard multi-agent algorithms rely on shared team rewards. They assume agents contribute to state changes at the same time. But HEIST forces agents into sequential dependencies. 

When a downstream agent fails late in an episode, the environment issues a negative team reward. Shared reward mechanisms push this negative signal backward to all agents. So upstream agents receive penalties even when they completed their required tasks. 

## Causal Credit Assignment Algorithms

HEIST includes two credit assignment algorithms designed to fix credit dilution.

### Causal Influence Routing (CIR)

CIR measures message influence in learned communication channels. The algorithm temporarily mutes each sender's message in the `CommAgent` policy network (`src/model.py`). Then it records the change in the receiver's value estimate. 

These value changes form an influence matrix. The trainer uses this matrix to route GAE advantage vectors from receiver agents back to the specific senders that enabled the reward:

$$\mathbf{A}_{routed} = \text{einsum}(\mathbf{I}, \mathbf{A})$$
$$\mathbf{A}_{final} = (1 - \alpha) \cdot \mathbf{A} + \alpha \cdot \mathbf{A}_{routed}$$

To enable CIR during training:
`uv run python src/train_comm.py --cir-coef 0.5`

### Counterfactual Affordance Reward (CAR)

CAR awards intrinsic rewards for expanding a teammate's action space. The environment tracks action masks during every step (`src/env.py`). When an agent takes an `INTERACT` action that turns a teammate's action mask from 0 to 1, the environment flags an affordance unlock.

The trainer grants the unlocking agent an intrinsic reward bonus based on the centralized critic's state evaluation:

$$R_{bonus} = \beta \cdot \max(0.0, V(S_{next}))$$

To enable CAR during training:
`uv run python src/train_mappo.py --car-coef 0.5`

## Environment Mechanics

Agents receive local observation dictionaries containing three tensors:

1. `observation` (5x5 matrix): A local grid view masked by raycasted Fog of War.
2. `action_mask` (6-element vector): Binary mask that blocks invalid or causally locked actions.
3. `role_id` (4-element vector): One-hot vector identifying the agent's assigned role.

Guards patrol the grid using random walks. If a guard moves within 1 tile of an agent, the alarm triggers and ends the episode with a -10.0 penalty. Every step also incurs a -0.01 time penalty to prevent looping.

## Setup and Development

Install CPython 3.12 and dependencies using `uv`:

```bash
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked
```

Run tests and code checks:

```bash
uv run pytest -q
uv run ruff check
uv run ruff format --check
```

## Running Training (`train.zsh`)

Use `train.zsh` to set up environments, verify CUDA availability, and run training campaigns.

```bash
# Run a 10,240-step sample run across all algorithms
./train.zsh --sample

# Run custom coefficient tuning on Stage 0
./train.zsh --steps 20480 --cir-coef 0.3 --car-coef 0.4

# Launch a background campaign across all 5 curriculum stages
./train.zsh --num-stages 5 --daemon
```

Monitor active runs with the status tool:

```bash
uv run python tools/status.py --watch
```

## Throughput Benchmarks (`assess-time.zsh`)

Run `./assess-time.zsh` to measure step latency and campaign runtime.

Measurements on an NVIDIA RTX 3000 Ada GPU:

* MAPPO: 1,076 steps/second (Stage 0)
* TarMAC Comm: 1,086 steps/second (Stage 0)
* IPPO: 871 steps/second (Stage 0)
* QMIX: 498 steps/second (Stage 0)

A full 5-stage campaign across all algorithms takes 32 hours on one GPU.

## Codebase Layout

```
train.zsh               Campaign launcher and environment setup
assess-time.zsh         Benchmark script wrapper
tools/assess_time.py    Hardware benchmark and throughput calculator
tools/status.py         Training status and log monitor
src/env.py              PettingZoo environment implementation
src/model.py            Policy networks and CIR influence matrix
src/vec_env.py          Vectorized environment wrapper
src/vision.py           Numba JIT line-of-sight and pathfinding
src/train_ippo.py       Independent PPO trainer
src/train_mappo.py      MAPPO trainer with CAR support
src/train_comm.py       TarMAC trainer with CIR support
src/train_qmix.py       QMIX trainer with GPU action selection
src/test_cir_smoke.py   CIR unit test
src/test_car_smoke.py   CAR unit test
```
