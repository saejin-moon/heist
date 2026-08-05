# HEIST: Hierarchical Environment for Interdependent Sequential Tasks

HEIST is a PettingZoo multi-agent reinforcement learning (MARL) environment and research benchmark designed to evaluate **cooperative credit assignment under sequential causal dependencies and partial observability**.

Four specialized agents—**Scout**, **Hacker**, **Muscle**, and **Extractor**—pull off a heist against rule-based security systems. Although agents act simultaneously each turn, their objectives follow a strict, recursively gated causal dependency chain:

```
Scout (Tag Terminal) ──► Hacker (Crack Terminal) ──► Extractor (Secure Loot) ──► Team (Extract)
```

---

## The Research Challenge: Causal Credit Dilution

Standard multi-agent RL algorithms (e.g., standard MAPPO or QMIX) rely on joint team rewards and assume simultaneous contribution. In an **RG-Dec-POMDP (Recursively Gated Dec-POMDP)**, this assumption breaks down due to **Causal Credit Dilution**:
* If the Extractor fails at step 50 due to a late positioning error, global negative reward (-10.0) propagates backward.
* Standard value mixers and joint critics penalize the Scout's optimal actions taken at step 5.
* Under partial observability, on-policy policy gradient methods collapse into early risk-aversion (standing still to avoid step/alarm penalties).

---

## 7-Model Benchmark Suite & Novel Mechanisms

HEIST includes a 7-model algorithm suite designed to diagnose and solve credit assignment and communication failure modes:

| Algorithm | Model Architecture | Core Mechanism |
| :--- | :--- | :--- |
| **`ippo`** | Independent PPO | Fully decentralized actor-critic per agent (baseline) |
| **`mappo`** | Centralized Critic PPO | Shared actor with centralized state-based critic |
| **`mappo_car`** | MAPPO + CAR | Intrinsic affordance reward shaping for unlocking teammate actions |
| **`comm`** | TarMAC Communication | Inter-agent attention-based message passing |
| **`comm_cir`** | Comm + CIR | Causal Information Regularization on message routing advantages |
| **`comm_cir_car`** | Comm + CIR + CAR | Combined communication, routing regularization, and affordance credit |
| **`qmix`** | Monotonic Value Decomposition | Off-policy joint value factorization ($Q_{\text{tot}}$) with hypernetwork mixer |

### Advanced Credit & Exploration Modules
* **Causal Information Regularization (CIR):** Measures message influence on value estimates by temporarily muting sender channels in `CommAgent` ([`src/train_comm.py`](file:///home/fuddle/git/heist/src/train_comm.py)), routing GAE advantage vectors back to enabling senders (`--cir-coef 0.5`).
* **Counterfactual Affordance Reward (CAR):** Grants intrinsic rewards when an agent's `INTERACT` action unlocks a teammate's dynamic action mask from 0 to 1 (`--car-coef 0.5`).
* **Random Network Distillation (RND):** Adds curiosity-driven intrinsic exploration rewards ([`src/exploration.py`](file:///home/fuddle/git/heist/src/exploration.py)) via fixed target and online predictor MSE loss to cross late-phase extraction walls (`--use-rnd --rnd-coef 0.05`).

---

## Stage-0 Baseline Results (300k Steps Benchmark)

60-episode greedy rollouts across 3 random seeds (seed 555 evaluation) on Stage 0 (11x11 grid, role-based spawns, max steps 60):

| Algorithm | Win Rate | Mean Return | Terminal Hack Rate | Loot Pickup Rate | Extraction Trigger | Mean Alarm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scripted (BFS Baseline)** | **1.000** | **+11.48** | **100%** | **100%** | **100%** | 6.0 |
| **`qmix`** | 0.000 | **-0.158** | **21.7%** | **1.7%** | **1.7%** | **14.0** |
| **`ippo`** | 0.000 | -0.546 | 3.3% | 0.0% | 0.0% | 0.2 |
| **`comm_cir`** | 0.000 | -0.596 | 0.0% | 0.0% | 0.0% | 0.0 |
| **`comm`** | 0.000 | -0.598 | 0.0% | 0.0% | 0.0% | 0.0 |
| **`mappo`** | 0.000 | -0.600 | 0.0% | 0.0% | 0.0% | 0.0 |
| **`mappo_car`** | 0.000 | -0.600 | 0.0% | 0.0% | 0.0% | 0.0 |
| **`comm_cir_car`** | 0.000 | -0.600 | 0.0% | 0.0% | 0.0% | 0.0 |

* **Scripted Baseline:** Proves the environment is 100% solvable (mean episode length 13).
* **QMIX Dominance:** Outperforms all policy-gradient baselines in task progression, achieving 21.7% terminal hacks and non-zero loot/extraction triggers.
* **CIR Communication Structuring:** CIR (`comm_cir`) successfully breaks uniform attention noise into structured role-based communication channels.

---

## Setup and Installation

Requirements: Python 3.12+ and `uv` package manager.

```bash
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked
```

Run test suite and static analysis (33 unit tests):

```bash
uv run pytest -v
uv run ruff check
```

---

## Campaign Execution & Evaluation CLI

Use `train.zsh` to launch multi-model training campaigns:

```bash
# Fast validation test across all 7 models (10,240 steps)
./train.zsh --fast --rnd

# Parallel execution with RND curiosity & custom step budget
./train.zsh --steps 1000000 --parallel 2 --rnd --rnd-coef 0.05

# Background daemon campaign across curriculum stages 0 and 1
./train.zsh --stages 0,1 --steps 1000000 --daemon
```

Monitor live status and logs:

```bash
uv run python tools/status.py --watch
```

Evaluate checkpoints and merge stage results:

```bash
# Evaluate Stage 0 checkpoints into a structured run directory
uv run python src/eval_stage.py --stage 0 --episodes 60

# Merge individual algorithm evaluations into summary.json
uv run python src/eval_stage.py --stage 0 --merge --run-id run017
```

---

## Codebase Layout

```
train.zsh                     Multi-model campaign launcher & process manager
tools/status.py               Live training log and checkpoint monitor
tools/assess_time.py          Hardware benchmark & throughput calculator
src/env.py                    PettingZoo parallel environment engine
src/curriculum.py             5-stage curriculum generator (11x11 to 50x50)
src/vision.py                 Numba JIT raycasting & fog-of-war engine
src/model.py                  Neural network policies & attention communication
src/vec_env.py                Multiprocessing vectorized environment wrapper
src/exploration.py            RND & count-based intrinsic curiosity modules
src/eval_stage.py             Post-experiment evaluation & JSON merging CLI
src/scripted.py               Near-optimal BFS controller baseline
src/train_ippo.py             Independent PPO trainer
src/train_mappo.py            MAPPO trainer with CAR support
src/train_comm.py             TarMAC trainer with CIR support
src/train_qmix.py             QMIX trainer with value decomposition
tests/                        Comprehensive Pytest unit test suite (33 tests)
results/                      Run artifacts, JSON summaries, and benchmark logs
```
