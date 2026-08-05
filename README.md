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

## 10-Model Benchmark Suite & Novel Mechanisms

HEIST includes a 10-model algorithm suite covering the major MARL paradigms (Independent, CTDE Value-Based, CTDE Policy-Based, CTDE Communication, and Causal Credit Routing):

| Algorithm | Model Architecture | Core Mechanism |
| :--- | :--- | :--- |
| **`ippo`** | Independent PPO | Fully decentralized actor-critic per agent (baseline) |
| **`mappo`** | Centralized Critic PPO | Shared actor with centralized state-based $V(s)$ critic |
| **`mappo_car`** | MAPPO + CAR | Intrinsic affordance reward shaping for unlocking teammate actions |
| **`mappo_cir`** | MAPPO + Non-Comm CIR | Causal Influence Routing via state feature ablation ($V(s)$ sensitivity) |
| **`comm`** | TarMAC Communication | Inter-agent attention-based message passing |
| **`comm_cir`** | Comm + CIR | Causal Influence Routing via counterfactual message ablation |
| **`comm_cir_car`** | Comm + CIR + CAR | Combined communication, message routing, and affordance credit |
| **`qmix`** | Monotonic Value Factorization | Off-policy joint value factorization ($Q_{\text{tot}}$) with hypernetwork mixer |
| **`coma`** | Counterfactual Policy Gradient | Centralized critic $Q_i(s, \mathbf{a}_{-i}, a_i)$ with counterfactual baseline |
| **`coma_cir`** | COMA + Non-Comm CIR | Causal Influence Routing via counterfactual action ablation ($Q_i$ sensitivity) |

### Advanced Credit & Exploration Modules
* **Causal Information Regularization (CIR):** Measures sender influence on receiver value/Q estimates:
  - **Communicating (`comm_cir`):** Evaluated via *Counterfactual Message Ablation* (zeroing message vectors).
  - **Non-Communicating (`mappo_cir`, `coma_cir`):** Evaluated via *Counterfactual Feature/Action Ablation* (zeroing spatial features in $V(s)$ or joint action vectors in $Q_i$).
  - **Advantage Routing:** Routes advantage vectors back to enabling senders: $\tilde{A}_i = (1 - \alpha) A_i + \alpha \sum_j M_{ij} A_j$ (`--cir-coef 0.5`).
* **Counterfactual Affordance Reward (CAR):** Grants intrinsic rewards when an agent's `INTERACT` action unlocks a teammate's dynamic action mask from 0 to 1 (`--car-coef 0.5`).
* **Random Network Distillation (RND):** Adds curiosity-driven intrinsic exploration rewards ([`src/exploration.py`](file:///home/fuddle/git/heist/src/exploration.py)) via fixed target and online predictor MSE loss (`--use-rnd --rnd-coef 0.05`).

---

## Benchmark Results (Stage 0, 1M Steps & Fast Validation)

### 1,000,000-Step Campaign Benchmark (Stage 0)

60-episode greedy rollouts across 3 random seeds on Stage 0 (11x11 grid, max steps 60):

| Algorithm | Win Rate | Mean Return | Terminal Hack Rate | Loot Pickup Rate | Extraction Rate | Mean Alarm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scripted (BFS Baseline)** | **1.000** | **+11.48** | **100%** | **100%** | **100%** | 6.0 |
| **`mappo_car` (MAPPO + CAR)** | **0.083** | **+1.534** | **93.3%** | **80.0%** | **80.0%** | **30.5** |
| **`ippo`** | 0.050 | +1.210 | 98.3% | 81.7% | 81.7% | 33.1 |
| **`mappo`** | 0.050 | +1.041 | 90.0% | 66.7% | 66.7% | 25.8 |
| **`comm_cir_car`** | 0.000 | -0.465 | 3.3% | 3.3% | 3.3% | 3.7 |
| **`comm_cir`** | 0.000 | -0.535 | 0.0% | 0.0% | 0.0% | 0.4 |
| **`comm`** | 0.000 | -0.590 | 0.0% | 0.0% | 0.0% | 0.1 |
| **`qmix`** | 0.000 | -0.440 | 0.0% | 0.0% | 0.0% | 6.8 |

### Fast 10k-Step Validation Benchmark (COMA & Non-Comm CIR)

Evaluating early counterfactual credit attribution across COMA and CIR variants:

| Model | Win Rate | Terminal Hack Rate | Mean Return | Mean Length | Mean Alarm |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`coma`** | 0.000 | **0.183 (18.3%)** | -6.971 | 15.0 | 95.0 |
| **`coma_cir`** | 0.000 | **0.183 (18.3%)** | -7.460 | 12.6 | 97.0 |
| **`mappo_cir`** | 0.000 | 0.000 (0.0%) | -0.600 | 60.0 | 0.0 |
| **`qmix`** | 0.000 | 0.217 (21.7%) | -0.158 | 60.0 | 14.0 |

* **CAR Affordance Dominance:** `mappo_car` achieves the **highest win rate (8.3%)** and **highest mean return (+1.534)** at 1M steps by resolving credit dilution.
* **COMA Counterfactual Precision:** `coma` and `coma_cir` achieve **18.3% terminal hack rates** within 10k steps, rapidly isolating agent action contributions via counterfactual baselines.
* **Scripted Baseline:** Proves environment 100% solvability (mean episode length 13 steps).

---

## Setup and Installation

Requirements: Python 3.12+ and `uv` package manager.

```bash
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked
```

Run Pytest suite (38 unit tests):

```bash
uv run pytest -v
uv run ruff check
```

---

## Campaign Execution & Model Filtering CLI

Use `train.zsh` to launch multi-model training campaigns:

```bash
# Fast validation test across all 10 models (10,240 steps)
./train.zsh --fast

# Train specific model(s) (e.g. coma, coma_cir, mappo_cir)
./train.zsh --models coma,coma_cir,mappo_cir --stages 0,1

# Parallel execution with custom step budget
./train.zsh --steps 300000 --parallel 4 --models ippo,mappo,qmix,coma

# Background daemon campaign across curriculum stages 0 and 1
./train.zsh --stages 0,1 --steps 300000 --daemon
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
uv run python src/eval_stage.py --stage 0 --merge --run-id run021
```

---

## Codebase Layout

```
train.zsh                     Multi-model campaign launcher with --models filtering
tools/status.py               Live training log and checkpoint monitor
tools/assess_time.py          Hardware benchmark & throughput calculator
src/env.py                    PettingZoo parallel environment engine
src/curriculum.py             6-stage curriculum generator (11x11 to 50x50)
src/vision.py                 Numba JIT raycasting & fog-of-war engine
src/model.py                  Neural network policies, TarMAC, MAPPO, and COMA models
src/ppo_utils.py              Counterfactual advantage math & checkpoint transfers
src/vec_env.py                Multiprocessing vectorized environment wrapper
src/exploration.py            RND intrinsic curiosity module
src/eval_stage.py             Post-experiment evaluation & JSON merging CLI
src/scripted.py               Near-optimal BFS controller baseline
src/train_ippo.py             Independent PPO trainer
src/train_mappo.py            MAPPO trainer (with CAR and CIR support)
src/train_coma.py             COMA trainer (with counterfactual & CIR support)
src/train_comm.py             TarMAC trainer (with CIR and CAR support)
src/train_qmix.py             QMIX trainer with value decomposition
tests/                        Comprehensive Pytest unit test suite (38 tests)
results/                      Run artifacts, JSON summaries, and benchmark logs
```
