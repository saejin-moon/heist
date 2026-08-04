# HEIST: Hierarchical Environment for Interdependent Sequential Tasks

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PettingZoo Compatible](https://img.shields.io/badge/PettingZoo-1.24+-green.svg)](https://pettingzoo.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HEIST** is a PettingZoo-compliant Multi-Agent Reinforcement Learning (MARL) benchmark specifically engineered to isolate, measure, and solve **Causal Credit Dilution** in cooperative multi-agent teams. 

By forcing four heterogeneous agents (**Scout**, **Hacker**, **Muscle**, **Extractor**) to operate in parallel physical views while strictly gated by a sequential causal dependency chain against dynamic rule-based security adversaries (Guards & Cameras), HEIST exposes the fundamental failure modes of standard value-decomposition and centralized-critic algorithms (e.g., QMIX, MAPPO).

---

## 🔬 Core Problem: Causal Credit Dilution

Standard cooperative MARL algorithms rely on shared team rewards, implicitly assuming that all agents contribute to global state transitions simultaneously. However, real-world multi-agent systems operate under **sequential causal dependencies**:

$$\text{Scout (Tag Terminal)} \longrightarrow \text{Hacker (Disable Security)} \longrightarrow \text{Extractor (Secure Loot)} \longrightarrow \text{Team (Extract)}$$

If a downstream agent (e.g., Extractor) fails late in the episode, standard algorithms propagate a negative team reward backward to all agents. This **dilutes upstream credit**, penalizing upstream agents (e.g., Scout and Hacker) even when they executed their prerequisite tasks flawlessly.

---

## ⚡ Novel Credit Assignment Algorithms

HEIST introduces two novel, theoretically grounded algorithms designed to eliminate Causal Credit Dilution:

### 1. CIR (Causal Influence Routing)
* **Concept:** Counterfactual message ablation in learned communication channels (`src/model.py:CommAgent.get_influence_matrix`).
* **Mechanism:** Measures the absolute change in a receiver agent's value function $V_r$ when muting each sender's TarMAC message $\mathbf{m}_s$. The normalized influence matrix $\mathbf{I}_{r,s}$ routes GAE advantage vectors from receivers back to the senders who causally enabled their reward:
$$\mathbf{A}_{routed} = \text{einsum}(\mathbf{I}, \mathbf{A})$$
$$\mathbf{A}_{final} = (1 - \alpha) \cdot \mathbf{A} + \alpha \cdot \mathbf{A}_{routed}$$
* **Usage:** `uv run python src/train_comm.py --cir-coef 0.5`

### 2. CAR (Counterfactual Affordance Reward)
* **Concept:** Intrinsic motivation for unlocking action space affordances (`src/env.py:HeistEnv.step`).
* **Mechanism:** Detects when an agent's `INTERACT` action flips a teammate's `action_mask` from $0 \to 1$ (expanding their legal action space). The unlocking agent receives an intrinsic reward bonus proportional to the Centralized Critic's valuation of the newly unlocked state:
$$R_{bonus} = \beta \cdot \max(0.0, V(S_{next}))$$
* **Usage:** `uv run python src/train_mappo.py --car-coef 0.5`

---

## 🎮 Environment Architecture

Each agent observes a multi-tensor `Dict` observation space designed to preserve the Markov property under partial observability:

1. **`observation` (5x5 matrix):** Local physical view restricted by dynamic Bresenham-raycast Fog of War.
2. **`action_mask` (6-element vector):** Enforces causal gates (e.g., extracting loot is masked until the security terminal is hacked).
3. **`role_id` (4-element one-hot):** Identifies agent specialization (`[1,0,0,0]` Scout, `[0,1,0,0]` Hacker, `[0,0,1,0]` Muscle, `[0,0,0,1]` Extractor).

### Adversaries & Environmental Dynamics
* **Guards:** Dynamic rule-based adversaries executing random-walk patrols and line-of-sight sweeps. Close proximity ($\le 1$ Manhattan distance) triggers a global alarm and terminates the episode with a catastrophic `-10.0` team penalty.
* **Time Bleed:** Baseline `-0.01` per-step penalty preventing policy stagnation and infinite looping.

---

## 🚀 Quick Start & Installation

### 1. Installation
The repository uses `uv` for ultra-fast, reproducible dependency management:

```bash
# Clone the repository
git clone https://github.com/saejin-moon/heist.git
cd heist

# Install CPython 3.12 and pinned dependencies
uv sync --locked
```

### 2. Developer Quality Gates
Run test suites and linters via `uv`:
```bash
uv run pytest -q                 # Run all 18 unit and mechanics smoke tests
uv run ruff check                # Run Linter
uv run ruff format --check       # Check formatting
```

---

## 🏃 Running Training Campaigns (`train.zsh`)

[`train.zsh`](file:///home/fuddle/git/heist/train.zsh) is the automated entrypoint for provisioning packages, verifying CUDA hardware, running validation gates, and launching multi-stage MARL campaigns.

```bash
# Quick 15-second sample test across all algorithm variants (IPPO, MAPPO, CAR, Comm, CIR, QMIX)
./train.zsh --sample

# Custom hyperparameter tuning (20,480 steps on Stage 0 with CIR and CAR)
./train.zsh --steps 20480 --cir-coef 0.3 --car-coef 0.4

# Launch full 5-stage research campaign as a background daemon
./train.zsh --num-stages 5 --daemon

# Print command-line options and usage help
./train.zsh --help
```

### Live Status Monitoring (`tools/status.py`)
Monitor active training runs, throughput (SPS), checkpoint completion, and log tails in real-time:
```bash
uv run python tools/status.py --watch
```

---

## 📊 Hardware Benchmarking (`assess-time.zsh`)

Measure exact CUDA rollout latency and extrapolate full campaign wall-clock times:

```bash
./assess-time.zsh
```

### Benchmark Results (NVIDIA RTX 3000 Ada / 22 CPU Cores)
| Algorithm | Stage-0 SPS | Stage-4 SPS | 5-Stage 1M Campaign Est. |
| :--- | :---: | :---: | :---: |
| **MAPPO / MAPPO+CAR** | **1,076 steps/s** | 219 steps/s | ~32 Hours |
| **TarMAC / Comm+CIR** | **1,086 steps/s** | 219 steps/s | ~32 Hours |
| **IPPO** | **871 steps/s** | 208 steps/s | ~38 Hours |
| **QMIX** | **498 steps/s** | 278 steps/s | ~42 Hours |

---

## 📁 Repository Structure

```
.
├── train.zsh           # Main entrypoint script for provisioning, setup, & campaign runs
├── assess-time.zsh     # Wall-clock throughput benchmark wrapper
├── tools/
│   ├── assess_time.py  # Empirically measured hardware throughput & campaign estimator
│   └── status.py       # Live training status monitor & log tailer
├── src/
│   ├── env.py          # PettingZoo HeistEnv implementation (causal gating & CAR tracking)
│   ├── model.py        # Neural networks (CommAgent with CIR matrix, HeistAgent, QMixer)
│   ├── vec_env.py      # Vectorized multi-agent environment wrapper
│   ├── vision.py       # Numba JIT-compiled Bresenham LOS & BFS pathfinding
│   ├── constants.py    # Environment constants, tile palette, and role definitions
│   ├── map_gen.py      # Procedural map generator
│   ├── curriculum.py   # 5 staged environment configurations (11x11 to 50x50)
│   ├── ppo_utils.py    # Vectorized GAE advantage calculation
│   ├── train_ippo.py   # Independent PPO baseline
│   ├── train_mappo.py  # MAPPO baseline + CAR intrinsic rewards (--car-coef)
│   ├── train_comm.py   # TarMAC communication baseline + CIR advantage routing (--cir-coef)
│   ├── train_qmix.py   # QMIX value-decomposition baseline (batched GPU selection)
│   ├── evaluate.py     # Evaluation metrics (Win Rate, Credit Attribution Index)
│   ├── test_cir_smoke.py  # CIR influence matrix & routing unit test
│   ├── test_car_smoke.py  # CAR affordance unlock unit test
│   └── test_qmix_opt_smoke.py # QMIX GPU action selection unit test
├── pyproject.toml      # Project configuration & dependencies
└── uv.lock             # Pinned dependency lockfile
```

---

## 📜 Citation & License

If you use HEIST, CIR, or CAR in your research, please cite this repository:

```bibtex
@software{heist_marl_2026,
  author = {Moon, Saejin},
  title = {HEIST: Hierarchical Environment for Interdependent Sequential Tasks},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/saejin-moon/heist}}
}
```

Licensed under the [MIT License](LICENSE).