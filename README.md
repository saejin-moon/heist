# HEIST: Hierarchical Environment for Interdependent Sequential Tasks

HEIST is a PettingZoo multi-agent reinforcement learning (MARL) environment and research benchmark designed to evaluate **cooperative credit assignment under sequential causal dependencies and partial observability**.

Four specialized agents—**Scout**, **Hacker**, **Muscle**, and **Extractor**—coordinate to pull off a heist against a dynamic, rule-based security system (guards, cameras, and an alarm meter). While agents act simultaneously each step, their actions follow a strict, recursively gated causal dependency chain:

```
  ┌─────────┐         ┌─────────┐         ┌───────────┐         ┌───────────┐
  │  SCOUT  │ ──────► │ HACKER  │ ──────► │ EXTRACTOR │ ──────► │   TEAM    │
  └─────────┘         └─────────┘         └───────────┘         └───────────┘
 Tag Terminal         Crack Terminal       Secure Loot           Escape Exit
```

---

## 🔬 The Research Challenge: Causal Credit Dilution

Standard multi-agent RL algorithms (e.g., standard MAPPO, QMIX, or COMA) rely on joint team rewards and assume simultaneous contribution. In a **Recursively Gated Dec-POMDP (RG-Dec-POMDP)**, this assumption fails due to **Causal Credit Dilution**:
* If the Extractor fails at step 50 due to a late positioning error, global negative reward (-10.0) propagates backward.
* Standard value mixers and joint critics erroneously penalize the Scout's optimal actions taken at step 5.
* Under partial observability, standard policy gradient methods collapse into early risk-aversion (standing still in spawn to avoid step and alarm penalties).

---

## ⚡ Key Environment Features & Refinements

### 1. Refined & Balanced Reward System
* **Step Time Bleed:** `-0.01` per step to discourage loafing.
* **Dense Hack Progress:** Hacker receives `+0.5` on turn 1, `+0.5` on turn 2, and `+1.0` on turn 3 (`+2.0` total sum), drastically reducing gradient variance during multi-turn hacks.
* **Deduplicated Task Milestones:** Muscle guard neutralizations and wall breaches are rewarded (`+2.0`) **once per unique guard/wall position** per episode to prevent reward farming.
* **Incremental Alarm Step Penalty:** Applies `-0.01 * delta_alarm` to provide continuous negative feedback when agents step into camera line-of-sight or get spotted by guards.
* **Shared Terminal Outcome:** `+10.0` for successful team extraction with loot; `-10.0` for alarm collapse (100.0) or getting caught.

### 2. Dynamic Side-Tasks Extension (`--side-tasks`)
When launched with the `--side-tasks` CLI trigger, agents gain role-specific secondary capabilities during waiting phases:
* 🕵️ **Scout (Decoy Noise Ping):** Emits a sound distraction in open space, drawing nearby guards within 6 tiles to search the Scout's location.
* 💻 **Hacker (Door Lock Override):** Force-unlocks adjacent doors permanently (`DOOR` $\rightarrow$ `EMPTY`) to clear fast transit routes for the team.
* Muscle (Shortcut Wall Breach):** Destroys internal walls (`WALL` $\rightarrow$ `EMPTY`) to create custom escape corridors.
* 🎒 **Extractor (Beacon Pre-Calibration):** Calibrates the extraction beacon early, dropping the final extraction countdown from 10 steps down to **3 steps**.

### 3. Integrated Curiosity & Execution Infrastructure
* **Random Network Distillation (RND):** Includes RND curiosity-driven exploration (`src/exploration.py`) by default (`USE_RND=1`) with fixed target and online predictor neural networks.
* **Timestamp & Wall-Clock Logging:** All training logs display UTC start/end timestamps and elapsed execution duration per model and stage.
* **Run Output Isolation:** Side-task runs use distinct experiment suffixes (`_st`) and result directories (`st001`, `st002`) to ensure 0 overwriting of baseline runs.

---

## 🧪 10-Model Benchmark Suite & Research Taxonomy

HEIST includes a 10-model algorithm suite covering the major MARL paradigms:

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

### 2x2 Factorial Design Matrix
To rigorously isolate **communication message passing** from **causal advantage routing**:

```
                       Standard Credit Assignment      Causal Advantage Routing (CIR)
                     ┌──────────────────────────────┬──────────────────────────────────┐
  Communicating      │  comm                        │  comm_cir, comm_cir_car          │
                     ├──────────────────────────────┼──────────────────────────────────┤
  Non-Communicating  │  ippo, mappo, qmix, coma     │  mappo_cir, coma_cir             │
                     └──────────────────────────────┴──────────────────────────────────┘
```

---

## 📊 Benchmark Results (Stage 0, 1M Steps & Fast Validation)

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

---

## 🛠️ Setup and Installation

Requirements: Python 3.12+ and `uv` package manager.

```bash
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked
```

Run Pytest unit test suite (45 tests):

```bash
uv run pytest -v
uv run ruff check
```

---

## 🚀 Campaign Execution CLI (`train.zsh`)

Use `train.zsh` to provision, sync, and launch training campaigns:

```bash
# Standard Stage 0 campaign across all models (RND enabled by default)
./train.zsh --stages 0 --steps 300000

# Launch with Dynamic Side-Tasks enabled
./train.zsh --side-tasks --stages 0,1

# Fast validation test across specific models (10k steps)
./train.zsh --fast --models ippo,coma,mappo_car

# Run without RND exploration (for baseline ablation)
./train.zsh --no-rnd --models ippo,mappo

# Background daemon campaign across curriculum stages 0 and 1
./train.zsh --stages 0,1 --parallel 4 --daemon
```

### CLI Flag Reference

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--stages STAGES` | Comma-separated curriculum stage indices (e.g. `0,1,2`) | `0` |
| `--models MODELS` | Comma-separated algorithm names to train | All 10 models |
| `--side-tasks` | Enables dynamic side-tasks (decoy ping, door override, wall breach, beacon calibration) | Disabled |
| `--no-rnd` | Disables Random Network Distillation curiosity exploration | RND Enabled |
| `--fast`, `--quick` | Fast local validation test (10k steps) | Disabled |
| `--parallel`, `-j N` | Max concurrent model training jobs | `2` |
| `--steps STEPS` | Training timesteps per model | `299,008` |
| `--daemon` | Backgrounds campaign execution with `nohup`; logs to `log/launch.out` | Foreground |

---

## 📁 Codebase Layout

```
train.zsh                     Multi-model campaign launcher & orchestrator
tools/status.py               Live training log and checkpoint monitor
tools/assess_time.py          Hardware benchmark & throughput calculator
src/env.py                    PettingZoo parallel environment engine & reward logic
src/curriculum.py             6-stage curriculum generator (11x11 to 50x50)
src/vision.py                 Numba JIT raycasting & fog-of-war vision engine
src/model.py                  Neural network architectures (TarMAC, MAPPO, COMA)
src/ppo_utils.py              Counterfactual advantage math & checkpoint serialization
src/vec_env.py                Multiprocessing vectorized environment wrapper
src/exploration.py            Random Network Distillation (RND) curiosity module
src/eval_stage.py             Post-experiment evaluation & JSON summary merger
src/scripted.py               Near-optimal BFS controller baseline
src/train_ippo.py             Independent PPO trainer
src/train_mappo.py            MAPPO trainer (with CAR and CIR support)
src/train_coma.py             COMA trainer (with counterfactual & CIR support)
src/train_comm.py             TarMAC trainer (with CIR and CAR support)
src/train_qmix.py             QMIX trainer with value decomposition
tests/                        Pytest unit test suite (45 tests)
results/                      Run artifacts, JSON summaries, and benchmark logs
```
