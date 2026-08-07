# AGENTS.md — System Guide for AI Assistants & Developers

Welcome to the **HEIST** repository! This document serves as the master architectural reference for AI coding agents (Antigravity, Gemini, Claude, Cursor) and human developers working on this codebase.

---

## 1. Project Overview & Core Mission

**HEIST** (*Hierarchical Environment for Interdependent Sequential Tasks*) is a partially observable multi-agent reinforcement learning (MARL) benchmark. It is designed to evaluate how MARL algorithms perform under **Causal Credit Dilution**—a structural failure mode in Dec-POMDPs where upstream agents enabling team success receive zero or negative immediate feedback, while downstream agents absorb shared terminal rewards.

The environment requires a team of **4 specialized agents** to collaborate sequentially:
1. **Scout ($S$):** Explores fog-of-war, discovers security terminals/doors, and tags points of interest.
2. **Hacker ($H$):** Navigates to security terminals, executes multi-turn hacks, and bypasses locked doors.
3. **Muscle ($M$):** Neutralizes patrolling security guards and breaches obstacles.
4. **Extractor ($E$):** Secures the vault loot, triggers the escape countdown, and leads all 4 agents to the exit.

---

## 2. Codebase Sitemap & Directory Structure

```
heist/
├── src/                        # Core Python package
│   ├── constants.py            # Single source of truth for constants, grid semantics, and rewards
│   ├── curriculum.py           # 5-stage geometric spatial curriculum specification
│   ├── env.py                  # PettingZoo-style Dec-POMDP environment (HeistEnv)
│   ├── evaluate.py             # Diagnostic evaluation engine (Causal Funnel, CAI, Loss Modes)
│   ├── exploration.py          # Intrinsic exploration modules (RND & Count-based)
│   ├── model.py                # PyTorch networks (HeistAgent, CommAgent, QNetwork, QMixMixing)
│   ├── ppo_utils.py            # Vectorized PPO rollouts and GAE calculation
│   ├── train_ippo.py           # Independent PPO trainer
│   ├── train_mappo.py          # Centralized Critic PPO trainer (supports --car and --cir)
│   ├── train_comm.py           # TarMAC Differentiable Communication trainer
│   ├── train_coma.py           # Counterfactual Advantage baseline trainer
│   └── train_qmix.py           # QMIX Monotonic Value Factorization trainer
├── scripts/                    # Production shell scripts
│   ├── train.zsh               # Campaign orchestrator & multi-process scheduler
│   ├── target.zsh              # 5-stage campaign entrypoint script
│   ├── assess-time.zsh         # Throughput benchmark runner script
│   ├── side-tasks.zsh          # Side-task ablation launcher script
│   ├── rsync-results.sh        # Remote sync helper (bae@forest.local)
│   └── evaluate.sh             # Full campaign evaluation runner script
├── tools/                      # Analytics & hardware protection CLI tools
│   ├── thermal_guard.py        # Hardware safety kill switch (CPU max 85°C, GPU max 83°C)
│   ├── status.py               # Terminal UI dashboard (Rich) for live campaign tracking
│   ├── assess_time.py          # Empirical step/sec throughput benchmark
│   └── evaluate_campaign.py    # Multi-stage evaluation exporter across checkpoints
├── paper/                      # Research paper typesetting suite (Quarkdown .qd format)
│   ├── main.qd                 # Main paper entrypoint
│   ├── 01_abstract_and_introduction.qd
│   ├── 02_environment_and_constants.qd
│   ├── 03_model_architectures_and_math.qd
│   ├── 04_curriculum_and_spatial_step_scaling.qd
│   └── 05_experimental_metrics_and_evaluation.qd
└── tests/                      # Unit test suite (55 PyTest cases)
```

---

## 3. Key Environment Contracts & Data Shapes

### Observation Space
* **Local Observation ($o_i \in \mathbb{R}^{53}$):** Each agent receives a $7 \times 7$ local view grid ($\text{OBSERVATION\_SIZE} = (7, 7)$, $\text{AGENT\_VISION\_RADIUS} = 3$) flattened to $49$ values, concatenated with a $4$-element one-hot role vector $e_{\text{role}_i} \in \{0, 1\}^4$.
* **Centralized State ($s \in \mathbb{R}^D$):** Obtained via `env.state()`. Consumed by MAPPO, QMIX, and COMA critics.

### Action Space ($|\mathcal{A}_i| = 6$)
$ \mathcal{A}_i = \{0: \text{UP}, 1: \text{DOWN}, 2: \text{LEFT}, 3: \text{RIGHT}, 4: \text{WAIT}, 5: \text{INTERACT}\} $

---

## 4. The 10-Model MARL Taxonomy

The codebase supports 10 distinct algorithm configurations across 4 fundamental paradigms:

| Model ID | Paradigm Class | Architecture Description |
| :--- | :--- | :--- |
| **`ippo`** | Independent RL | Decentralized PPO actors & local critics |
| **`mappo`** | Centralized Critic | Shared actor network + Centralized Critic $V_{\Phi}(s)$ |
| **`mappo_car`** | Reward Shaping | MAPPO + Causal Affordance Credit (CAR) |
| **`mappo_cir`** | Advantage Routing | MAPPO + Causal Advantage Routing (CIR) |
| **`comm`** | Differentiable Comm | TarMAC attention message passing ($\bar{m}_i \in \mathbb{R}^{32}$) |
| **`comm_cir`** | Comm + Routing | TarMAC Communication + CIR Advantage Routing |
| **`comm_cir_car`**| Unified Multi-Factor | TarMAC + CIR + CAR |
| **`qmix`** | Value Factorization | Monotonic mixing network $Q_{\text{tot}}(s, \mathbf{a}) = f_{\text{mixing}}(Q_1, \dots, Q_4; s)$ |
| **`coma`** | Counterfactual Baseline| Counterfactual Advantage $A_i = Q(s, \mathbf{a}) - \sum \pi_i Q(s, (a_i', \mathbf{a}_{-i}))$ |
| **`coma_cir`** | Counterfactual + CIR | COMA + CIR Advantage Routing |

---

## 5. Standard Developer Workflow & Commands

### Running Unit Tests
Always run PyTest after modifying environment or model logic:
```bash
uv run pytest
```

### Checking Linting & Formatting
Enforce PEP 8 / ISort standards via `ruff`:
```bash
uv run ruff check src/ tests/ tools/
uv run ruff format src/ tests/ tools/
```

### Compiling Research Paper (`paper/`)
Compile the Quarkdown documentation suite:
```bash
quarkdown c paper/main.qd --strict --out /tmp/quarkdown-verify
```

### Running a Campaign
Launch full 5-stage campaign (4-hour fast budget):
```bash
./scripts/train.zsh -j 5 --daemon --steps 75000 --stages 0,1,2,3,4
```

### Monitoring Active Campaigns
Launch live fullscreen dashboard:
```bash
uv run python tools/status.py --watch
```

---

## 6. Development Rules for AI Agents

1. **Single Source of Truth:** Never hardcode constants or dimensions in trainer or model files. Always import from `src/constants.py` and `src/curriculum.py`.
2. **Preserve Contracts:** When modifying `HeistEnv.step()` or `run_episode()`, ensure observation, reward, and info dictionary schemas are strictly preserved.
3. **Thermal Safety:** Never disable `tools/thermal_guard.py` checks in `train.zsh`.
4. **Verification:** Never declare success on an issue without running `uv run pytest` to verify zero regression.
