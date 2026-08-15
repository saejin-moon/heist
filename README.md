# HEIST: Hierarchical Environment for Interdependent Sequential Tasks

HEIST is a PettingZoo multi-agent reinforcement learning (MARL) environment and research benchmark designed to evaluate **cooperative credit assignment under sequential causal dependencies and partial observability**.

Four specialized agents—**Scout**, **Hacker**, **Muscle**, and **Extractor**—coordinate to pull off a heist against a dynamic, rule-based security system (guards, cameras, and an alarm meter). While agents act simultaneously each step, their actions follow a strict, recursively gated causal dependency chain:

```
  ┌─────────┐         ┌─────────┐         ┌───────────┐         ┌───────────┐
  │  SCOUT  │ ──────> │ HACKER  │ ──────> │ EXTRACTOR │ ──────> │   TEAM    │
  └─────────┘         └─────────┘         └───────────┘         └───────────┘
 Tag Terminal         Crack Terminal       Secure Loot           Escape Exit
```

---

## The Research Challenge: Causal Credit Dilution

Standard multi-agent RL algorithms (e.g., standard MAPPO, QMIX, or COMA) rely on joint team rewards and assume simultaneous contribution. In a **Recursively Gated Dec-POMDP (RG-Dec-POMDP)**, this assumption fails due to **Causal Credit Dilution**:
* If the Extractor fails at step 50 due to a late positioning error, global negative reward (-10.0) propagates backward.
* Standard value mixers and joint critics erroneously penalize the Scout's optimal actions taken at step 5.
* Under partial observability, standard policy gradient methods collapse into early risk-aversion (standing still in spawn to avoid step and alarm penalties).

---

## Key Environment Mechanics & Refinements

### 1. Refined & Balanced Reward System
* **Step Time Bleed:** `-0.01` per step to discourage loafing.
* **Dense Hack Progress:** Hacker receives `+0.5` on turn 1, `+0.5` on turn 2, and `+1.0` on turn 3 (`+2.0` total sum), drastically reducing gradient variance during multi-turn hacks.
* **Deduplicated Task Milestones:** Muscle guard neutralizations and wall breaches are rewarded (`+2.0`) **once per unique guard/wall position** per episode to prevent reward farming.
* **Incremental Alarm Step Penalty:** Applies `-0.01 * delta_alarm` to provide continuous negative feedback when agents step into camera line-of-sight or get spotted by guards.
* **Shared Terminal Outcome:** `+10.0` for successful team extraction with loot; `-10.0` for alarm collapse (100.0) or getting caught.

### 2. Dynamic Side-Tasks Extension (`--side-tasks`)
When launched with the `--side-tasks` CLI trigger, agents gain role-specific secondary capabilities during waiting phases:
* **Scout (Decoy Noise Ping):** Emits a sound distraction in open space to draw nearby guards within 6 tiles to search the Scout's location.
* **Hacker (Door Lock Override):** Force-unlocks adjacent doors permanently (`DOOR` $\rightarrow$ `EMPTY`) to clear fast transit routes for the team.
* **Muscle (Shortcut Wall Breach):** Destroys internal walls (`WALL` $\rightarrow$ `EMPTY`) to create custom escape corridors.
* **Extractor (Beacon Pre-Calibration):** Calibrates the extraction beacon early, which drops the final extraction countdown from 10 steps down to **3 steps**.

---

## Benchmark Suite & Research Taxonomy (9 Baselines + 1 Novel Algorithm)

HEIST evaluates a 9-baseline benchmark suite alongside our flagship novel algorithm **MARC**:

| Model ID | Paradigm / Family | Core Mechanism |
| :--- | :--- | :--- |
| **`ippo`** | Independent RL | Fully decentralized actor-critic per agent (baseline) |
| **`mappo`** | Centralized Critic | Shared actor with centralized state-based $V(s)$ critic |
| **`coma`** | Counterfactual Advantage | Centralized critic $Q_i(s, \mathbf{a}_{-i}, a_i)$ with counterfactual baseline |
| **`comm`** | Differentiable Comm | TarMAC inter-agent attention-based message passing (self-attention excluded) |
| **`mappo_car`** | Affordance Shaping | Intrinsic affordance reward shaping ($\sum \Delta \text{Mask}$) for unlocking teammate actions |
| **`mappo_cir`** | Advantage Routing | Causal Influence Routing (Paper: $\nabla_a Q$ Jacobian $\vert$ Code: Approximated via $V(s)$ positional feature ablation) |
| **`loo`** | Leave-One-Out (C3-Style) | Marginal counterfactual baseline isolating $i$-th agent's contribution |
| **`ate`** | Average Treatment Effect | Contrastive advantage against explicit WAIT null action ($a_{\text{WAIT}} = 4$) |
| **`macca`** | Dynamic Bayesian Graph | Dynamic Bayesian Network (DBN) factorizing global state transitions |
| **`marc`** | **Novel Flagship Algorithm** | **Marginal Action Retroactive Credit** with binary success masking |
| **`marc_no_shielding`** | MARC Ablation | MARC without binary success masking |
| **`marc_no_macro`** | MARC Ablation | MARC without Macro Weighting ($\Omega_t = 1.0$) |
| **`marc_no_affordance`** | MARC Ablation | MARC without local affordance delta boost |
| **`charm`** | Hierarchical RL | Continuous Hierarchical Agent with Top-Down Manager |
| **`roma`** | Hierarchical RL | Role-Oriented Multi-Agent reinforcement learning |
| **`mahiro`** | Hierarchical RL | Multi-Agent Hierarchical reinforcement learning |
| **`lrs`** | Hierarchical RL | Latent Role Space baseline |
| **`coop`** | **Novel Flagship** | **Confidence-Oriented Option Pool** with structural affordances |
| **`coop_fixed`** | CO-OP Ablation | CO-OP without dynamic spawning (fixed pool) |
| **`coop_no_car`** | CO-OP Ablation | CO-OP without Causal Affordance Credit |
| **`coop_top_down`** | CO-OP Ablation | CO-OP using traditional Top-Down Manager instead of voting |

---

## The MARC Architecture

The **MARC (Marginal Action Retroactive Credit)** model resolves the Sparsity Wall via three mechanics:
1. **Local Affordance Deltas:** Local action advantages are boosted when an agent's direct interaction triggers an environment state-change (e.g., neutralizing a guard).
2. **Macro Weighting:** Global reward allocation decays exponentially relative to the environmental alarm level, naturally penalizing noisy/inefficient runs even if they eventually succeed.
3. **Binary Success Masking:** On team failure, negative credit targets direct failure triggers while shielding upstream enabling actions ($\Omega_t = 1.0$ for enablers).
4. **Retroactive Advantage Propagation ($t = T \to 0$):** Propagates advantages backward through time:
   $$\hat{A}_{i, t}^{\text{MARC}} = \mu_{i, t} \cdot \Omega_t + \gamma_{\text{causal}} \hat{A}_{i, t+1}^{\text{MARC}}$$

### MARC Component Ablations
- **`marc_no_shielding`:** Disables enabler shielding on team failure (tests necessity of shielding enablers from penalty leakage).
- **`marc_no_macro`:** Disables macro weighting ($\Omega_t = 1.0$) (tests necessity of global alarm and win/loss scaling).
- **`marc_no_affordance`:** Disables local affordance boost ($\beta_{\text{affordance}} = 0.0$) (tests necessity of action-mask expansion feedback).

---

## Setup and Installation

Requirements: Python 3.12+ and `uv` package manager.

```bash
git clone https://github.com/saejin-moon/heist.git
cd heist
uv sync --locked
```

Run Pytest unit test suite (59 tests):

```bash
uv run pytest -v
uv run ruff check
```

---

## Campaign Execution CLI (`scripts/train.zsh`)

Use `scripts/train.zsh` to provision, sync, and launch training campaigns:

```bash
# Standard Stage 0 campaign across all 10 models (RND enabled by default)
./scripts/train.zsh --stages 0 --steps 300000

# Fast turbo validation run (<15s)
./scripts/train.zsh --turbo --stages 0 -j 1

# Train specific models (e.g. MARC vs MAPPO)
./scripts/train.zsh --models marc,mappo_cir,mappo --stages 0

# Launch with Dynamic Side-Tasks enabled
./scripts/train.zsh --side-tasks --stages 0,1
```

---

## Codebase Layout

```
scripts/train.zsh             Multi-model campaign launcher & orchestrator
tools/status.py               Live training log and checkpoint monitor
tools/assess_time.py          Hardware benchmark & throughput calculator
tools/thermal_guard.py        Hardware thermal protection kill switch
src/env.py                    PettingZoo parallel environment engine & reward logic
src/curriculum.py             Spatial step density scaling & curriculum specification
src/model.py                  Neural network architectures (TarMAC, MAPPO, COMA)
src/ppo_utils.py              Counterfactual, LOO, ATE advantage math & utils
src/exploration.py            Random Network Distillation (RND) curiosity module
src/eval_stage.py             Post-experiment evaluation & JSON summary merger
src/train_ippo.py             Independent PPO trainer
src/train_mappo.py            MAPPO trainer (with CAR and CIR support)
src/train_coma.py             COMA counterfactual trainer
src/train_comm.py             TarMAC communication trainer
src/train_loo.py              Leave-One-Out (C3-style) counterfactual trainer
src/train_ate.py              Average Treatment Effect contrastive trainer
src/train_macca.py            Dynamic Bayesian Network causal credit trainer
src/train_marc.py             MARC flagship novel algorithm trainer
paper/                        Quarkdown research paper documentation suite
tests/                        Pytest unit test suite (59 PyTest cases)
results/                      Run artifacts, JSON summaries, and benchmark logs
```
