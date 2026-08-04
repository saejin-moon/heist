# HEIST: PLANS, TECHNICAL SPECIFICATIONS & RESEARCH FORMULATION

## OVERVIEW & CONCEPT
HEIST (Hierarchical Environment for Interdependent Sequential Tasks) is a cooperative multi-agent RL (MARL) environment where a team of four specialized agents pulls off a heist against a rule-based (and eventually learned) security system. 

Each agent has a distinct role, partial information about the world, and unique available actions. They must coordinate under sequential dependency constraints: the Scout must act before the Hacker can be effective, and the Hacker must succeed before extraction is possible. The core research focus is solving credit/blame assignment across a sequential, causal dependency chain under partial observability with late-arriving rewards.

---

## DESIGN DOCUMENT

### Environment Overview
* **Type:** Cooperative multi-agent, partially observable, discrete turn-based (prototype), with adversarial rule-based opponent.
* **Grid:** 2D tile-based map, procedurally generated per episode. Tiles include open floor, walls, locked doors, security camera coverage zones, guard patrol paths, terminals (for the hacker), loot locations, and extraction points.
* **Win Condition:** All agents reach the extraction point with the loot before the timer expires and without triggering a full alarm.
* **Lose Conditions:** Any agent is caught by a guard, alarm threshold exceeded, or timer runs out.
* **Alarm System:** Shared global alarm meter. Events raise it incrementally (camera spot: slight, guard spot: significant, terminal hack failure: moderate). At 100%, guards converge and the episode terminates.

---

### The Four Agents

1. **Scout**
   * **Role:** Moves ahead of the team, revealing tile information (guard positions, camera coverage, door lock types, terminal locations) masked to other agents.
   * **Observation Space:** Widest field of view (N tiles in all directions), including guard patrol states and camera rotation angles.
   * **Action Space:** Move (8 directions), wait, tag tile (broadcasts location info to team), use distraction item (draws guard attention temporarily).
   * **Unique Mechanic:** Other agents' observation spaces are partially gated by what the Scout has revealed.
   * **Risk Profile:** Highest exposure; highest risk of triggering alarms if reckless.

2. **Hacker**
   * **Role:** Interfaces with terminals to disable cameras, unlock doors, and open the vault. Cannot safely act on a terminal unless tagged by the Scout.
   * **Observation Space:** Narrow field of view, but reads terminal status info (encryption level, time-to-crack, camera oversight).
   * **Action Space:** Move, wait, hack terminal (requires adjacency, takes multiple turns, interruptible), bypass door (faster, higher alarm risk).
   * **Unique Mechanic:** Multi-turn hacking creates windows of vulnerability requiring team coverage. Interruption causes loss of progress and alarm increase.
   * **Risk Profile:** Stationary and vulnerable during hacks; dependent on Scout intel and Muscle coverage.

3. **Muscle (or "Ghost")**
   * **Role:** Handles physical obstacles—guards that cannot be avoided, doors that cannot be hacked, or distraction/misdirection needs.
   * **Observation Space:** Medium range, oriented specifically toward guard positions.
   * **Action Space:** Move, wait, neutralize guard (removes them for N turns, raises alarm), create noise distraction, carry loot.
   * **Unique Mechanic:** Neutralizing guards carries a delayed alarm cost when the guard's absence is noticed and triggers a search pattern.
   * **Risk Profile:** High alarm cost actions; creates time-sensitive downstream pressure for extraction.

4. **Extractor**
   * **Role:** Final link in the dependency chain. Picks up loot, coordinates final phase timing, and clears the path to the extraction point.
   * **Observation Space:** Narrowest field of view, global knowledge of timer and rough agent status signals (safe/at risk/compromised).
   * **Action Space:** Move, wait, carry loot, call extraction (signals convergence), use escape tool (one-time path clear).
   * **Unique Mechanic:** Calling extraction initiates a countdown; all agents must reach extraction within K turns.
   * **Risk Profile:** Lowest direct exposure; carries coordination burden of the final phase.

---

## RESEARCH FORMULATION & NOVELTY HOOK

### Mathematical Formalization: Recursively Gated Dec-POMDP (RG-Dec-POMDP)
The environment is formalized as a **Recursively Gated Dec-POMDP (RG-Dec-POMDP)**. Unlike standard Dec-POMDPs that permit independent parallel action selection, an RG-Dec-POMDP enforces a directed causal chain of action-space gating.

Let the effective action space of agent $i$ at time step $t$ be $A_i(s_t)$. The cardinality of $A_i(s_t)$ is conditionally constrained by the historical trajectory $\tau$ of its predecessor agent $i-1$:

$$A_i(s_t) = 
\begin{cases} 
\{ \text{wait} \} & \text{if } f(\tau_{1:t-1}, i-1) = 0 \\
A_i^{\text{full}} & \text{if } f(\tau_{1:t-1}, i-1) = 1 
\end{cases}$$

where $f$ is an indicator transition function determining whether the upstream agent has successfully resolved its phase of the dependency chain. Downstream agents receive zero policy gradient signal until upstream agents solve their tasks.

### Causal Credit Dilution
Standard value-decomposition networks (e.g., QMIX) assume the joint action-value function can be factored as a monotonic combination of individual utilities:

$$Q_{\text{tot}}(\mathbf{o}, \mathbf{a}) = g(Q_1(o_1, a_1), Q_2(o_2, a_2), \dots, Q_n(o_n, a_n))$$

In an RG-Dec-POMDP, this assumption breaks down due to **Causal Credit Dilution**:
* If the Extractor fails at step $T$ due to a late error, negative global reward propagates backward.
* The joint value mixer falsely penalizes the Scout's optimal actions taken at step $T-N$.
* **Diagnostic Experiment:** Track gradients $\nabla_{\theta_i} Q_i$ relative to final joint reward $R$ and measure the "Credit Attribution Index" across training epochs to evaluate credit assignment failure points.

---

## TECHNICAL STACK & ARCHITECTURE

* **Environment:** Python, built as a Gymnasium-compatible multi-agent environment (PettingZoo `ParallelEnv` API). Custom 2D grid engine built with pure NumPy tensor operations (headless training).
* **Rendering:** Pygame visualizer (`manual_control.py`) for debugging/demos.
* **Algorithms Benchmarked:** IPPO (independent baseline), MAPPO (cooperative baseline), QMIX (value decomposition baseline), and optional communication variants (CommNet / TarMAC style).
* **Map Generation:** Procedural room/corridor connections, guard patrol waypoint loops, camera line-of-sight cones, and randomized terminal/loot placements.
* **Observation Encoding:** Dict observation space per agent:
  1. `observation`: 5x5 Fog-Masked local box (masked by Fog of War except where Scout revealed tiles).
  2. `action_mask`: 6-element binary vector enforcing dynamic action constraints.
  3. `global_state`: 4-element vector (alarm level, timer remaining, agent status signals).
* **Reward Structure:**
  * Shared terminal reward: $+10.0$ (successful extraction with loot), $-10.0$ (caught/alarm/timeout).
  * Intermediate rewards: $+2.0$ for key task completions (Scout revealing info, Hacker completing terminal, Muscle neutralizing guard, Extractor securing loot).
  * Baseline time penalty: $-0.01$ per step time bleed.

---

## LOGS & IMPLEMENTATION PROGRESS

### Mission Accomplishments
1. **Core Environment Engine (PettingZoo Compliant):**
   * Built `ParallelEnv` MARL framework using NumPy row-major tensor operations.
   * Passed strict PettingZoo `parallel_api_test` for RLlib/CleanRL integration.
2. **Mathematical Execution (RG-Dec-POMDP):**
   * Implemented **Dynamic Action Masking** enforcing causal dependency chains (Scout $\rightarrow$ Hacker $\rightarrow$ Extractor).
   * Constructed multi-tensor `Dict` observation space (`observation`, `action_mask`, `global_state`).
   * Configured shaped reward parameters (time bleed $-0.01$, intermediate objectives $+2.0$, catastrophic failure $-10.0$, global win $+10.0$).
3. **Adversary & Map Mechanics:**
   * Procedural map generation bound to PettingZoo RNG seeds to prevent policy memorization.
   * Rule-Based Adversaries (Guards) running random-walk patrols with Manhattan-distance collision triggers (triggering $-10.0$ failure).
   * Security cameras with Bresenham line-of-sight exposure and an incremental shared alarm meter (camera spot, hack attempts, guard neutralization, terminal bypass all raise alarm; 100 = episode loss).
   * Multi-turn interruptible terminal hacking (interruption resets progress and raises alarm), scout tagging, muscle neutralization, extractor loot pickup, and a timed extraction countdown.
4. **Engineering Infrastructure:**
   * Refactored into modular layout (`env.py`, `vision.py`, `map_gen.py`, `constants.py`).
   * Implemented Pygame visualizer (`manual_control.py`).
   * Integrated Bresenham's Line Algorithm for Fog of War and Line-of-Sight tensor masking, JIT-compiled with **Numba** (`vision.py`) so training does not stall on Python raycasting loops.
5. **Training Stack:**
   * `vec_env.py`: numba-free vectorized wrapper for dense PPO-style rollouts.
   * `train_ippo.py`: independent PPO (with `--shared` parameter sharing).
   * `train_mappo.py`: shared actor with centralized state-based critic.
   * `train_qmix.py`: off-policy value decomposition with replay buffer and monotonic hypernetwork mixer.
   * All trainers: TensorBoard logging, checkpointing to `src/runs/<algo>_s<seed>/`, JSON `--env-config`, eval every N updates, `--no-cuda`/`--no-save-model` flags. IPPO/MAPPO/QMIX smoke tests verified end-to-end on CUDA.
6. **Evaluation & Diagnostics (`evaluate.py`):**
   * Win rate, mean return/length, final alarm, and per-phase task completion (terminal disabled, loot, extraction).
   * **Credit Attribution Index (CAI):** Pearson correlation of per-agent shaped credit vs episode outcome.
   * **Counterfactual importance:** win-rate drop when each agent is replaced by a no-op.
   * `summarize()` prints a full diagnostic report; verified against random policies.
7. **Curriculum Learning (`curriculum.py`):**
   * Five staged configs ramping from 11x11 / no guards / no cameras up to the 50x50 full benchmark, addressing the "Sparsity Wall".

### Document & Model Updates
* **Formal Title:** HEIST: Hierarchical Environment for Interdependent Sequential Tasks.
* **Observation Space Model:** Confirmed 3-part Dict space: `(1) 5x5 Fog-Masked Box, (2) 6-element Action Mask, (3) 4-element Global State Vector`.
* **Adversaries:** Explicit rule-based guard patrols with $-10.0$ catastrophic collision conditions.

### Future Optimization & Technical Insights
* **Performance Horizon (Raycasting Bottleneck):** Python loops in Bresenham raycasting (`vision.py`) throttle performance during high-FPS training rollouts.
  * *Resolved:* `vision.py` raycasting/camera-exposure kernels are JIT-compiled with **Numba**; the hot path no longer stalls in pure Python. Remaining throughput is ~40-70 SPS (CUDA) due to per-step Python env stepping, which is the next optimization target (e.g., stepping the whole env vector with NumPy ops instead of a Python `for` loop over envs).
* **The Sparsity Wall:** High risk of standard MARL flatlining on complex procedural maps despite shaped intermediate rewards.
  * *Resolved:* Curriculum learning implemented (`curriculum.py`) with 5 staged configs (11x11 no-security up to full 50x50). Intrinsic-motivation exploration bonuses remain an optional future enhancement.
* **QMIX Triton warmup:** the first GPU run spends minutes JIT-compiling new ops; use `--no-cuda` for quick tests or budget a long first-run timeout.

### Known Issues & Next Steps
* **Throughput:** ~40-70 SPS on CUDA; the per-env Python loop in `vec_env.step` and per-agent tensor moves dominate. Next step: batch env stepping or move the sim loop into NumPy/Numba.
* **Reward/alarm balance:** alarm builds fast enough that a 3-turn hack chain raises it ~7 units; longer runs are needed to tune this so mid-difficulty stages are solvable but not trivial.
* **Baseline comparison:** run IPPO / MAPPO / QMIX to convergence on each curriculum stage and emit the CAI + counterfactual tables for the research writeup.
* **Initial commit:** the repository is now version-controlled with an initial commit; long training runs and result tables are the remaining evidence-gathering step.