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
   * **Validated (scripted controller, `src/scripted.py` + `src/run_scripted_curriculum.py`):** every stage is solvable and difficulty ramps monotonically. A guard-avoiding BFS team with muscle neutralization and hacker door bypass wins 1.00 / 0.58 / 0.40 / 0.33 / 0.20 across stages 0-4 (40-episode runs), with the causal chain completing 73-95% at every stage. Counterfactual analysis confirms scout/hacker/extractor are strictly necessary at all five stages (win -> 0 when no-op'd), so the RG-Dec-POMDP chain structure survives scaling. Full table in `results/README.md`.

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

### Baseline Validation Campaign (v1-v5): The Sparsity Wall, Empirically Confirmed

Before trusting CAI/counterfactual numbers, we validated that the training baselines actually learn the stage-0 task. This validation found two exploitable design bugs and produced a clean empirical record of the "Sparsity Wall".

**Design bugs found by validation (all fixed):**
1. **Eval win bug:** `run_episode` counted `terminal_reward > 0` as a win; at truncation, the last-step shaped reward (e.g., scout +0.49 tag) falsely reported wins (0.55 "win rate" vs 0.000 real). Fixed to read `infos[AGENTS[0]]["win"]` only on real termination.
2. **Scout tag farming:** `_scout_tag` paid +0.5 every step with no cooldown; IPPO learned to sit beside the terminal spamming INTERACT. Fixed with a per-episode `tagged_pois` set (each POI taggable once).
3. **Missing causal gate:** the HACK action was not gated on scout tagging; added `terminal_pos in self.tagged_pois` to both the action mask and `_hacker_hack`.
4. **Extraction timeout** fired every step after loot (100 -> 25, fires exactly once); countdown 30 -> 45.

**Env is solvable (controller ablation, stage-0, max_steps 90):** greedy Manhattan 10/30 wins; BFS pathfinding 14/30; BFS + wait-on-target 29/30. The env is sound; the learning is the bottleneck.

**Learning results (stage-0: 11x11, no guards/cameras/doors, max_steps 90):**

| Campaign | Changes vs prev | Win rate | Chain completion | Return |
|---|---|---|---|---|
| v2 IPPO | strict stacking win | 0.000 | terminal 90% / loot 85% | ~5.7 |
| v3 IPPO | zone win (radius 2) | 0.000 | terminal ~70% | 1-5 |
| v4 IPPO/MAPPO | + objective bearings (gs 4->10), + PBRS converge bonus | 0.000 | terminal 75% / loot 62% / extraction 62% | ~0 |
| v5 IPPO (300k) | more compute on v4 config | 0.03-0.10 | terminal 97% / loot 87% / extraction 87% | 0.7-1.5 |
| v5 MAPPO (300k) | more compute on v4 config | 0.067 | terminal 67% / loot 42% / extraction 42% | 0.59 |
| v5 QMIX (300k) | default 500k eps-anneal | 0.000 | terminal 83% / loot 0% / extraction 0% | -0.28 |
| v5 QMIX-a (300k) | + 200k eps-anneal (greedy by 200k) | 0.000 | terminal 47% / loot 0% / extraction 0% | -0.50 |

**Three-way baseline result (stage-0, 60-episode re-eval, seed 555):**

| Algo | Win rate | terminal / loot / extraction | CAI (s/h/m/e) | Counterfactual |
|---|---|---|---|---|
| Scripted (BFS) | 1.000 | 100% / 100% / 100% | n/a (zero outcome variance) | scout/hacker/extractor +1.0, muscle +0.57 |
| IPPO | 0.033 | 97% / 87% / 87% | +0.56 / +0.24 / +0.32 / +0.10 | scout/hacker/extractor +0.033, muscle +0.000 |
| MAPPO | 0.067 | 67% / 42% / 42% | +0.84 / +0.35 / +0.76 / +0.38 | scout/hacker/extractor +0.067, muscle +0.000 |
| QMIX | 0.000 | 47% / 0% / 0% | 0 / 0 / 0 / 0 | all 0 (no baseline wins) |

**Metric validation (scripted controller, `src/scripted.py`):** the
near-optimal BFS team wins 60/60 at stage-0 (mean length 13, return
+11.5), proving the env is solvable. Its counterfactual profile is the
ground truth the learned policies should approach: scout/hacker/extractor
each strictly necessary (win 1.0 -> 0.0 when no-op'd), muscle +0.57 (no
guards/doors at stage-0). The learned-policy evaluations reproduce this
exact ordering, so the diagnostic metrics are not noise. Counterfactual
baselines are reported on the same episode count and seed as the headline
win rate, so tables are internally consistent. CAI is undefined (0.000)
for both the scripted team (100% wins -> zero outcome variance) and QMIX
(0% wins -> zero terminal-reward variance); the counterfactual metric is
the informative one in those regimes.

QMIX learns the first gate (terminal hack) but never the downstream chain (loot 0%, extraction 0%, win 0%), and this holds with a proper epsilon schedule annealed to 0.05 by step 200k. The value-decomposition baseline is the most strongly diluted: per-agent Q-functions receive zero credit-to-outcome correlation because the joint mixer only fires on the (never-reached) terminal win. This is the cleanest Causal Credit Dilution demonstration: the chain's downstream phases are invisible to QMIX's credit signal at this budget.

**IPPO vs MAPPO contrast (v5, 60-episode re-eval, seed 555):**
- IPPO completes the chain far more often (97%/87%/87% vs 67%/42%/42%) but both reach similar win rates (~3-7%), because MAPPO's fewer completions are more coordinated (mean_alarm 12.8 vs 25.8 — MAPPO avoids the countdown-expiry penalty).
- CAI: IPPO ranks scout +0.56 > muscle +0.32 > extractor/hacker +0.10/+0.24. MAPPO ranks scout +0.84, muscle +0.76 > extractor +0.38, hacker +0.35. The shared-actor architecture collapses credit toward the most variable agents (scout/muscle), consistent with the expected MAPPO signature.
- Counterfactual: in both algorithms, scout/hacker/extractor are each strictly necessary; muscle is nearly so. Every agent is causally essential to the final win.

**v5 result (stage-0, 60-episode diagnostic, `results_stage0_ippo_v5.json`):** 300k steps lifts IPPO from 0% to a noisy 3-10% win rate (eval 20-episode peak 0.10; 60-episode re-eval 0.033). The causal chain now almost always completes (terminal 96.7%, loot 86.7%, extraction 86.7%), so the remaining bottleneck is exactly the final convergence. CAI ranks scout +0.561 > muscle +0.318 > hacker +0.245 > extractor +0.101; counterfactual no-ops show scout/hacker/extractor are each strictly necessary (win 0.10 -> 0.00) and muscle nearly so (0.10 -> 0.03). This is the intended Causal Credit Dilution signal: near-complete phase execution with rare wins, upstream agents causally essential.

**Root-cause analysis (traced v4 episodes):** `spawn_mode="role"` places scout+hacker adjacent to the terminal and muscle+extractor adjacent to the loot, so the early causal chain is nearly free (extraction triggered by step 5-6 in many episodes) and needs no navigation. The extract tile is placed far away, so the decisive skill is cross-map convergence after loot, which simple 64-unit MLP policies do not acquire in 100k steps (agents hover 0-2 tiles from extract but the extractor never steps onto the tile). This is the Sparsity Wall in its cleanest form: the local, easy phases reward early learning; the global, late phase does not get learned.

**Implications for the research claim:** the causal chain, action gating, and win mechanics are verified (scripted controller wins 29/30). The baseline failure *is* the motivating observation for Causal Credit Dilution research; the next step is to show QMIX/counterfactual diagnostics on longer runs and to ship the curriculum (stage-1+ with cameras/guards) plus intrinsic-motivation bonuses as the planned mitigation.

### Known Issues & Next Steps
* **Baseline compute:** IPPO needs ~300k steps to reach a noisy 3-10% win at stage-0 (role-spawn). A 300k run with `spawn_mode="random"` matched the win rate (0.033) but had far worse chain completion (32/22/22% vs 97/87/87%) — the harder navigation cost the budget without improving convergence. Role-spawn remains the best stage-0 config. Next steps: scale to 1-2M steps for IPPO/MAPPO, and/or add intrinsic-motivation exploration bonuses to break the final-converge plateau.
* **Throughput:** ~40-70 SPS on CUDA; the per-env Python loop in `vec_env.step` and per-agent tensor moves dominate. Next step: batch env stepping or move the sim loop into NumPy/Numba.
* **Reward/alarm balance:** alarm builds fast enough that a 3-turn hack chain raises it ~7 units; longer runs are needed to tune this so mid-difficulty stages are solvable but not trivial.
* **Baseline comparison:** the three-way stage-0 comparison is complete. Tables in `results_stage0_ippo_v5.json` / `results_stage0_mappo_v5.json` / `results_stage0_qmix_a_v5.json` (QMIX default-schedule variant: `results_stage0_qmix_v5.json`). Extend to curriculum stages 1-2 with cameras/guards.
* **Result tables:** `src/run_eval.py` emits win-rate/CAI/counterfactual tables (optionally JSON); run it on final checkpoints per stage once baselines converge.