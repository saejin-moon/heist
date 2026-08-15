# Architecture & Environment Optimization Log

## 1. Environment Balancing
- **Extractor Burden Removal:** Remove `EXTRACTOR_BURDEN_TURNS`. The movement penalty guarantees extraction timeouts and mathematically skews the loss landscape toward failure.
- **Breach Penalty Reduction:** Lower `ALARM_BREACH` from 30.0 to **10.0**. Empirical RL research on risk-reward trade-offs dictates that penalties exceeding 15% of a terminal failure threshold deter exploration. At 10.0, the Muscle can execute three strategic breaches before risking catastrophic convergence, making it a viable heuristic shortcut.
- **Muscle Breach Heuristic:** Implement a simple, dense scalar reward (e.g., +2.0) for successfully breaching an obstacle. Do NOT dynamically calculate Breadth-First Search (BFS) shortest-path deltas mid-episode; executing BFS pathfinding immediately before and after every wall destruction injects catastrophic computational overhead into vectorized multi-agent rollouts.
- **Scout Tagging Radius:** Relax the strict adjacency requirement for `INTERACT`. The Scout can successfully tag a landmark from anywhere within its 3-cell vision radius, provided it executes the `INTERACT` command once per distinct landmark.

## 2. Curriculum Extension
- **Stage 5 (Side-Tasks):** Add a final curriculum stage. Stage 5 will run on the Stage 4 spatial geometry (50x50, 4 guards, 3 cameras) but inject orthogonal side-tasks (e.g., secondary loot caches, optional terminal hacks). This tests the architecture's resistance to credit dilution when presented with non-critical reward signals.

## 3. Evaluation Mechanics
- **Dual-Track Evaluation:** Execute both stochastic and deterministic evaluation loops. Stochastic evaluation measures reliance on entropy crutches. Deterministic evaluation explicitly exposes structural routing failures and chattering loops.

## 4. Algorithmic Tracking
- **E-COOP Transition & Hazard Mitigation:**
  - **FIM-Scaled Asexual Mutation (Git Re-Basin Rollback):** Git Re-Basin biparental crossover is strictly incompatible with the GRU recurrent layers used in HEIST. Linear weight interpolation forces variance collapse, but Variance Restoration destroys the recurrent hidden state $h_t$ by saturating the sigmoid gating mechanisms. We abandon biparental crossover. E-COOP will rely entirely on Cloned Survival and FIM-scaled asexual mutation. The child expert inherits an exact clone of the best parent's Actor and Critic networks. We compute the FIM via `torch.func.vmap`, chunked into micro-batches, strictly summing unnormalized squared gradients across chunks, followed by a single global $1/N$ division. To avoid mathematical corruption, these gradients MUST be computed exclusively from the pure log-likelihood $\nabla_\theta \log \pi_\theta(a|s)$, completely bypassing the PPO clipped surrogate loss.
  - **Generational Cadence & Pool Size:** Trigger the first evolutionary crossover at epoch 500, then execute subsequent crossovers every 200 epochs. The expert pool size ($K$) will be exposed as a dynamic argument (`--ecoop-pool-size`) in `train.zsh` for empirical scaling tests.
  - **Dynamic Routing Immunity (Grace Periods & Hysteresis):** Abandon UCB, MC Dropout, and Frequency Penalties. Maintain standard $\arg\max_k V_k(s)$ deterministic bidding. To prevent microscopic chattering caused by floating-point noise or zero-crossings during sequence transitions, introduce **Hybrid Routing Hysteresis**: an active expert yields control only if $V_{rival} > V_{current} + \max(\epsilon_{abs}, \epsilon_{rel} \cdot |V_{current}|)$. To prevent mutant starvation, enforce a "Routing Grace Period." Unconditionally route to newly spawned child experts for a fixed number of epochs, bypassing the $\arg\max$ vote.
  - **Fisher Inverse Clipping:** Scale mutational noise variance strictly by $1 / \sqrt{F_i}$, not $1 / F_i$. A $1/F_i$ variance scaling mathematically forces the KL divergence to infinity as $F_i \to 0$. You must hard-clip this inverse square root scalar before applying noise. FIM provides only a local quadratic approximation; injecting massive unclipped noise into highly uncertain parameters violently ejects the weights into non-linear cliffs, shattering the policy.
  - **Immediate Joint Optimization (Critic Burn-In Rollback):** We completely eliminate the decoupled Critic Burn-In phase. Strict on-policy burn-in on a frozen mutant Actor truncates sequential state distributions, causing the monolithic inherited Critic to suffer catastrophic forgetting via parameter interference. Upon mutation, immediately execute standard joint PPO updates (unfreezing both Actor and Critic simultaneously) during the Grace Period. **Adam Cold-Start:** You must explicitly discard the parent's Adam optimizer state ($m_t, v_t$). FIM translates the weights to a new coordinate in the loss geometry. Inheriting stale momentum injects uncalibrated velocity that physically overshoots the PPO trust region during the first Grace Period update.

## 5. Procedural & Diagnostic Tools
- **ASCII Rendering (`src/ascii.py`):** Implement ANSI color codes for terminal rendering. Assign `█` for walls, `▒` for doors, `G` for guards, `C` for cameras. Dynamically map agents to their specific role characters (`S, H, M, E`) instead of generic ally markers.
- **Sequential Map Generation (`src/map_gen.py`):**
  - Guarantee L-shaped corridors are carved at exactly 2 tiles wide to prevent movement blockades.
  - Hardcode agent spawns to cluster exclusively inside the Entrance Room (Room 0). This strict causal constraint must be enforced across all curriculum levels, starting immediately at Stage 0, as the 11x11 grid is small enough for the agents to mathematically solve the baseline pathfinding without proximity warm-starts.
  - Enforce causal object placement: sequence the Terminal to spawn in early rooms, and the Loot/Extraction points to spawn in terminal rooms, guaranteeing Hacker access before Door bottlenecks.

## 6. Experimental Campaign
- **Ablation Definitions (E-COOP):**
  - `ecoop_uniform_noise`: Replaces inverse-FIM scaling with uniform Gaussian noise to test if Fisher-bounding prevents genetic decay.
  - `ecoop_reactive`: Reverts the rigid generational ES cadence to value-threshold ($\tau_{spawn}$) reactive spawning.
  - `ecoop_no_grace`: Disables the routing Grace Period to explicitly verify if newly spawned mutants immediately starve without forced burn-in.
  - `ecoop_no_hysteresis`: Disables Hybrid Routing Hysteresis to test for policy chattering and deadlocks at zero-crossings.
- **Final Evaluation Suite:** Run full 5-stage benchmark campaign comparing `mappo`, `coma`, `marc`, `charm`, `mahiro`, `coop`, and `ecoop` (plus ablations), alongside a fully scripted expert heuristic to establish the absolute CAI mathematical bounds.