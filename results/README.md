# HEIST Results & Benchmark Evaluation

This directory contains benchmark results, evaluation summaries, and empirical diagnostic data across all 7 algorithms (`ippo`, `mappo`, `mappo_car`, `comm`, `comm_cir`, `comm_cir_car`, `qmix`) and the near-optimal scripted controller baseline ([`src/scripted.py`](file:///home/fuddle/git/heist/src/scripted.py)).

Primary summary artifacts:
* [`stage0_comparison.json`](file:///home/fuddle/git/heist/results/stage0_comparison.json): Consolidated comparison table across 10k, 300k, and 1,000,000 step budgets.
* [`run018/summary.json`](file:///home/fuddle/git/heist/results/run018/summary.json): Full 1M-step evaluation output, CAI correlation matrices, and counterfactual importance rankings.

---

## Stage-0 Headline Benchmark Table (1,000,000 Steps)

60-episode greedy rollouts across 3 random seeds on Stage 0 (11x11 grid, 1-2 rooms, max steps 60):

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

## Credit Attribution Index (CAI Correlations at 1M Steps)

Pearson correlation of per-agent shaped credit with terminal episode outcome:

| Algorithm | Scout | Hacker | Muscle | Extractor |
| :--- | :---: | :---: | :---: | :---: |
| **`mappo_car`** | **+0.622** | **+0.285** | **+0.190** | **+0.273** |
| **`ippo`** | +0.609 | +0.405 | -0.009 | +0.177 |
| **`mappo`** | +0.551 | +0.312 | +0.045 | +0.201 |

---

## Counterfactual Importance (Baseline Win Rate - No-Op Win Rate)

| Algorithm | Baseline Win | Scout No-Op | Hacker No-Op | Muscle No-Op | Extractor No-Op |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`mappo_car`** | **0.083** | **+0.083** | **+0.083** | **+0.017** | **+0.083** |
| **`ippo`** | 0.050 | +0.050 | +0.050 | -0.017 | +0.050 |
| **`mappo`** | 0.050 | +0.050 | +0.050 | +0.000 | +0.050 |

---

## Empirical Findings & Analytical Takeaways

1. **CAR Affordance Dominance:** `mappo_car` achieves both the **highest win rate (8.3%)** and **highest mean return (+1.534)**. Intrinsic affordance rewards grant credit when an action turns a teammate's dynamic action mask from 0 to 1, effectively pulling centralized critics out of early risk-aversion traps.
2. **Causal Chain Essentiality:** Replacing Scout, Hacker, or Extractor with a no-op zeroes out the win rate (dropping from 8.3% $\rightarrow$ 0.0%), confirming strict causal necessity across the 3 sequential chain links.
3. **Multi-Budget Phase Transition (10k vs 300k vs 1M steps):**
   - **Low Compute (10k-300k steps):** QMIX learns early local sub-goals faster due to off-policy value decomposition sample efficiency.
   - **High Compute (1.0M steps):** On-policy PPO with CAR (`mappo_car`) **dominates QMIX**, completing the full causal chain at 80%–93% rates.

---

## Curriculum Solvability (Scripted BFS Controller)

The near-optimal scripted controller ([`src/scripted.py`](file:///home/fuddle/git/heist/src/scripted.py)) validates that every curriculum stage is solvable:

| Stage | Map & Security Config | Win Rate | Terminal Rate | Loot Rate | Extraction Rate | Mean Alarm |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | 11x11, 0 guards, 0 doors | **1.000** | **1.000** | **1.000** | **1.000** | 6.0 |
| **1** | 15x15, 2 guards, 1 door | **0.725** | **0.925** | **0.900** | **0.900** | 30.9 |
| **2** | 21x21, 3 guards, 2 cameras, 2 doors | **0.575** | **0.850** | **0.850** | **0.850** | 49.8 |
| **3** | 35x35, 5 guards, 3 cameras, 3 doors | **0.550** | **0.900** | **0.900** | **0.900** | 45.7 |
| **4** | 50x50, 6 guards, 3 cameras, 4 doors | **0.450** | **0.850** | **0.825** | **0.825** | 50.6 |
