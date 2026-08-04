# Stage-0 Baseline Results (three-way comparison)

All campaigns: 300k env steps, stage-0 curriculum config
(11x11, 1-2 rooms, no guards/cameras/doors, max_steps 90).
60-episode re-eval on the final checkpoint, seed 555, greedy rollouts.
Full JSON artifacts in this directory (`results_stage0_*.json`).

## Headline table

| Algo | Win rate | Terminal | Loot | Extraction | Mean return | Mean alarm |
|---|---|---|---|---|---|---|
| IPPO (independent PPO) | 0.033 | 96.7% | 86.7% | 86.7% | +0.71 | 25.8 |
| MAPPO (shared actor + central critic) | 0.067 | 66.7% | 41.7% | 41.7% | +0.59 | 12.8 |
| QMIX (value decomposition) | 0.000 | 46.7% | 0.0% | 0.0% | -0.50 | 5.4 |

QMIX was also run with the trainer default epsilon schedule (annealed to
0.05 only at 500k, so ~0.46 at step 300k): terminal 83.3%, but still
0.0% loot / 0.0% extraction / 0.000 win rate.

## Credit Attribution Index (Pearson correlation of per-agent shaped
## credit with terminal outcome)

| Algo | Scout | Hacker | Muscle | Extractor |
|---|---|---|---|---|
| IPPO | +0.561 | +0.245 | +0.318 | +0.101 |
| MAPPO | +0.838 | +0.349 | +0.761 | +0.377 |
| QMIX | +0.000 | +0.000 | +0.000 | +0.000 |

## Counterfactual importance (baseline win rate minus no-op win rate)

| Algo | Baseline | Scout | Hacker | Muscle | Extractor |
|---|---|---|---|---|---|
| IPPO | 0.100 | +0.100 | +0.100 | +0.033 | +0.100 |
| MAPPO | 0.200 | +0.200 | +0.200 | +0.100 | +0.200 |
| QMIX | 0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Reading

1. **All four roles are causally essential.** Replacing any of
   scout/hacker/extractor with a no-op zeroes out the win rate in both
   policy-gradient baselines; muscle is nearly so. This validates the
   RG-Dec-POMDP chain design.
2. **IPPO learns the whole chain but rarely converts.** Chain completion
   is 97/87/87% yet wins are ~3%; the residual failure is the final
   cross-map convergence under the extraction countdown.
3. **QMIX exhibits the strongest Causal Credit Dilution.** It learns the
   first gate (terminal hack) but never reaches loot/extraction: its
   joint mixer only fires on the never-achieved terminal win, so per-agent
   credit correlates with nothing (CAI 0.0) and it never learns the
   downstream chain. This is the predicted value-decomposition failure.
4. **MAPPO's shared critic focuses credit on the variable agents**
   (scout/muscle) rather than the chain's bottleneck, matching the
   central-critic signature.
