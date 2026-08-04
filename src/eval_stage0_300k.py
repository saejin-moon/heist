#!/usr/bin/env python3
"""
Post-experiment evaluation: loads trained checkpoints from the 300k stage-0
campaign and produces the G4 comparison table (IPPO floor vs comm).

Run after experiment_stage0_300k.sh completes:
    uv run python eval_stage0_300k.py

Saves JSON to results/stage0_300k_comparison.json.
"""
import json
import os

import torch

from constants import AGENTS
from curriculum import CURRICULUM
from env import HeistEnv
from evaluate import (
    counterfactual_importance,
    credit_attribution_index,
    evaluate_comm_policies,
    evaluate_policies,
    message_outcome_correlation,
)
from model import CommAgent, HeistAgent, MappoAgent, QNetwork
from run_eval import load_policies

STAGE0 = CURRICULUM[0]
EVAL_EPISODES = 60
EVAL_SEED = 555
CHECKPOINT_ROOT = "checkpoints"

ALGOS = {
    "ippo":  {"exp_name": "ippo_s0",  "class": "ippo"},
    "mappo": {"exp_name": "mappo_s0", "class": "mappo"},
    "qmix":  {"exp_name": "qmix_s0",  "class": "qmix"},
    "comm":  {"exp_name": "comm_s0",  "class": "comm"},
}
SEEDS = [0, 1, 2]


def eval_one(algo, exp_name, seed, env, state_dim, device):
    """Evaluate a single checkpoint. Returns a result dict or None if missing."""
    run_dir = os.path.join(CHECKPOINT_ROOT, f"{exp_name}_s{seed}")
    if not os.path.isdir(run_dir) or not os.listdir(run_dir):
        print(f"  SKIP {run_dir} (no checkpoint)")
        return None

    print(f"  Evaluating {algo} {exp_name}_s{seed} ...")
    policies = load_policies(algo, run_dir, state_dim, device)

    if algo == "comm":
        metrics = evaluate_comm_policies(
            policies, env, episodes=EVAL_EPISODES, seed=EVAL_SEED, device=device
        )
        diag = message_outcome_correlation(
            policies, env, episodes=EVAL_EPISODES, seed=EVAL_SEED, device=device
        )
        return {
            "metrics": metrics,
            "diag": {
                "max_terminal_message_corr": diag["max_terminal_message_corr"],
                "mean_terminal_message_corr": diag["mean_terminal_message_corr"],
                "mean_attention": diag["mean_attention"].tolist(),
            },
        }
    else:
        metrics = evaluate_policies(
            policies, env, episodes=EVAL_EPISODES, seed=EVAL_SEED, device=device
        )
        cai = credit_attribution_index(
            policies, env, episodes=EVAL_EPISODES, seed=EVAL_SEED, device=device
        )
        imp = counterfactual_importance(
            policies, env, episodes=EVAL_EPISODES, seed=EVAL_SEED, device=device
        )
        return {"metrics": metrics, "cai": cai, "counterfactual": imp}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = HeistEnv(STAGE0)
    state_dim = env.state().shape[0]

    all_results = {}
    for algo, info in ALGOS.items():
        print(f"\n{'='*64}\n  {algo.upper()}\n{'='*64}")
        seeds = []
        for seed in SEEDS:
            r = eval_one(algo, info["exp_name"], seed, env, state_dim, device)
            if r is not None:
                seeds.append({"seed": seed, **r})
        all_results[algo] = seeds

    # ── summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("G4 COMPARISON: stage-0 300k-step campaign (60-episode eval, seed 555)")
    print(f"{'='*80}")

    header = f"{'Algo':<8} {'Seed':>4}  {'Win':>5} {'Term':>5} {'Loot':>5} {'Extr':>5} {'Return':>7}"
    print(header)
    print("-" * len(header))

    floor_wins, comm_wins = [], []
    for algo, entries in all_results.items():
        for entry in entries:
            m = entry["metrics"]
            seed_label = f"s{entry['seed']}"
            wr = m.get("win_rate", 0)
            tr = m.get("terminal_rate", 0)
            lr = m.get("loot_rate", 0)
            er = m.get("extraction_rate", 0)
            ret = m.get("mean_return", 0)
            print(f"{algo:<8} {seed_label:>4}  {wr:>5.3f} {tr:>5.3f} {lr:>5.3f} {er:>5.3f} {ret:>7.3f}")
            if algo == "ippo":
                floor_wins.append(wr)
            if algo == "comm":
                comm_wins.append(wr)
        if entries:
            avg_m = {}
            for k in entries[0]["metrics"]:
                vals = [e["metrics"][k] for e in entries if k in e["metrics"]]
                if vals and isinstance(vals[0], (int, float)):
                    avg_m[k] = sum(vals) / len(vals)
            wr = avg_m.get("win_rate", 0)
            tr = avg_m.get("terminal_rate", 0)
            lr = avg_m.get("loot_rate", 0)
            er = avg_m.get("extraction_rate", 0)
            ret = avg_m.get("mean_return", 0)
            print(f"{'>>> '+algo:<8} {'avg':>4}  {wr:>5.3f} {tr:>5.3f} {lr:>5.3f} {er:>5.3f} {ret:>7.3f}")

    # ── CAI summary ────────────────────────────────────────────────────
    for algo in ["ippo", "mappo", "qmix"]:
        entries = all_results.get(algo, [])
        if entries and "cai" in entries[0]:
            avg_cai = {}
            for a in AGENTS:
                vals = [e["cai"][a] for e in entries if a in e["cai"]]
                if vals:
                    avg_cai[a] = sum(vals) / len(vals)
            print(f"\n  CAI ({algo} avg): " + " ".join(
                f"{a}:{avg_cai.get(a, 0):+.3f}" for a in AGENTS
            ))

    # ── comm diagnostic summary ────────────────────────────────────────
    for entry in all_results.get("comm", []):
        if "diag" in entry:
            print(f"\n  Comm diagnostics (seed {entry['seed']}):")
            print(f"    max terminal corr : {entry['diag']['max_terminal_message_corr']:.4f}")
            print(f"    mean terminal corr: {entry['diag']['mean_terminal_message_corr']:.4f}")
            att = entry["diag"]["mean_attention"]
            print("    mean attention (rows=receiver, cols=sender):")
            for i, a in enumerate(AGENTS):
                print(f"      {a:>9}: " + " ".join(f"{att[i][j]:.3f}" for j in range(len(AGENTS))))

    # ── G4 verdict ─────────────────────────────────────────────────────
    if floor_wins and comm_wins:
        avg_floor = sum(floor_wins) / len(floor_wins)
        avg_comm = sum(comm_wins) / len(comm_wins)
        verdict = "PASS" if avg_comm >= avg_floor else "FAIL"
        print(f"\n  G4 verdict: comm avg win={avg_comm:.3f} vs IPPO floor={avg_floor:.3f} -> {verdict}")

    # ── save JSON ──────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/stage0_300k_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved full results to {out_path}")


if __name__ == "__main__":
    main()
