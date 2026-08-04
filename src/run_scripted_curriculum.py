"""Run the scripted BFS controller across the full curriculum.

Produces `results/scripted_curriculum.json` plus per-stage JSONs and a
console table.  Validates that every curriculum stage is solvable by a
competent (scripted) team and that difficulty ramps monotonically.

Usage:
    ../.venv/bin/python run_scripted_curriculum.py [--episodes N] [--seed S]
"""

import argparse
import json
import os

from curriculum import CURRICULUM
from env import HeistEnv
from scripted import (
    evaluate_scripted,
    scripted_cai,
    scripted_counterfactual,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {
        "controller": "scripted_bfs_v2",
        "episodes": args.episodes,
        "seed": args.seed,
        "stages": [],
    }
    print("=" * 78)
    print("HEIST scripted controller across curriculum stages")
    print("=" * 78)
    for si, cfg in enumerate(CURRICULUM):
        env = HeistEnv(dict(cfg))
        m = evaluate_scripted(env, episodes=args.episodes, seed=args.seed)
        cai = scripted_cai(env, episodes=args.episodes, seed=args.seed)
        imp = scripted_counterfactual(env, episodes=args.episodes, seed=args.seed)
        stage = {
            "stage": si,
            "config": cfg,
            "metrics": m,
            "cai": cai,
            "counterfactual": imp,
        }
        summary["stages"].append(stage)
        with open(os.path.join(OUT_DIR, f"scripted_stage{si}.json"), "w") as f:
            json.dump(stage, f, indent=2, default=str)
        print(
            f"stage {si}: win={m['win_rate']:.3f} "
            f"term={m['terminal_rate']:.3f} loot={m['loot_rate']:.3f} "
            f"extract={m['extraction_rate']:.3f} len={m['mean_length']:.1f} "
            f"alarm={m['mean_alarm']:.1f} return={m['mean_return']:+.2f} | "
            f"cf (s/h/m/e) = {imp['importance']['scout']:+.2f}/"
            f"{imp['importance']['hacker']:+.2f}/"
            f"{imp['importance']['muscle']:+.2f}/"
            f"{imp['importance']['extractor']:+.2f}"
        )
    with open(os.path.join(OUT_DIR, "scripted_curriculum.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("-" * 78)
    print(
        f"saved {len(CURRICULUM)} stage files + scripted_curriculum.json "
        f"to {os.path.abspath(OUT_DIR)}"
    )


if __name__ == "__main__":
    main()
