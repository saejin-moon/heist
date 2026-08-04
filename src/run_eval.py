"""
Evaluation CLI for trained HEIST policies.

Loads the final checkpoints written by a trainer and produces the full
diagnostic report (win rate, phase completion, Credit Attribution Index,
counterfactual importance) as console output and/or a JSON file.

Example:
    ../.venv/bin/python run_eval.py --run-dir checkpoints/ippo_s0 \
        --algo ippo --episodes 60 --seed 777 \
        --env-config '{"map_size": [11, 11], "guard_count": 0, "camera_count": 0, "max_steps": 60}' \
        --out results/ippo_stage0.json
"""

import argparse
import json
import os

import torch

from env import HeistEnv, AGENTS
from model import HeistAgent, MappoAgent
from evaluate import evaluate_policies, credit_attribution_index, counterfactual_importance


def load_policies(algo, run_dir, state_dim, device):
    policies = {}
    for a in AGENTS:
        path = os.path.join(run_dir, f"{a}.pt")
        if algo == "mappo":
            p = MappoAgent(state_dim=state_dim)
        else:
            p = HeistAgent()
        sd = torch.load(path, map_location=device, weights_only=True)
        p.load_state_dict(sd)
        p.to(device).eval()
        policies[a] = p
    return policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--algo", default="ippo", choices=["ippo", "mappo", "qmix"])
    ap.add_argument("--env-config", type=str, default="{}")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = HeistEnv(json.loads(args.env_config))
    state_dim = env.state().shape[0]
    policies = load_policies(args.algo, args.run_dir, state_dim, args.device)

    print("=" * 64)
    print(f"HEIST evaluation: algo={args.algo} run={args.run_dir}")
    print("=" * 64)
    metrics = evaluate_policies(policies, env, episodes=args.episodes,
                                seed=args.seed, device=args.device)
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")

    print("-" * 64)
    cai = credit_attribution_index(policies, env, episodes=args.episodes,
                                   seed=args.seed, device=args.device)
    print("Credit Attribution Index (corr credit vs outcome):")
    for a in AGENTS:
        print(f"  {a:>9}: {cai[a]:+.3f}")

    print("-" * 64)
    imp = counterfactual_importance(policies, env, episodes=max(args.episodes // 2, 10),
                                    seed=args.seed + 10_000, device=args.device)
    print("Counterfactual importance (baseline - no-op win rate):")
    print(f"  baseline win rate: {imp['baseline_win_rate']:.3f}")
    for a in AGENTS:
        print(f"  {a:>9}: {imp['importance'][a]:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"metrics": metrics, "cai": cai, "counterfactual": imp,
                   "algo": args.algo, "run_dir": args.run_dir,
                   "env_config": json.loads(args.env_config),
                   "episodes": args.episodes, "seed": args.seed}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved results to {args.out}")


if __name__ == "__main__":
    main()
