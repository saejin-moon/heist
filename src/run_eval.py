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

from constants import ACTION_SPACE_SIZE as ACTION_DIM
from constants import N_AGENTS
from env import AGENTS, HeistEnv
from evaluate import (
    counterfactual_importance,
    credit_attribution_index,
    evaluate_comm_policies,
    evaluate_policies,
    message_outcome_correlation,
)
from model import ComaAgent, CommAgent, HeistAgent, MappoAgent, QNetwork


def load_policies(algo, run_dir, state_dim, device):
    policies = {}
    if algo == "comm":
        # REV-7: shared CommAgent with TarMAC messages, saved as comm.pt.
        p = CommAgent(state_dim=state_dim, centralized=False)
        sd = torch.load(
            os.path.join(run_dir, "comm.pt"), map_location=device, weights_only=True
        )
        p.load_state_dict(sd)
        p.to(device).eval()
        return p
    if algo == "mappo":
        # Shared-actor MAPPO saves one policy.pt; all agents use the same weights.
        sd = torch.load(
            os.path.join(run_dir, "policy.pt"), map_location=device, weights_only=True
        )
        ckpt_state_dim = (
            sd["critic.0.weight"].shape[1] if "critic.0.weight" in sd else state_dim
        )
        p = MappoAgent(state_dim=ckpt_state_dim)
        p.load_state_dict(sd)
        p.to(device).eval()
        return {a: p for a in AGENTS}
    if algo == "coma":
        # Shared-actor COMA saves policy.pt; all agents use the same weights.
        sd = torch.load(
            os.path.join(run_dir, "policy.pt"), map_location=device, weights_only=True
        )
        ckpt_state_dim = (
            sd["critic.net.0.weight"].shape[1] - (N_AGENTS - 1) * ACTION_DIM
            if "critic.net.0.weight" in sd
            else state_dim
        )
        p = ComaAgent(state_dim=ckpt_state_dim)
        p.load_state_dict(sd)
        p.to(device).eval()
        return {a: p for a in AGENTS}
    if algo == "qmix":
        # QMIX saves per-agent DQN heads ({agent}_q.pt) plus mixing.pt.
        # Greedy action selection only needs the per-agent Q-networks.
        for a in AGENTS:
            p = QNetwork()
            sd = torch.load(
                os.path.join(run_dir, f"{a}_q.pt"),
                map_location=device,
                weights_only=True,
            )
            p.load_state_dict(sd)
            p.to(device).eval()
            policies[a] = p
        return policies
    for a in AGENTS:
        p = HeistAgent()
        sd = torch.load(
            os.path.join(run_dir, f"{a}.pt"), map_location=device, weights_only=True
        )
        p.load_state_dict(sd)
        p.to(device).eval()
        policies[a] = p
    return policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument(
        "--algo",
        default="ippo",
        choices=["ippo", "mappo", "qmix", "comm", "coma"],
    )

    ap.add_argument("--env-config", type=str, default="{}")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = HeistEnv(json.loads(args.env_config))
    state_dim = env.state().shape[0]
    policies = load_policies(args.algo, args.run_dir, state_dim, args.device)

    if args.algo == "comm":
        print("=" * 64)
        print(f"HEIST evaluation: algo={args.algo} run={args.run_dir}")
        print("=" * 64)
        metrics = evaluate_comm_policies(
            policies, env, episodes=args.episodes, seed=args.seed, device=args.device
        )
        for k, v in metrics.items():
            print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")
        print("-" * 64)
        diag = message_outcome_correlation(
            policies, env, episodes=args.episodes, seed=args.seed, device=args.device
        )
        print("Message-outcome correlation (REV-7 Phase C):")
        print(f"  max terminal corr : {diag['max_terminal_message_corr']:.4f}")
        print(f"  mean terminal corr: {diag['mean_terminal_message_corr']:.4f}")
        print("  mean attention matrix (receiver rows):")
        for i, a in enumerate(AGENTS):
            print(
                f"    {a:>9}: "
                + " ".join(
                    f"{diag['mean_attention'][i, j]:.3f}" for j in range(len(AGENTS))
                )
            )
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            payload = {
                "metrics": metrics,
                "diag": {
                    "max_terminal_message_corr": diag["max_terminal_message_corr"],
                    "mean_terminal_message_corr": diag["mean_terminal_message_corr"],
                    "mean_attention": diag["mean_attention"].tolist(),
                },
                "algo": args.algo,
                "run_dir": args.run_dir,
                "env_config": json.loads(args.env_config),
                "episodes": args.episodes,
                "seed": args.seed,
            }
            with open(args.out, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"\nsaved results to {args.out}")
        return

    print("=" * 64)
    print(f"HEIST evaluation: algo={args.algo} run={args.run_dir}")
    print("=" * 64)
    metrics = evaluate_policies(
        policies, env, episodes=args.episodes, seed=args.seed, device=args.device
    )
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")

    print("-" * 64)
    cai = credit_attribution_index(
        policies, env, episodes=args.episodes, seed=args.seed, device=args.device
    )
    print("Credit Attribution Index (corr credit vs outcome):")
    for a in AGENTS:
        print(f"  {a:>9}: {cai[a]:+.3f}")

    print("-" * 64)
    imp = counterfactual_importance(
        policies, env, episodes=args.episodes, seed=args.seed, device=args.device
    )
    print("Counterfactual importance (baseline - no-op win rate):")
    print(f"  baseline win rate: {imp['baseline_win_rate']:.3f}")
    for a in AGENTS:
        print(f"  {a:>9}: {imp['importance'][a]:+.3f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {
            "metrics": metrics,
            "cai": cai,
            "counterfactual": imp,
            "algo": args.algo,
            "run_dir": args.run_dir,
            "env_config": json.loads(args.env_config),
            "episodes": args.episodes,
            "seed": args.seed,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nsaved results to {args.out}")


if __name__ == "__main__":
    main()
