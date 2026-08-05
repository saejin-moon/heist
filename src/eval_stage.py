#!/usr/bin/env python3
"""
Post-experiment evaluation CLI for HEIST curriculum stages.

Loads trained checkpoints for any specified curriculum stage and step budget,
evaluating all executed model variants (ippo, mappo, mappo_car, comm, comm_cir,
comm_cir_car, qmix). Results are saved to a 6-character max subfolder in results/.

Usage:
    uv run python src/eval_stage.py --stage 0
    uv run python src/eval_stage.py --stage 0 --steps 10240
    uv run python src/eval_stage.py --run-id run002
"""

import argparse
import json
import os

import torch

from curriculum import CURRICULUM
from env import HeistEnv
from evaluate import (
    counterfactual_importance,
    credit_attribution_index,
    evaluate_comm_policies,
    evaluate_policies,
    message_outcome_correlation,
)
from run_eval import load_policies

EVAL_EPISODES = 60
EVAL_SEED = 555
CHECKPOINT_ROOT = "checkpoints"

DEFAULT_ALGOS = {
    "ippo": {"exp_name": "ippo", "class": "ippo"},
    "mappo": {"exp_name": "mappo", "class": "mappo"},
    "mappo_car": {"exp_name": "mappo_car", "class": "mappo"},
    "mappo_cir": {"exp_name": "mappo_cir", "class": "mappo"},
    "comm": {"exp_name": "comm", "class": "comm"},
    "comm_cir": {"exp_name": "comm_cir", "class": "comm"},
    "comm_cir_car": {"exp_name": "comm_cir_car", "class": "comm"},
    "qmix": {"exp_name": "qmix", "class": "qmix"},
    "coma": {"exp_name": "coma", "class": "coma"},
    "coma_cir": {"exp_name": "coma_cir", "class": "coma"},
}


SEEDS = [0, 1, 2]


def get_next_run_id(base_dir="results", prefix="run"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d.startswith(prefix)
        and d[len(prefix) :].isdigit()
    ]
    nums = [int(d[len(prefix) :]) for d in existing if len(d) <= 6]
    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}{next_num:03d}"[:6]


def eval_one(
    algo, exp_base, stage, seed, env, state_dim, device, checkpoint_root, episodes
):
    """Evaluate a single checkpoint. Returns a result dict or None if missing."""
    # Check possible checkpoint naming formats: exp_s{stage}_s{seed} or exp_s{stage} or exp_s{seed}
    possible_names = [
        f"{exp_base}_s{stage}_s{seed}",
        f"{exp_base}_s{stage}",
        f"{exp_base}_s{seed}",
        exp_base,
    ]
    run_dir = None
    for name in possible_names:
        candidate = os.path.join(checkpoint_root, name)
        if os.path.isdir(candidate) and os.listdir(candidate):
            run_dir = candidate
            break

    if run_dir is None:
        return None

    policy_type = DEFAULT_ALGOS[algo]["class"]
    policies = load_policies(policy_type, run_dir, state_dim, device)

    if policy_type == "comm":
        metrics = evaluate_comm_policies(
            policies, env, episodes=episodes, seed=EVAL_SEED, device=device
        )
        diag = message_outcome_correlation(
            policies, env, episodes=episodes, seed=EVAL_SEED, device=device
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
            policies, env, episodes=episodes, seed=EVAL_SEED, device=device
        )
        cai = credit_attribution_index(
            policies, env, episodes=episodes, seed=EVAL_SEED, device=device
        )
        imp = counterfactual_importance(
            policies, env, episodes=episodes, seed=EVAL_SEED, device=device
        )
        return {"metrics": metrics, "cai": cai, "counterfactual": imp}


def evaluate_single_job(task):
    algo, exp_base, stage, seed, checkpoint_root, episodes = task

    # Prevent PyTorch from spawning a thread pool per worker (avoid CPU thrashing)
    torch.set_num_threads(1)

    stage_cfg = CURRICULUM[stage] if stage < len(CURRICULUM) else CURRICULUM[0]
    env = HeistEnv(stage_cfg)
    state_dim = env.state().shape[0]
    # Force CPU for evaluation to bypass CUDA initialization and resource contention issues in parallel workers
    device = torch.device("cpu")

    res = eval_one(
        algo, exp_base, stage, seed, env, state_dim, device, checkpoint_root, episodes
    )
    env.close()
    return algo, seed, res


def main():
    parser = argparse.ArgumentParser(description="HEIST Stage Evaluation Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="0",
        help="Stage index or comma-separated stages (e.g. 0 or 0,1,2)",
    )
    parser.add_argument("--steps", type=int, default=299008, help="Step budget label")
    parser.add_argument("--episodes", type=int, default=60, help="Evaluation episodes")
    parser.add_argument(
        "--run-id", type=str, default="", help="6-char max results subdirectory name"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=CHECKPOINT_ROOT,
        help="Checkpoints root dir",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="",
        help="Only evaluate this specific algorithm",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge individual algorithm evaluation JSONs into summary.json",
    )
    args = parser.parse_args()

    stages = [int(s.strip()) for s in args.stage.split(",") if s.strip().isdigit()]
    run_id = args.run_id[:6] if args.run_id else get_next_run_id()
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)

    if args.merge:
        # Merge individual files
        all_stage_results = {}
        for stg in stages:
            stage_key = f"stage_{stg}"
            all_stage_results[stage_key] = {algo: [] for algo in DEFAULT_ALGOS}

            for algo in DEFAULT_ALGOS:
                fpath = os.path.join(results_dir, f"individual_{algo}_stage{stg}.json")
                if os.path.isfile(fpath):
                    try:
                        with open(fpath) as f:
                            all_stage_results[stage_key][algo] = json.load(f)
                        os.remove(fpath)  # clean up
                    except Exception as e:
                        print(
                            f"Warning: failed to load individual results for {algo} stage {stg}: {e}"
                        )

            # Print stage summary table
            print(f"\n{'=' * 80}")
            print(f"STAGE {stg} SUMMARY (steps={args.steps})")
            print(f"{'=' * 80}")
            header = f"{'Algo':<14} {'Seed':>4}  {'Win':>5} {'Term':>5} {'Loot':>5} {'Extr':>5} {'Return':>7}"
            print(header)
            print("-" * len(header))

            for algo, entries in all_stage_results[stage_key].items():
                for entry in entries:
                    m = entry["metrics"]
                    seed_label = f"s{entry['seed']}"
                    wr, tr, lr, er, ret = (
                        m.get("win_rate", 0),
                        m.get("terminal_rate", 0),
                        m.get("loot_rate", 0),
                        m.get("extraction_rate", 0),
                        m.get("mean_return", 0),
                    )
                    print(
                        f"{algo:<14} {seed_label:>4}  {wr:>5.3f} {tr:>5.3f} {lr:>5.3f} {er:>5.3f} {ret:>7.3f}"
                    )

        # Save merged summary
        out_path = os.path.join(results_dir, "summary.json")
        with open(out_path, "w") as f:
            json.dump(all_stage_results, f, indent=2, default=str)

        if 0 in stages:
            with open("results/stage0_300k_comparison.json", "w") as f:
                json.dump(
                    all_stage_results.get("stage_0", {}), f, indent=2, default=str
                )

        print(f"\n  Merged all evaluations into {out_path} (subdir: {run_id})")
        return

    # Normal or algorithm-specific evaluation
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    for stg in stages:
        # Assemble tasks for this stage
        tasks = []
        target_algos = [args.algo] if args.algo else list(DEFAULT_ALGOS.keys())

        for algo in target_algos:
            info = DEFAULT_ALGOS[algo]
            for seed in SEEDS:
                tasks.append(
                    (
                        algo,
                        info["exp_name"],
                        stg,
                        seed,
                        args.checkpoint_root,
                        args.episodes,
                    )
                )

        if not tasks:
            continue

        num_workers = min(len(tasks), max(1, os.cpu_count() or 4))

        with ctx.Pool(num_workers) as pool:
            raw_results = pool.map(evaluate_single_job, tasks)

        # Structure results back
        all_results = {algo: [] for algo in target_algos}
        for algo, seed, res in raw_results:
            if res is not None:
                all_results[algo].append({"seed": seed, **res})

        # Sort entries by seed
        for algo in all_results:
            all_results[algo].sort(key=lambda x: x["seed"])

        if args.algo:
            # Save individual results for this algorithm
            out_file = os.path.join(
                results_dir, f"individual_{args.algo}_stage{stg}.json"
            )
            with open(out_file, "w") as f:
                json.dump(all_results[args.algo], f, indent=2, default=str)
        else:
            # Save standard summary
            all_stage_results = {f"stage_{stg}": all_results}
            out_path = os.path.join(results_dir, "summary.json")
            with open(out_path, "w") as f:
                json.dump(all_stage_results, f, indent=2, default=str)

            if 0 in stages:
                with open("results/stage0_300k_comparison.json", "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

            # Print summary table
            print(f"\n{'=' * 80}")
            print(
                f"EVALUATING STAGE {stg} (steps={args.steps}, episodes={args.episodes})"
            )
            print(f"{'=' * 80}")
            header = f"{'Algo':<14} {'Seed':>4}  {'Win':>5} {'Term':>5} {'Loot':>5} {'Extr':>5} {'Return':>7}"
            print(f"\n{header}")
            print("-" * len(header))
            for algo, entries in all_results.items():
                for entry in entries:
                    m = entry["metrics"]
                    seed_label = f"s{entry['seed']}"
                    wr, tr, lr, er, ret = (
                        m.get("win_rate", 0),
                        m.get("terminal_rate", 0),
                        m.get("loot_rate", 0),
                        m.get("extraction_rate", 0),
                        m.get("mean_return", 0),
                    )
                    print(
                        f"{algo:<14} {seed_label:>4}  {wr:>5.3f} {tr:>5.3f} {lr:>5.3f} {er:>5.3f} {ret:>7.3f}"
                    )
            print(f"\n  Saved full evaluation results to {out_path} (subdir: {run_id})")


if __name__ == "__main__":
    main()
