#!/usr/bin/env python3
"""
assess_time.py -- measure throughput and estimate campaign wall-clock time.

Run on the TARGET machine (the box that will run the campaign) BEFORE
launching training, to get a real estimate of how long the runs will take:

    uv run python tools/assess_time.py                 # full assessment
    uv run python tools/assess_time.py --trainer-steps 10240   # shorter
    uv run python tools/assess_time.py --algos ippo comm       # subset

What it measures:
  1. System: GPU name/VRAM/driver, CPU cores, torch/CUDA versions.
  2. Raw env stepping cost per curriculum stage (0..4), wait-only and
     random-action modes.
  3. One full vec-env rollout step (tensor creation + policy forward +
     env step) per stage on the active device.
  4. Real end-to-end trainer throughput (steps/s) for each selected algo by
     running a SHORT real training run at the campaign batch shape
     (8 envs x 256 steps, stage-0) with checkpointing disabled.

Then it extrapolates three campaign scenarios:
  A. The 300k stage-0 campaign (experiment_stage0_300k.sh): 4 algos x 3
     seeds x 300k steps, sequential on one GPU.
  B. The full 1M-step, 5-stage grid: 4 algos x 3 seeds x 5 stages.
  C. The documented plan (FUTURE_PLANS.md): stage-0 full (4 algos x 3
     seeds x 300k) + IPPO 1M x 1 seed over stages 1-4.

Stage-0 sps are measured directly.  Stages 1-4 are estimated by scaling the
constant policy/tensor overhead against the measured per-stage env cost
(conservative: the constant overhead dominates at small maps, so the
estimate is biased toward the stage-0 number).

Exit code 0 on success; 2 if torch has no CUDA (warns but still measures).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from curriculum import CURRICULUM  # noqa: E402
from env import AGENTS  # noqa: E402

ALGOS = {
    "ippo": {
        "script": "src/train_ippo.py",
        "flags": ["--num-envs", "8", "--num-steps", "256", "--no-save-model"],
        "step_flag": "--total-timesteps",
        "vec": True,
    },
    "mappo": {
        "script": "src/train_mappo.py",
        "flags": ["--num-envs", "8", "--num-steps", "256", "--no-save-model"],
        "step_flag": "--total-timesteps",
        "vec": True,
    },
    "qmix": {
        "script": "src/train_qmix.py",
        # Match the campaign's lower update-to-data ratio.  QMIX remains
        # single-env here; vectorized collection is a separate experiment.
        "flags": ["--train-freq", "4"],
        "step_flag": "--total-steps",
        "vec": False,
    },
    "comm": {
        # no --no-save-model: comm only saves with an explicit --save-model
        "script": "src/train_comm.py",
        "flags": ["--num-envs", "8", "--num-steps", "256"],
        "step_flag": "--total-steps",
        "vec": True,
    },
}


def system_info() -> dict:
    import platform

    info = {
        "host": platform.node(),
        "cpu_cores": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["vram_gb"] = round(p.total_memory / 1e9, 1)
        info["cuda"] = torch.version.cuda
    return info


def env_step_bench(steps: int = 200) -> list[dict]:
    """Per-stage single-env step cost, wait-only and random-action.

    Episodes end at max_steps, so the env is re-reset whenever it becomes
    inactive (mirrors what VectorEnv does during real rollouts).  The
    measured per-step cost therefore includes the occasional reset.
    """
    from env import HeistEnv

    def _reset(env, seed):
        o, _ = env.reset(seed=seed)
        return o

    rows = []
    for s, cfg in enumerate(CURRICULUM):
        env = HeistEnv(cfg)
        seed = 123 + s
        obs = _reset(env, seed)

        # warmup (compiles numba kernels, spawns guards/cams)
        for _ in range(5):
            obs, *_ = env.step({a: 4 for a in AGENTS})
            if not env.agents:
                obs = _reset(env, seed)

        t0 = time.perf_counter()
        for _ in range(steps):
            obs, *_ = env.step({a: 4 for a in AGENTS})
            if not env.agents:
                obs = _reset(env, seed)
        wait_ms = (time.perf_counter() - t0) / steps * 1e3

        obs = _reset(env, seed)
        acts = {}
        for a in AGENTS:
            legal = np.flatnonzero(obs[a]["action_mask"])
            acts[a] = int(legal[len(legal) // 2])
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, *_ = env.step(acts)
            if not env.agents:
                obs = _reset(env, seed)
            for a in AGENTS:
                legal = np.flatnonzero(obs[a]["action_mask"])
                acts[a] = int(legal[len(legal) // 2])
        rand_ms = (time.perf_counter() - t0) / steps * 1e3

        rows.append(
            {
                "stage": s,
                "map": f"{cfg['map_size'][0]}x{cfg['map_size'][1]}",
                "guards": cfg["guard_count"],
                "wait_ms": round(wait_ms, 4),
                "random_ms": round(rand_ms, 4),
            }
        )
    return rows


def vec_rollout_bench(steps: int = 20, num_envs: int = 8) -> list[dict]:
    """Full rollout-step cost per stage: tensor copies + policy + env step."""
    from model import HeistAgent
    from vec_env import VectorEnv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for s, cfg in enumerate(CURRICULUM):
        vec = VectorEnv(num_envs, config=cfg, base_seed=200 + s)
        obs, _ = vec.reset(seed=200 + s)
        # Match the optimized shared-policy rollout path: VectorEnv provides
        # agent-major arrays, which are transferred once per observation
        # component and evaluated by one [agents * envs, ...] forward pass.
        policy = HeistAgent().to(device)
        for _ in range(3):  # warmup
            stacked = obs["_stacked"]
            obs_t = torch.as_tensor(stacked["observation"], device=device)
            role_t = torch.as_tensor(stacked["role_id"], device=device)
            mask_t = torch.as_tensor(stacked["action_mask"], device=device)
            with torch.no_grad():
                act, *_ = policy.get_action_and_value(
                    obs_t.flatten(0, 1),
                    role_t.flatten(0, 1),
                    mask_t.flatten(0, 1),
                )
            actions = {
                a: act.view(len(AGENTS), num_envs)[i].cpu().numpy()
                for i, a in enumerate(AGENTS)
            }
            obs, *_ = vec.step(actions)

        t0 = time.perf_counter()
        for _ in range(steps):
            stacked = obs["_stacked"]
            obs_t = torch.as_tensor(stacked["observation"], device=device)
            role_t = torch.as_tensor(stacked["role_id"], device=device)
            mask_t = torch.as_tensor(stacked["action_mask"], device=device)
            with torch.no_grad():
                act, *_ = policy.get_action_and_value(
                    obs_t.flatten(0, 1),
                    role_t.flatten(0, 1),
                    mask_t.flatten(0, 1),
                )
            actions = {
                a: act.view(len(AGENTS), num_envs)[i].cpu().numpy()
                for i, a in enumerate(AGENTS)
            }
            obs, *_ = vec.step(actions)
        per_step_s = (time.perf_counter() - t0) / steps
        vec.close()
        rows.append(
            {
                "stage": s,
                "vec_step_ms": round(per_step_s * 1e3, 2),
                "vec_sps": int(1 / per_step_s),
            }
        )
    return rows


def run_trainer(algo: str, steps: int) -> float:
    """Run a short real training run; return measured steps/s."""
    run_dir = REPO_ROOT / "runs" / f"assess_{algo}_s999"
    ckpt_dir = REPO_ROOT / "checkpoints" / f"assess_{algo}_s999"
    if run_dir.is_dir():
        shutil.rmtree(run_dir, ignore_errors=True)
    if ckpt_dir.is_dir():
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    spec = ALGOS[algo]
    cfg = json.dumps(CURRICULUM[0])
    cmd = [
        sys.executable,
        spec["script"],
        spec["step_flag"],
        str(steps),
        "--seed",
        "999",
        "--exp-name",
        f"assess_{algo}",
        "--env-config",
        cfg,
        "--eval-every",
        str(10**9),
    ]
    cmd += spec["flags"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-15:])
        tail += "\n" + "\n".join((proc.stderr or "").splitlines()[-15:])
        print(f"    ! {algo} benchmark FAILED (exit {proc.returncode}):\n{tail}")
        return 0.0
    sps = steps / wall
    return sps


def estimate_trainer_sps(
    measured: dict[str, float],
    env_rows: list[dict],
    vec_rows: list[dict] | None = None,
) -> dict[str, list[float]]:
    """Model per-stage sps per algo from measured stage-0 sps + vec-rollout scaling."""
    out: dict[str, list[float]] = {}
    for algo, sps0 in measured.items():
        if sps0 <= 0:
            out[algo] = [0.0] * len(env_rows)
            continue

        per_step0 = 1.0 / sps0
        vec = ALGOS[algo]["vec"]

        if vec_rows and len(vec_rows) == len(env_rows):
            base_vec_ms = vec_rows[0]["vec_step_ms"]
            est = []
            for s in range(len(env_rows)):
                stage_vec_ms = vec_rows[s]["vec_step_ms"]
                non_rollout_s = max(
                    0.0,
                    per_step0 - (base_vec_ms / 1e3 if vec else base_vec_ms / (8 * 1e3)),
                )
                stage_step_s = (
                    stage_vec_ms / 1e3 if vec else stage_vec_ms / (8 * 1e3)
                ) + non_rollout_s
                est.append(1.0 / stage_step_s if stage_step_s > 0 else 0.0)
            out[algo] = est
        else:
            env_ms = {r["stage"]: r["random_ms"] for r in env_rows}
            mult = 8.0 if vec else 1.0
            const = max(0.0, per_step0 - mult * env_ms[0] / 1e3)
            est = []
            for s in range(len(env_rows)):
                per_step_s = mult * env_ms[s] / 1e3 + const
                est.append(1.0 / per_step_s if per_step_s > 0 else 0.0)
            out[algo] = est
    return out


def hours(steps: int, sps: float) -> float:
    return steps / sps / 3600.0 if sps > 0 else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trainer-steps",
        type=int,
        default=10_240,
        help="steps per trainer benchmark run (default 10240 = 5 updates)",
    )
    ap.add_argument(
        "--env-bench-steps",
        type=int,
        default=200,
        help="steps in the per-stage env micro-benchmark (default 200)",
    )
    ap.add_argument(
        "--algos",
        type=str,
        default="ippo,mappo,qmix,comm",
        help="comma-separated algos to measure end-to-end (default all)",
    )
    ap.add_argument(
        "--skip-vec",
        action="store_true",
        help="skip the vec-rollout overhead benchmark",
    )
    args = ap.parse_args()
    selected = [a.strip() for a in args.algos.split(",") if a.strip()]
    unknown = [a for a in selected if a not in ALGOS]
    if unknown:
        print(f"unknown algos: {unknown}; known: {sorted(ALGOS)}")
        return 2

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("HEIST throughput & wall-clock assessment")
    print("=" * 72)

    info = system_info()
    print("\n[1] System")
    for k, v in info.items():
        print(f"    {k:<14}: {v}")
    if not torch.cuda.is_available():
        print(
            "    !!! CUDA NOT AVAILABLE: estimates reflect CPU, not the campaign GPU."
        )
        print("        Install an NVIDIA driver and a CUDA-enabled torch build.")

    print("\n[2] Raw env step cost per stage (single env)")
    env_rows = env_step_bench(args.env_bench_steps)
    print(
        f"    {'stage':<6}{'map':<8}{'guards':<7}{'wait ms':<10}{'random ms':<10}"
        f"{'rand steps/s':<12}"
    )
    for r in env_rows:
        print(
            f"    {r['stage']:<6}{r['map']:<8}{r['guards']:<7}"
            f"{r['wait_ms']:<10.4f}{r['random_ms']:<10.4f}"
            f"{1000 / max(r['random_ms'], 1e-9):<12.0f}"
        )

    if not args.skip_vec:
        print(
            f"\n[3] One vec-env rollout step (8 envs, {device_name}): "
            f"tensor + policy + env.step"
        )
        vec_rows = vec_rollout_bench()
        print(f"    {'stage':<6}{'vec_step_ms':<12}{'vec steps/s':<12}")
        for r in vec_rows:
            print(f"    {r['stage']:<6}{r['vec_step_ms']:<12.2f}{r['vec_sps']:<12}")

    print(
        f"\n[4] End-to-end trainer throughput at stage-0 "
        f"({args.trainer_steps} steps each, {device_name})"
    )
    measured: dict[str, float] = {}
    for algo in selected:
        sps = run_trainer(algo, args.trainer_steps)
        measured[algo] = sps
        print(
            f"    {algo:<6}: {sps:8.1f} steps/s  "
            f"({hours(args.trainer_steps, sps) * 60:.1f} min for this run)"
        )
    if not measured or all(v == 0 for v in measured.values()):
        print("    ! all trainer benchmarks failed; cannot extrapolate.")
        return 1

    vec_rows_list = vec_rows if not args.skip_vec else None
    est = estimate_trainer_sps(measured, env_rows, vec_rows_list)

    print(
        "\n[5] Estimated per-stage steps/s (stage-0 measured, 1-4 scaled by env cost)"
    )
    print(f"    {'algo':<6}" + "".join(f"{'s' + str(s):>10}" for s in range(5)))
    for algo in selected:
        print(f"    {algo:<6}" + "".join(f"{est[algo][s]:>10.0f}" for s in range(5)))

    print("\n[6] Campaign wall-clock estimates (sequential on one GPU, 24/7)")
    line = "    {:<44}{:>10}{:>10}"
    print(line.format("scenario", "hours", "days"))

    # Scenario A: the 300k stage-0 campaign (experiment_stage0_300k.sh)
    tot_a = sum(hours(300_000, est[a][0]) * 3 for a in selected if est[a][0] > 0)
    print(
        line.format(
            f"A. stage-0 300k ({len(selected)} algos x 3 seeds)",
            f"{tot_a:,.0f}",
            f"{tot_a / 24:,.1f}",
        )
    )

    # Scenario B: full 1M grid, all algos x 3 seeds x 5 stages
    tot_b = 0.0
    for a in selected:
        for s in range(5):
            if est[a][s] > 0:
                tot_b += 3 * hours(1_000_000, est[a][s])
    print(
        line.format(
            "B. full 1M grid (all algos x 3 seeds x 5 stages)",
            f"{tot_b:,.0f}",
            f"{tot_b / 24:,.1f}",
        )
    )

    # Scenario C: documented plan (FUTURE_PLANS.md)
    tot_c = 0.0
    for a in selected:
        if est[a][0] > 0:
            tot_c += 3 * hours(300_000, est[a][0])
    if "ippo" in selected and est["ippo"][0] > 0:
        for s in range(1, 5):
            if est["ippo"][s] > 0:
                tot_c += hours(1_000_000, est["ippo"][s])
    print(
        line.format(
            "C. documented plan (stage-0 full + IPPO 1M s1-4)",
            f"{tot_c:,.0f}",
            f"{tot_c / 24:,.1f}",
        )
    )

    print("NOTE: stages 1-4 assume per-step non-env overhead stays constant;")
    print("      real numbers may differ as obs/state dims grow.  Re-run this")
    print("      script after any P1/P2 performance change to see the gain.")
    print("")
    print("Units: trainer logs print sps in rollout-ITERATIONS/s (each iteration")
    print("      steps num_envs=8 envs).  This tool measures env-steps/s directly,")
    print("      so a campaign-log sps of 74 = 8 x 74 = ~590 env-steps/s.")
    print("      The 20480-step benchmark runs with evals disabled and captures")
    print("      early-throughput; real campaign sps runs ~10% lower (eval +")
    print("      degradation), so expect ~1.1x the hours below on a full run.")
    print("Done.")
    return 0 if torch.cuda.is_available() else 2


if __name__ == "__main__":
    raise SystemExit(main())
