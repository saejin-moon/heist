"""Shared tensor and run-metadata helpers for the PPO-family trainers."""

import json
from pathlib import Path

import torch


def compute_gae(
    rewards,
    values,
    terminated,
    truncated,
    bootstrap,
    next_values,
    next_terminated,
    gamma,
    gae_lambda,
):
    """Compute GAE for all agents together.

    Inputs use ``[agents, time, env]`` layout. Time is necessarily a
    backwards recurrence, but the agent and environment dimensions are
    vectorized so a rollout no longer launches four independent GPU-op chains.
    """
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values)
    num_steps = rewards.shape[1]
    for step in range(num_steps - 1, -1, -1):
        if step == num_steps - 1:
            following_values = next_values
        else:
            following_values = torch.where(
                truncated[:, step].bool(),
                bootstrap[:, step],
                values[:, step + 1],
            )
        nonterminal = 1.0 - terminated[:, step]
        gae_mask = (1.0 - terminated[:, step]) * (1.0 - truncated[:, step])
        delta = (
            rewards[:, step] + gamma * following_values * nonterminal - values[:, step]
        )
        last_gae = delta + gamma * gae_lambda * gae_mask * last_gae
        advantages[:, step] = last_gae
    return advantages, advantages + values


def write_completion(run_name, algorithm, requested_steps, completed_steps):
    """Atomically mark a checkpoint directory as a successful completed run."""
    checkpoint_dir = Path("checkpoints") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker = checkpoint_dir / "complete.json"
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "algorithm": algorithm,
                "requested_steps": requested_steps,
                "completed_steps": completed_steps,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(marker)


def load_matching_weights(model, filepath, device):
    """Loads weights from filepath into model, filtering out dimension mismatches (e.g. centralized critic shape changes)."""
    import os

    if not filepath or not os.path.isfile(filepath):
        return False
    try:
        sd = torch.load(filepath, map_location=device)
        model_sd = model.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in model_sd:
                if v.shape == model_sd[k].shape:
                    filtered_sd[k] = v
                else:
                    print(
                        f"  [Transfer] Shape mismatch for {k}: checkpoint {v.shape} vs model {model_sd[k].shape}. Reinitializing."
                    )
        model_sd.update(filtered_sd)
        model.load_state_dict(model_sd)
        print(f"  [Transfer] Loaded {len(filtered_sd)} matching layers from {filepath}")
        return True
    except Exception as e:
        print(f"  [Transfer] Warning: could not load checkpoint from {filepath}: {e}")
        return False
