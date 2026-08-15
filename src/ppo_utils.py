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

    nonterminal_mask = 1.0 - terminated
    gae_mask_all = nonterminal_mask * (1.0 - truncated)

    for step in range(num_steps - 1, -1, -1):
        if step == num_steps - 1:
            following_values = next_values
        else:
            following_values = torch.where(
                truncated[:, step].bool(),
                bootstrap[:, step],
                values[:, step + 1],
            )
        nonterminal = nonterminal_mask[:, step]
        gae_mask = gae_mask_all[:, step]
        delta = (
            rewards[:, step] + gamma * following_values * nonterminal - values[:, step]
        )
        last_gae = delta + gamma * gae_lambda * gae_mask * last_gae
        advantages[:, step] = last_gae
    return advantages, advantages + values


def compute_gae_simple(
    rewards,
    values,
    next_values,
    dones,
    gamma,
    gae_lambda,
):
    """Simplified GAE computation for agents that don't track truncated/bootstrap."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values)
    num_steps = rewards.shape[0]
    nonterminal_mask = 1.0 - dones
    for step in range(num_steps - 1, -1, -1):
        following_values = next_values if step == num_steps - 1 else values[step + 1]
        nonterminal = nonterminal_mask[step]
        delta = rewards[step] + gamma * following_values * nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[step] = last_gae
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
        sd = torch.load(filepath, map_location=device, weights_only=True)
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


def get_previous_stage_checkpoint(run_name, exp_name=""):
    """Parse experiment/run name to find candidate checkpoint paths from the previous stage or normal environment."""
    import os
    import re

    for name in [exp_name, run_name]:
        if not name:
            continue
        match = re.search(r"^(.*?)(_st)?_s(\d+)(?:_s\d+)?$", name)
        if match:
            prefix, st, stage_str = match.group(1), match.group(2) or "", match.group(3)
            stage = int(stage_str)
            if stage > 0:
                prev_stage = stage - 1
                candidates = [
                    os.path.join("checkpoints", f"{prefix}{st}_s{prev_stage}"),
                    os.path.join("checkpoints", f"{prefix}_s{prev_stage}"),
                ]
                for c in candidates:
                    if os.path.isdir(c) and os.listdir(c):
                        return c
            elif st == "_st":
                # For Stage 0 side-tasks, work backward from highest finished stage (s4 -> s3 -> s2 -> s1 -> s0)
                for s_idx in range(4, -1, -1):
                    c = os.path.join("checkpoints", f"{prefix}_s{s_idx}")
                    if os.path.isdir(c) and os.listdir(c):
                        return c
    return None


def compute_counterfactual_advantage(
    policy_probs, q_values_all, taken_actions, action_mask
):
    """Computes COMA counterfactual advantage for a agent batch.

    Args:
        policy_probs:  [B, ACTION_DIM] Action probabilities pi_i(a_i' | o_i) from actor
        q_values_all:  [B, ACTION_DIM] Q-values for all actions Q_i(s, a_{-i}, .) from critic
        taken_actions: [B] Action indices actually taken by agent i
        action_mask:   [B, ACTION_DIM] Binary mask (1=legal, 0=illegal)

    Returns:
        advantage: [B] Counterfactual advantage A_i(s, a)
        q_taken:   [B] Q-value of the executed joint action Q_i(s, a_{-i}, a_i)
    """
    # 1. Mask illegal Q-values to avoid zero-probability/negative-infinity issues
    masked_q = torch.where(
        action_mask == 1, q_values_all, torch.zeros_like(q_values_all)
    )

    # 2. Extract Q-value of taken action
    q_taken = q_values_all.gather(1, taken_actions.unsqueeze(1)).squeeze(1)

    # 3. Compute Counterfactual Baseline: \sum_{a_i'} pi_i(a_i' | o_i) * Q_i(s, a_{-i}, a_i')
    baseline = torch.sum(policy_probs * masked_q, dim=-1)

    # 4. Counterfactual Advantage
    advantage = q_taken - baseline
    return advantage, q_taken


def compute_loo_advantage(q_values_all, taken_actions, action_mask):
    r"""Computes Leave-One-Out (C3-style) marginal advantage for an agent batch.

    Formula:
        A_i^{LOO}(s, a_i) = Q_i(s, a_{-i}, a_i) - (1 / (|A_legal| - 1)) * \sum_{a' != a_i} Q_i(s, a_{-i}, a')
    """
    q_taken = q_values_all.gather(1, taken_actions.unsqueeze(1)).squeeze(1)
    legal_counts = torch.sum(action_mask, dim=-1) - 1.0
    legal_counts = torch.clamp(legal_counts, min=1.0)

    # Sum of all legal Q-values minus the taken action Q-value
    masked_q = torch.where(
        action_mask == 1, q_values_all, torch.zeros_like(q_values_all)
    )
    sum_all_q = torch.sum(masked_q, dim=-1)
    sum_other_q = sum_all_q - q_taken

    baseline = sum_other_q / legal_counts
    advantage = q_taken - baseline
    return advantage, q_taken


def compute_ate_advantage(q_values_all, taken_actions, action_mask, null_action_idx=4):
    """Computes Average Treatment Effect (ATE) advantage against the explicit WAIT null action.

    Formula:
        A_i^{ATE}(s, a_i) = Q_i(s, a_{-i}, a_i) - Q_i(s, a_{-i}, a_{WAIT})
    """
    q_taken = q_values_all.gather(1, taken_actions.unsqueeze(1)).squeeze(1)
    q_wait = q_values_all[:, null_action_idx]
    advantage = q_taken - q_wait
    return advantage, q_taken
