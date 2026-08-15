"""
Smoke test for MARC (Marginal Action Retroactive Credit) advantage computation.

Validates that:
1. compute_marc_advantages produces correct tensor shapes [N_AGENTS, T, B] for advantages and returns.
2. Full MARC, marc_no_shielding, marc_no_macro, and marc_no_affordance compute correctly.
3. Binary success masking protects upstream enablers on team failure.

Run with: uv run pytest tests/test_marc_smoke.py
"""

import torch

from train_marc import compute_marc_advantages


def test_marc_advantages_shape_and_ablation_modes():
    num_agents = 4
    num_steps = 16
    num_envs = 2

    rewards = torch.zeros(num_agents, num_steps, num_envs)
    values = torch.zeros(num_agents, num_steps, num_envs)
    dones = torch.zeros(num_agents, num_steps, num_envs)
    truncs = torch.zeros(num_agents, num_steps, num_envs)
    alarms = torch.zeros(num_steps, num_envs)
    car_events = torch.zeros(num_agents, num_steps, num_envs, dtype=torch.bool)

    # Set terminal win in env 0
    rewards[:, -1, 0] = 10.0
    # Set step rewards and alarm in env 1
    rewards[0, 2, 1] = 0.5
    car_events[0, 2, 1] = True
    alarms[:, 1] = 25.0

    # 1. Full MARC
    adv_full, ret_full = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.5,
        no_shielding=False,
        no_macro=False,
    )
    assert adv_full.shape == (num_agents, num_steps, num_envs)
    assert ret_full.shape == (num_agents, num_steps, num_envs)

    # 2. MARC No Shielding
    adv_no_shield, _ = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.5,
        no_shielding=True,
        no_macro=False,
    )
    assert adv_no_shield.shape == (num_agents, num_steps, num_envs)

    # 3. MARC No Macro
    adv_no_macro, _ = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.5,
        no_shielding=False,
        no_macro=True,
    )
    assert adv_no_macro.shape == (num_agents, num_steps, num_envs)

    # 4. MARC No Affordance
    adv_no_aff, _ = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.0,
        no_shielding=False,
        no_macro=False,
    )
    assert adv_no_aff.shape == (num_agents, num_steps, num_envs)


def test_binary_success_masking():
    # In a losing environment (rewards <= 0), an enabler (car_events=True) should be shielded
    num_agents = 4
    num_steps = 4
    num_envs = 1

    rewards = torch.zeros(num_agents, num_steps, num_envs)
    values = torch.zeros(num_agents, num_steps, num_envs)
    dones = torch.zeros(num_agents, num_steps, num_envs)
    truncs = torch.zeros(num_agents, num_steps, num_envs)
    alarms = torch.zeros(num_steps, num_envs)
    car_events = torch.zeros(num_agents, num_steps, num_envs, dtype=torch.bool)

    # Agent 0 (Scout) unlocks terminal at step 1
    car_events[0, 1, 0] = True
    rewards[0, 1, 0] = 0.5

    # Full MARC (with shielding)
    adv_shielded, _ = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.5,
        no_shielding=False,
        no_macro=False,
    )

    # MARC No Shielding (without shielding)
    adv_unshielded, _ = compute_marc_advantages(
        rewards,
        values,
        dones,
        truncs,
        alarms,
        car_events,
        gamma=0.99,
        gae_lambda=0.95,
        alpha_alarm=1.5,
        gamma_causal=0.95,
        affordance_coef=0.5,
        no_shielding=True,
        no_macro=False,
    )

    # Agent 1 (non-enabler) gets negative outcome penalty in unshielded
    # Agent 0 (enabler) maintains higher advantage in shielded than unshielded on failure
    assert adv_shielded[0, 1, 0] > adv_unshielded[0, 1, 0]
