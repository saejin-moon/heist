import torch

from constants import ACTION_SPACE_SIZE, OBSERVATION_SIZE
from model import EcoopAgent


def test_ecoop_agent_forward():
    state_dim = 100
    max_experts = 3
    agent = EcoopAgent(state_dim=state_dim, max_experts=max_experts)

    B = 4
    obs = torch.randn(B, *OBSERVATION_SIZE)
    state = torch.randn(B, state_dim)
    role_id = torch.eye(4)[torch.randint(0, 4, (B,))]
    action_mask = torch.ones(B, ACTION_SPACE_SIZE)

    active_experts = 2
    previous_expert = torch.zeros(B, dtype=torch.long)

    # Test normal forward with hysteresis
    action, log_prob, entropy, value, chosen_expert, true_expert = (
        agent.get_action_and_value(
            obs,
            state,
            role_id,
            action_mask,
            active_experts,
            previous_expert,
            epsilon_abs=0.05,
            epsilon_rel=0.05,
        )
    )

    assert action.shape == (B,)
    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)
    assert value.shape == (B, 1)
    assert chosen_expert.shape == (B,)
    assert true_expert.shape == (B,)


def test_ecoop_agent_expert_override():
    state_dim = 100
    max_experts = 3
    agent = EcoopAgent(state_dim=state_dim, max_experts=max_experts)

    B = 2
    obs = torch.randn(B, *OBSERVATION_SIZE)
    state = torch.randn(B, state_dim)
    role_id = torch.eye(4)[torch.randint(0, 4, (B,))]
    action_mask = torch.ones(B, ACTION_SPACE_SIZE)

    active_experts = 2
    previous_expert = torch.zeros(B, dtype=torch.long)
    expert_idx = torch.ones(B, dtype=torch.long)

    # Test overriding expert
    action, log_prob, entropy, value, chosen_expert, true_expert = (
        agent.get_action_and_value(
            obs,
            state,
            role_id,
            action_mask,
            active_experts,
            previous_expert,
            expert_idx=expert_idx,
        )
    )

    assert torch.all(chosen_expert == expert_idx)
    assert torch.all(true_expert == expert_idx)


def test_ecoop_agent_grace_period():
    state_dim = 100
    max_experts = 3
    agent = EcoopAgent(state_dim=state_dim, max_experts=max_experts)

    B = 2
    obs = torch.randn(B, *OBSERVATION_SIZE)
    state = torch.randn(B, state_dim)
    role_id = torch.eye(4)[torch.randint(0, 4, (B,))]
    action_mask = torch.ones(B, ACTION_SPACE_SIZE)

    active_experts = 3
    previous_expert = torch.zeros(B, dtype=torch.long)
    grace_expert = 2

    # Test grace period
    action, log_prob, entropy, value, chosen_expert, true_expert = (
        agent.get_action_and_value(
            obs,
            state,
            role_id,
            action_mask,
            active_experts,
            previous_expert,
            grace_period_expert=grace_expert,
        )
    )

    assert torch.all(chosen_expert == grace_expert)
