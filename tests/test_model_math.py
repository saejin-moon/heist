import pytest
import torch

from constants import ACTION_SPACE_SIZE, N_AGENTS, OBSERVATION_SIZE
from model import (
    ComaAgent,
    CommAgent,
    ContinuousHierarchicalAgent,
    CoopAgent,
    HeistAgent,
    MappoAgent,
    QMixMixing,
)

ACTION_DIM = ACTION_SPACE_SIZE

# Dummy state size
STATE_DIM = 121 + 6 + 8


def test_heist_agent_shapes():
    agent = HeistAgent(STATE_DIM)
    obs = torch.zeros(2, *OBSERVATION_SIZE)
    role = torch.zeros(2, N_AGENTS)
    mask = torch.ones(2, ACTION_DIM)

    action, log_prob, entropy, value = agent.get_action_and_value(obs, role, mask)
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)


def test_heist_agent_nan_inf_inputs():
    agent = HeistAgent(STATE_DIM)
    obs = torch.full((2, *OBSERVATION_SIZE), float("nan"))
    role = torch.full((2, N_AGENTS), float("nan"))
    mask = torch.ones(2, ACTION_DIM)

    # We expect ValueError because PyTorch Categorical doesn't accept NaNs
    with pytest.raises(ValueError):
        _, log_prob, _, value = agent.get_action_and_value(obs, role, mask)


def test_mappo_agent_influence_scalar():
    agent = MappoAgent(STATE_DIM)
    state = torch.randn(2, STATE_DIM)
    inf = agent.get_influence_scalar(state)
    assert inf.shape == (2, 4)
    # Check row sums to 1
    assert torch.allclose(inf.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_coma_agent_influence_matrix():
    agent = ComaAgent(STATE_DIM)
    state = torch.randn(2, STATE_DIM)
    actions = {i: torch.randint(0, ACTION_SPACE_SIZE, (2,)) for i in range(4)}
    inf = agent.get_influence_matrix(state, actions)
    assert inf.shape == (2, 4, 4)
    assert torch.allclose(inf.diagonal(dim1=1, dim2=2), torch.zeros(2, 4))
    assert torch.allclose(inf.sum(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_comm_agent_shapes():
    agent = CommAgent(STATE_DIM)
    obs_list = [torch.zeros(2, *OBSERVATION_SIZE) for _ in range(N_AGENTS)]
    role_list = [torch.zeros(2, N_AGENTS) for _ in range(N_AGENTS)]
    mask_list = [torch.ones(2, ACTION_DIM) for _ in range(N_AGENTS)]
    state = torch.zeros(2, STATE_DIM)

    action, log_prob, entropy, value = agent.get_action_and_value(
        obs_list, role_list, mask_list, state
    )
    assert action.shape == (2, 4)
    assert log_prob.shape == (2, 4)
    assert entropy.shape == (2, 4)
    assert value.shape == (2, 4)


def test_coop_agent_shapes():
    agent = CoopAgent(STATE_DIM, max_experts=3)
    obs = torch.zeros(2, *OBSERVATION_SIZE)
    role = torch.zeros(2, N_AGENTS)
    mask = torch.ones(2, ACTION_DIM)
    state = torch.zeros(2, STATE_DIM)

    action, log_prob, entropy, value, chosen_expert, true_expert = (
        agent.get_action_and_value(obs, state, role, mask, active_experts=3)
    )
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)
    assert chosen_expert.shape == (2,)
    assert true_expert.shape == (2,)


def test_qmix_mixing_network():
    net = QMixMixing(4, STATE_DIM)
    q_vals = torch.randn(2, 4)
    state = torch.randn(2, STATE_DIM)

    q_tot = net(q_vals, state)
    assert q_tot.shape == (2,)


def test_continuous_hierarchical_agent():
    agent = ContinuousHierarchicalAgent(STATE_DIM)
    obs = torch.zeros(2, *OBSERVATION_SIZE)
    role = torch.zeros(2, N_AGENTS)
    state = torch.zeros(2, STATE_DIM)
    mask = torch.ones(2, ACTION_DIM)

    goal, g_log_prob, g_entropy, g_value = agent.get_manager_action_and_value(
        obs, state, role
    )
    assert goal.shape == (2, 2)
    assert g_log_prob.shape == (2,)
    assert g_entropy.shape == (2,)
    assert g_value.shape == (2, 1)

    action, log_prob, entropy, value = agent.get_worker_action_and_value(
        obs, state, role, goal, mask
    )
    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)


def test_zero_division_avoidance_in_softmax():
    # If mask is all 0, ensure we don't NaN out in categorical
    agent = HeistAgent(STATE_DIM)
    obs = torch.zeros(2, *OBSERVATION_SIZE)
    role = torch.zeros(2, N_AGENTS)
    mask = torch.zeros(2, ACTION_DIM)

    action, log_prob, entropy, value = agent.get_action_and_value(obs, role, mask)
    assert not torch.isnan(action).any()
