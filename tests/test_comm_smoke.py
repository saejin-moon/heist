"""
Communication pipeline smoke test (REV-7, Phase B).

Verifies the TarMACComm module, CommAgent, and the full rollout + gradient
step for the shared communication trainer.

Run with:  uv run python src/test_comm_smoke.py
"""

import torch

from constants import N_AGENTS, OBSERVATION_SIZE
from env import HeistEnv
from model import (
    COMM_HIDDEN_DIM,
    COMM_MESSAGE_DIM,
    LOCAL_INPUT_DIM,
    CommAgent,
    TarMACComm,
)
from vec_env import VectorEnv


def test_tarmac_forward():
    """TarMACComm produces correct shapes."""
    comm = TarMACComm(COMM_HIDDEN_DIM, COMM_MESSAGE_DIM)
    B = 4
    features = torch.randn(B, N_AGENTS, COMM_HIDDEN_DIM)
    aggregated, msgs_gated, attn = comm(features)
    assert aggregated.shape == (B, N_AGENTS, COMM_MESSAGE_DIM), (
        f"expected ({B},{N_AGENTS},{COMM_MESSAGE_DIM}), got {aggregated.shape}"
    )
    assert msgs_gated.shape == aggregated.shape
    assert attn.shape == (B, N_AGENTS, N_AGENTS)
    # attention rows should sum to 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"attention rows don't sum to 1: {row_sums}"
    )
    print("  TarMACComm forward OK")


def test_comm_agent_shapes():
    """CommAgent produces correct action/value shapes."""
    agent = CommAgent(state_dim=100, centralized=False)
    B = 4
    obs = torch.randn(B, N_AGENTS, OBSERVATION_SIZE[0], OBSERVATION_SIZE[1])
    roles = [torch.zeros(B, N_AGENTS) for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        roles[i][:, i] = 1.0
    masks = [torch.ones(B, 6) for _ in range(N_AGENTS)]
    obs_list = [obs[:, i] for i in range(N_AGENTS)]

    actions, logprobs, entropy, values = agent.get_action_and_value(
        obs_list, roles, masks
    )
    assert actions.shape == (B, N_AGENTS)
    assert logprobs.shape == (B, N_AGENTS)
    assert entropy.shape == (B, N_AGENTS)
    assert values.shape == (B, N_AGENTS)
    assert agent._last_attention is not None
    assert agent._last_attention.shape == (B, N_AGENTS, N_AGENTS)
    print("  CommAgent shapes OK")


def test_comm_agent_centralized():
    """MAPPO-style CommAgent with centralized critic."""
    agent = CommAgent(state_dim=100, centralized=True)
    B = 4
    obs = torch.randn(B, N_AGENTS, OBSERVATION_SIZE[0], OBSERVATION_SIZE[1])
    roles = [torch.zeros(B, N_AGENTS) for _ in range(N_AGENTS)]
    for i in range(N_AGENTS):
        roles[i][:, i] = 1.0
    masks = [torch.ones(B, 6) for _ in range(N_AGENTS)]
    obs_list = [obs[:, i] for i in range(N_AGENTS)]
    state = torch.randn(B, 100)

    _, _, _, values = agent.get_action_and_value(obs_list, roles, masks, state=state)
    assert values.shape == (B, N_AGENTS)
    print("  CommAgent centralized OK")


def test_comm_gradient_step():
    """One gradient step through CommAgent with the vectorized env."""
    env_config = {
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
    }
    vec_env = VectorEnv(2, config=env_config, base_seed=0)
    next_obs, next_state = vec_env.reset(seed=0)

    agent = CommAgent(state_dim=vec_env.state_dim, centralized=False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    # collect one step
    obs_list = [
        torch.tensor(next_obs[a]["observation"], device="cpu")
        for a in vec_env.envs[0].possible_agents
    ]
    role_list = [
        torch.tensor(next_obs[a]["role_id"], device="cpu")
        for a in vec_env.envs[0].possible_agents
    ]
    mask_list = [
        torch.tensor(next_obs[a]["action_mask"], device="cpu")
        for a in vec_env.envs[0].possible_agents
    ]

    actions, logprobs, entropy, values = agent.get_action_and_value(
        obs_list, role_list, mask_list
    )

    # dummy loss
    loss = -entropy.mean() + values.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  CommAgent gradient step OK")


def test_obs_contract():
    """Env obs dict no longer contains global_state."""
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 60,
        }
    )
    obs, _ = env.reset(seed=0)
    for a in env.agents:
        assert "observation" in obs[a]
        assert "action_mask" in obs[a]
        assert "role_id" in obs[a]
        assert "global_state" not in obs[a], (
            f"global_state should be removed (REV-7), but found in obs for {a}"
        )
    print("  obs contract (no global_state) OK")


def test_local_input_dim():
    """LOCAL_INPUT_DIM = obs_flat + N_AGENTS (no global_state)."""
    expected = OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1] + N_AGENTS
    assert LOCAL_INPUT_DIM == expected, f"expected {expected}, got {LOCAL_INPUT_DIM}"
    print("  LOCAL_INPUT_DIM OK")


if __name__ == "__main__":
    print("test_comm_smoke:")
    test_obs_contract()
    test_local_input_dim()
    test_tarmac_forward()
    test_comm_agent_shapes()
    test_comm_agent_centralized()
    test_comm_gradient_step()
    print("all communication pipeline tests PASSED")
