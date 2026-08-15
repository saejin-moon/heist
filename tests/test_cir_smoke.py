"""
Smoke test for CIR (Causal Influence Routing) in CommAgent, MappoAgent, and ComaAgent.

Validates that:
1. get_influence_matrix outputs routing matrix [B, N_AGENTS, N_AGENTS].
2. Diagonal elements (self-influence) are 0.
3. Rows sum to 1.0 (normalized routing weights across senders).

Run with: uv run pytest tests/test_cir_smoke.py
"""

import torch

from env import AGENTS
from model import ComaAgent, CommAgent, MappoAgent
from vec_env import VectorEnv


def test_comm_cir_influence_matrix():
    vec = VectorEnv(2, config={"map_size": (11, 11), "guard_count": 0}, base_seed=0)
    obs, _ = vec.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = CommAgent(state_dim=vec.state_dim, centralized=False).to(device)

    stacked = obs["_stacked"]
    obs_t = torch.as_tensor(stacked["observation"], device=device)
    role_t = torch.as_tensor(stacked["role_id"], device=device)

    obs_list = [obs_t[i] for i in range(len(AGENTS))]
    role_list = [role_t[i] for i in range(len(AGENTS))]

    routing = policy.get_influence_matrix(obs_list, role_list)

    assert routing.shape == (2, len(AGENTS), len(AGENTS))

    # Check diagonal (self-influence) is 0
    for i in range(len(AGENTS)):
        assert torch.allclose(
            routing[:, i, i], torch.zeros(2, device=device), atol=1e-5
        )

    # Check row sums equal 1.0
    row_sums = routing.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums, device=device), atol=1e-4)

    vec.close()


def test_mappo_cir_influence_scalar():
    vec = VectorEnv(2, config={"map_size": (11, 11), "guard_count": 0}, base_seed=0)
    vec.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = MappoAgent(state_dim=vec.state_dim).to(device)
    state_t = torch.as_tensor(vec.state, device=device)

    routing = policy.get_influence_scalar(state_t)

    assert routing.shape == (2, len(AGENTS))

    row_sums = routing.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums, device=device), atol=1e-4)

    vec.close()


def test_coma_cir_influence_matrix():
    vec = VectorEnv(2, config={"map_size": (11, 11), "guard_count": 0}, base_seed=0)
    vec.reset()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = ComaAgent(state_dim=vec.state_dim).to(device)
    state_t = torch.as_tensor(vec.state, device=device)
    actions_t = torch.randint(0, 6, (len(AGENTS), 2), device=device)

    routing = policy.get_influence_matrix(state_t, actions_t)

    assert routing.shape == (2, len(AGENTS), len(AGENTS))

    for i in range(len(AGENTS)):
        assert torch.allclose(
            routing[:, i, i], torch.zeros(2, device=device), atol=1e-5
        )

    row_sums = routing.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums, device=device), atol=1e-4)

    vec.close()


def test_cir_advantage_routing_math():
    steps, envs, n_agents = 10, 4, 4
    buf_influence = torch.rand((steps, envs, n_agents, n_agents))
    for i in range(n_agents):
        buf_influence[:, :, i, i] = 0.0
    buf_influence = buf_influence / (buf_influence.sum(dim=-1, keepdim=True) + 1e-8)

    adv_t = torch.randn((steps, envs, n_agents))
    cir_coef = 0.5

    routed_adv = torch.einsum("sten,ste->stn", buf_influence, adv_t)
    adv_t_new = (1.0 - cir_coef) * adv_t + (cir_coef * routed_adv)

    assert adv_t_new.shape == (steps, envs, n_agents)
    assert not torch.isnan(adv_t_new).any()
