"""
Smoke test for CIR (Causal Influence Routing) in CommAgent.

Validates that:
1. CommAgent.get_influence_matrix outputs routing matrix [B, N_AGENTS, N_AGENTS].
2. Diagonal elements (self-influence) are 0.
3. Rows sum to 1.0 (normalized routing weights across senders).

Run with: uv run pytest src/test_cir_smoke.py
"""

import torch

from env import AGENTS
from model import CommAgent
from vec_env import VectorEnv


def test_cir_influence_matrix():
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
