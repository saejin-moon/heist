"""
Dummy actor pipeline test.

A minimal PyTorch policy that verifies the env -> tensor -> network ->
masked action pipeline end to end.  This is the smallest possible "does the
environment talk to the network" smoke test and is the first thing to run
after modifying env.py or model.py:

    uv run python src/dummy.py
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from env import HeistEnv, AGENTS
from constants import ACTION_SPACE_SIZE, GLOBAL_STATE_DIM, N_AGENTS


class DummyActor(nn.Module):
    """39 inputs (5x5 obs + 10-vec global_state + 4 role one-hot) -> 6 logits.

    REV-2 (REVISION_PLAN.md §2): the input dim is rebuilt from
    GLOBAL_STATE_DIM + N_AGENTS so the smoke test tracks the real contract.
    REV-7 (REVISION_PLAN.md §6): global_state is deleted in the learned-
    communication milestone; this actor then receives only obs + role + message.
    """

    def __init__(self):
        super().__init__()
        in_dim = 25 + GLOBAL_STATE_DIM + N_AGENTS
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_SPACE_SIZE),
        )

    def forward(self, obs_matrix, global_state, role_id, action_mask):
        flat_obs = torch.flatten(obs_matrix)
        combined = torch.cat((flat_obs, global_state, role_id))
        logits = self.network(combined)
        masked_logits = torch.where(action_mask == 1, logits, torch.tensor(-1e9))
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        return action.item(), dist.probs


def make_env():
    """Factory used by `pettingzoo.test.parallel_test -e dummy:make_env`."""
    return HeistEnv()


if __name__ == "__main__":
    env = HeistEnv()
    obs, _ = env.reset(seed=0)

    actor = DummyActor()
    for agent in AGENTS:
        o = obs[agent]
        t_obs = torch.tensor(o["observation"], dtype=torch.float32)
        t_gs = torch.tensor(o["global_state"], dtype=torch.float32)
        t_role = torch.tensor(o["role_id"], dtype=torch.float32)
        t_mask = torch.tensor(o["action_mask"], dtype=torch.int32)
        chosen, probs = actor(t_obs, t_gs, t_role, t_mask)
        print(f"{agent:>9} mask={o['action_mask']} action={chosen} probs={probs}")
