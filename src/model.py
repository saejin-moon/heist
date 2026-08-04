"""
Neural network components for HEIST training.

All networks share the observation contract from env.py:
    observation  : 5x5 Fog-Masked Box (25 values after flatten)
    action_mask  : 6-element binary gate
    global_state : 4-element phase vector (for IPPO actor-critic inputs)

MAPPO and QMIX additionally consume the richer centralized `env.state()`.

Classes
-------
HeistAgent   : independent IPPO actor-critic (flattened obs + 4-vec global_state)
MappoAgent   : shared actor + centralized critic (consumes env.state())
QNetwork     : per-agent deep Q network (for QMIX / independent DQN)
QMixMixing   : QMIX monotonic mixing network conditioned on global state
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from constants import GLOBAL_STATE_DIM

# obs (25) + global_state (GLOBAL_STATE_DIM) fused into the local feature vector
LOCAL_INPUT_DIM = 25 + GLOBAL_STATE_DIM
ACTION_DIM = 6
HIDDEN_DIM = 64


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HeistAgent(nn.Module):
    """Independent PPO actor-critic used by IPPO.

    The critic and actor consume the same local feature vector
    (flattened 5x5 observation + 4-element global_state), which is the
    standard IPPO (independent PPO) setup.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )

    @staticmethod
    def _process_inputs(obs, global_state):
        flat_obs = torch.flatten(obs, start_dim=1).float()
        return torch.cat((flat_obs, global_state.float()), dim=1)

    def get_value(self, obs, global_state):
        return self.critic(self._process_inputs(obs, global_state))

    def get_action_and_value(self, obs, global_state, action_mask, action=None):
        x = self._process_inputs(obs, global_state)
        logits = self.actor(x)

        HUGE_NEG = torch.tensor(-1e9, device=logits.device)
        masked_logits = torch.where(action_mask == 1, logits, HUGE_NEG)

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class MappoAgent(nn.Module):
    """MAPPO: shared actor (local obs) + centralized critic (global state).

    The actor is identical to HeistAgent's; the critic consumes the full
    `env.state()` vector (grid + positions + phase flags), which removes
    partial observability for value estimation only.
    """

    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, global_state):
        return self.critic(global_state)

    def get_action_and_value(self, obs, global_state, action_mask, state, action=None):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), global_state.float()), dim=1)
        logits = self.actor(x)
        HUGE_NEG = torch.tensor(-1e9, device=logits.device)
        masked_logits = torch.where(action_mask == 1, logits, HUGE_NEG)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)


class QNetwork(nn.Module):
    """Per-agent deep Q network (DQN-style).  Used by QMIX.

    Consumes the flattened 5x5 observation + 4-element global_state and
    outputs Q-values for all 6 actions.  Invalid actions are masked with
    a large negative value at mixing time.
    """

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=1.0),
        )

    def forward(self, obs, global_state):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), global_state.float()), dim=1)
        return self.net(x)


class QMixMixing(nn.Module):
    """QMIX monotonic mixing network.

    Given per-agent Q-values Q_i and the global state, outputs the joint
    Q_tot.  Hypernetworks produce the mixing weights from the global state;
    the absolute-value reparameterization guarantees the monotonicity
    constraint dQ_tot / dQ_i >= 0 required for value decomposition.
    """

    def __init__(self, n_agents, state_dim, embed_dim=32, hyper_hidden=64):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # hypernetworks: state -> per-agent mixing weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim * n_agents),
        )
        # state -> scalar bias for first mixing layer
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # hypernetwork: state -> weights for second mixing layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, 1),
        )

    def forward(self, q_values, states):
        """q_values: [B, n_agents]; states: [B, state_dim] -> Q_tot [B, 1]."""
        batch = states.shape[0]
        w1 = torch.abs(self.hyper_w1(states))  # [B, embed*n]
        w1 = w1.view(batch, self.n_agents, -1)  # [B, n, embed]
        b1 = self.hyper_b1(states).view(batch, 1, -1)  # [B, 1, embed]
        hidden = torch.relu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # [B, 1, embed]
        w2 = torch.abs(self.hyper_w2(states)).view(batch, -1, 1)  # [B, embed, 1]
        b2 = self.hyper_b2(states).view(batch, 1, 1)  # [B, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2  # [B, 1, 1]
        return q_tot.squeeze(-1).squeeze(-1)  # [B]
