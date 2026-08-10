"""
Neural network components for HEIST training.

All networks share the observation contract from env.py:
    observation  : Flattened 7x7 Fog-Masked Box (49 values)
    action_mask  : 6-element binary gate
    role_id      : 4-element one-hot identifying the controlling role (REV-2)

REV-7 (REVISION_PLAN.md §6): global_state is deleted from the per-agent
observation contract.  Phase A baselines navigate purely from local view.
Phase B communication agents learn to share phase status via TarMAC messages.

MAPPO and QMIX additionally consume the richer centralized ``env.state()``.

Classes
-------
HeistAgent    : independent IPPO actor-critic  (Phase A baseline)
MappoAgent    : shared actor + centralized critic (Phase A baseline)
QNetwork      : per-agent DQN for QMIX (Phase A baseline)
QMixMixing    : QMIX monotonic mixing network
TarMACComm    : differentiable attention-based message passing (Phase B)
CommAgent     : actor-critic that consumes TarMAC aggregated messages
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from constants import N_AGENTS, OBSERVATION_SIZE

# local obs (7x7=49) + role one-hot (4) = 53
LOCAL_INPUT_DIM = OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1] + N_AGENTS  # 53

# TarMAC communication dimensions (Phase B)
COMM_HIDDEN_DIM = 64  # shared encoder dim for message / query
COMM_MESSAGE_DIM = 32  # per-agent message vector length
# Comm agent input: obs (49) + role (4) + attended message (32) = 85
COMM_INPUT_DIM = LOCAL_INPUT_DIM + COMM_MESSAGE_DIM  # 85

ACTION_DIM = 6
HIDDEN_DIM = 64


def layer_init(layer, std=None, bias_const=0.0):
    if std is None:
        std = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ──────────────────────────────────────────────────────────────────────
# Phase A baseline agents (no communication)
# ──────────────────────────────────────────────────────────────────────


class HeistAgent(nn.Module):
    """Independent PPO actor-critic used by IPPO.

    The critic and actor consume the same local feature vector
    (flattened 5x5 observation + role one-hot).
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
    def _process_inputs(obs, role_id):
        flat_obs = torch.flatten(obs, start_dim=1).float()
        return torch.cat((flat_obs, role_id.float()), dim=1)

    def get_value(self, obs, role_id):
        return self.critic(self._process_inputs(obs, role_id))

    def get_action_and_value(self, obs, role_id, action_mask, action=None):
        x = self._process_inputs(obs, role_id)
        logits = self.actor(x)

        masked_logits = logits.masked_fill(action_mask != 1, -1e9)

        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class MappoAgent(nn.Module):
    """MAPPO: shared actor (local obs) + centralized critic (global state).

    The actor is identical to HeistAgent's; the critic consumes the full
    ``env.state()`` vector (grid + positions + phase flags), which removes
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

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, obs, role_id, action_mask, state, action=None):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), role_id.float()), dim=1)
        logits = self.actor(x)
        masked_logits = logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)

    def get_influence_matrix(self, state):
        """Calculates Non-Communicating Causal Influence Routing (CIR) matrix via Counterfactual Feature Ablation for MAPPO."""
        B, state_dim = state.shape
        v_base = self.critic(state).squeeze(-1)  # [B]

        # Calculate map grid length: state_dim = 6 + H*W + 2*N_AGENTS + 2*guard_count + guard_count
        n_rem = state_dim - 6 - 8
        grid_len = 121
        for sq in (2500, 961, 441, 225, 121):
            if sq <= n_rem:
                grid_len = sq
                break

        agent_pos_start = 6 + grid_len
        influence_matrix = torch.zeros((B, N_AGENTS, N_AGENTS), device=state.device)

        for j in range(N_AGENTS):
            state_alt = state.clone()
            pos_col = agent_pos_start + 2 * j
            state_alt[:, pos_col : pos_col + 2] = 0.0

            v_alt = self.critic(state_alt).squeeze(-1)  # [B]
            diff_j = torch.abs(v_base - v_alt)  # [B]

            for i in range(N_AGENTS):
                if i != j:
                    influence_matrix[:, i, j] = diff_j

        mask = torch.eye(N_AGENTS, device=state.device).bool()
        influence_matrix.masked_fill_(mask, 0.0)

        row_sums = influence_matrix.sum(dim=-1, keepdim=True) + 1e-8
        return influence_matrix / row_sums


class QNetwork(nn.Module):
    """Per-agent deep Q network (DQN-style).  Used by QMIX.

    Consumes the flattened 5x5 observation + role one-hot and
    outputs Q-values for all 6 actions.
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

    def forward(self, obs, role_id):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), role_id.float()), dim=1)
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
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim * n_agents),
        )
        # state -> scalar bias for first mixing layer
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        # hypernetwork: state -> weights for second mixing layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
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


# ──────────────────────────────────────────────────────────────────────
# Phase B: TarMAC-style learned communication
# ──────────────────────────────────────────────────────────────────────


class TarMACComm(nn.Module):
    """TarMAC differentiable communication module.

    Reference: Dasari et al., "Learning to Act from Actionless Video
    through Spatial-Temporal Correspondences" ... actually the TarMAC
    paper is "Cooperative Multi-Agent Communication" by Foerster et al.

    Per the REVISION_PLAN §6:
      - Each sender encodes its local feature into a message (key) and
        gates it with a sigmoid so messages can be "off".
      - Each receiver encodes its own local feature into a query.
      - Attention scores = softmax(q . k^T / sqrt(d)).
      - Attended messages = gate * message, weighted by attention.

    Args:
        in_dim: per-agent feature dimension before communication.
        message_dim: length of each agent's message vector.
    """

    def __init__(self, in_dim, message_dim, hidden_dim=COMM_HIDDEN_DIM):
        super().__init__()
        self.message_dim = message_dim
        # sender: feature -> message
        self.key = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, message_dim),
        )
        # receiver: feature -> query
        self.query = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, message_dim),
        )
        # per-agent gate (sigmoid) controls message strength
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, message_dim),
            nn.Sigmoid(),
        )
        self.scale = math.sqrt(message_dim)

    def forward(self, features):
        """
        Args:
            features: [batch, n_agents, in_dim]
        Returns:
            aggregated: [batch, n_agents, message_dim]
            messages_gated: [batch, n_agents, message_dim]  (for diagnostics)
            attention: [batch, n_agents, n_agents]
        """
        keys = self.key(features)  # [B, n, msg]
        queries = self.query(features)  # [B, n, msg]
        gates = self.gate(features)  # [B, n, msg]

        # attention scores: [B, n, n]  (receiver x sender)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale

        # Mask out self-attention to match TarMAC research (sum over j != i)
        mask = torch.eye(scores.size(1), device=scores.device).bool()
        scores.masked_fill_(mask.unsqueeze(0), -1e9)

        attention = torch.softmax(scores, dim=-1)

        messages_gated = gates * keys  # [B, n, msg]
        # each receiver aggregates across all other senders (excluding self)
        aggregated = torch.bmm(attention, messages_gated)  # [B, n, msg]

        return aggregated, messages_gated, attention


class CommAgent(nn.Module):
    """Actor-critic with TarMAC learned communication (Phase B).

    Architecture:
      1. Per-agent encoder: LOCAL_INPUT_DIM -> COMM_HIDDEN_DIM
      2. TarMACComm: message passing among agents
      3. Concatenate local features + attended messages -> COMM_INPUT_DIM
      4. Separate actor and critic heads

    For centralized critics (MAPPO), the critic still uses ``env.state()``
    and ignores communication.
    """

    def __init__(self, state_dim=None, hidden_dim=HIDDEN_DIM, centralized=False):
        super().__init__()
        self.centralized = centralized

        # per-agent encoder (projects obs+role into hidden for comm)
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
        )

        # TarMAC communication
        self.comm = TarMACComm(hidden_dim, COMM_MESSAGE_DIM)

        # actor: consumes COMM_INPUT_DIM (obs+role + message)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(COMM_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )

        if centralized:
            assert state_dim is not None, "MAPPO CommAgent needs state_dim"
            self.critic = nn.Sequential(
                layer_init(nn.Linear(state_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )
        else:
            self.critic = nn.Sequential(
                layer_init(nn.Linear(COMM_INPUT_DIM, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, 1), std=1.0),
            )

        # Store last attention/messages for diagnostics
        self._last_attention = None
        self._last_messages_gated = None

    def _joint_features(self, obs_list, role_list):
        """
        Args:
            obs_list: list of [B, C, H, W] per agent
            role_list: list of [B, N_AGENTS] per agent
        Returns:
            features: [B, N_AGENTS, LOCAL_INPUT_DIM]
        """
        # stack per-agent obs: [n_agents, B, C, H, W] -> [B, n_agents, C*H*W]
        flat = torch.stack(
            [torch.flatten(o, start_dim=1).float() for o in obs_list], dim=1
        )  # [B, n, 25]
        roles = torch.stack([r.float() for r in role_list], dim=1)  # [B, n, 4]
        return torch.cat([flat, roles], dim=-1)  # [B, n, 29]

    def get_value(self, obs_list, role_list, state=None):
        features = self._joint_features(obs_list, role_list)  # [B, n, 29]
        encoded = self.encoder(features)  # [B, n, hidden]
        aggregated, _, _ = self.comm(encoded)  # [B, n, msg]

        if self.centralized and state is not None:
            return self.critic(state)

        # per-agent critic: combine local features + attended message
        B, n, _ = features.shape
        x = torch.cat([features, aggregated], dim=-1)  # [B, n, 61]
        # flatten for critic: [B*n, 61]
        return self.critic(x.view(B * n, -1)).view(B, n)

    def get_influence_matrix(self, obs_list, role_list):
        """Calculates the Causal Influence Routing (CIR) matrix via Counterfactual Message Ablation."""
        features = self._joint_features(obs_list, role_list)
        encoded = self.encoder(features)
        aggregated, messages_gated, attention = self.comm(encoded)

        B, n, _ = features.shape
        x_base = torch.cat([features, aggregated], dim=-1)
        base_values = self.critic(x_base.view(B * n, -1)).view(B, n)

        influence_matrix = torch.zeros((B, n, n), device=features.device)

        for sender in range(n):
            alt_aggregated = aggregated.clone()
            msg_s = messages_gated[:, sender : sender + 1, :]  # [B, 1, dim]
            att_s = attention[:, :, sender : sender + 1]  # [B, n, 1]

            alt_aggregated -= att_s * msg_s

            x_alt = torch.cat([features, alt_aggregated], dim=-1)
            alt_values = self.critic(x_alt.view(B * n, -1)).view(B, n)

            influence_matrix[:, :, sender] = torch.abs(base_values - alt_values)

        mask = torch.eye(n, device=features.device).bool()
        influence_matrix.masked_fill_(mask, 0.0)

        row_sums = influence_matrix.sum(dim=-1, keepdim=True) + 1e-8
        routing_weights = influence_matrix / row_sums

        return routing_weights

    def get_action_and_value(
        self, obs_list, role_list, action_mask_list, state=None, action=None
    ):
        """
        Args:
            obs_list: list of [B, C, H, W] for n_agents
            role_list: list of [B, N_AGENTS] for n_agents
            action_mask_list: list of [B, ACTION_DIM] for n_agents
            state: [B, state_dim] for MAPPO centralized critic (optional)
            action: [B, n_agents] to evaluate, or None to sample
        Returns:
            actions, log_probs, entropy, values  (all appropriately shaped)
        """
        features = self._joint_features(obs_list, role_list)  # [B, n, 29]
        encoded = self.encoder(features)  # [B, n, hidden]
        aggregated, messages_gated, attention = self.comm(encoded)

        # store for diagnostics
        self._last_attention = attention.detach()
        self._last_messages_gated = messages_gated.detach()

        B, n, _ = features.shape
        agent_input = torch.cat([features, aggregated], dim=-1)  # [B, n, 61]

        # flatten to [B*n, 61] for per-agent actor/critic
        flat_input = agent_input.view(B * n, -1)
        logits = self.actor(flat_input)  # [B*n, ACTION_DIM]

        # flatten masks
        flat_mask = torch.cat(action_mask_list, dim=0)  # [B*n, ACTION_DIM]
        masked_logits = logits.masked_fill(flat_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)

        actions = probs.sample().view(B, n) if action is None else action
        flat_actions = actions.view(B * n)
        log_probs = probs.log_prob(flat_actions).view(B, n)
        entropy = probs.entropy().view(B, n)

        # values
        if self.centralized and state is not None:
            values = self.critic(state).expand(B, n)  # shared value for all agents
        else:
            values = self.critic(flat_input).view(B, n)

        return actions, log_probs, entropy, values


def construct_other_actions_onehot(actions_tensor, focal_agent_idx):
    """
    Args:
        actions_tensor: [N_AGENTS, B] integer actions
        focal_agent_idx: int index of focal agent i
    Returns:
        [B, (N_AGENTS - 1) * ACTION_DIM] float one-hot encoding of other agents' actions
    """
    other_list = []
    for k in range(N_AGENTS):
        if k != focal_agent_idx:
            oh = torch.nn.functional.one_hot(
                actions_tensor[k], num_classes=ACTION_DIM
            ).float()
            other_list.append(oh)
    return torch.cat(other_list, dim=-1)


# ──────────────────────────────────────────────────────────────────────
# COMA (Counterfactual Multi-Agent Policy Gradients)
# ──────────────────────────────────────────────────────────────────────


class ComaCritic(nn.Module):
    """COMA Centralized Critic.

    Inputs:
        state: Global state vector [B, state_dim]
        other_actions_onehot: One-hot joint actions of all OTHER agents [B, (N_AGENTS-1) * ACTION_DIM]
    Returns:
        Q-values for ALL possible actions of the focal agent [B, ACTION_DIM]
    """

    def __init__(
        self, state_dim, n_agents=N_AGENTS, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM
    ):
        super().__init__()
        other_actions_dim = (n_agents - 1) * action_dim
        in_dim = state_dim + other_actions_dim

        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=1.0),
        )

    def forward(self, state, other_actions_onehot):
        x = torch.cat([state, other_actions_onehot], dim=-1)
        return self.net(x)


class ComaAgent(nn.Module):
    """COMA Agent: decentralized actor (local obs + role) + centralized critic (state + other actions)."""

    def __init__(
        self, state_dim, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_agents=N_AGENTS
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.critic = ComaCritic(
            state_dim=state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

    def get_action_and_probs(self, obs, role_id, action_mask, action=None):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), role_id.float()), dim=1)
        logits = self.actor(x)
        masked_logits = logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.probs, probs.entropy()

    def get_value(self, state, other_actions_onehot):
        return self.critic(state, other_actions_onehot)

    def get_action_and_value(
        self, obs, role_id, action_mask, state, other_actions_onehot, action=None
    ):
        act, log_prob, probs, entropy = self.get_action_and_probs(
            obs, role_id, action_mask, action=action
        )
        q_vals = self.get_value(state, other_actions_onehot)
        return act, log_prob, entropy, q_vals

    def get_influence_matrix(self, state, actions_tensor):
        """Calculates Non-Communicating Causal Influence Routing (CIR) matrix via Counterfactual Action Ablation for COMA."""
        B = state.shape[0]
        influence_matrix = torch.zeros((B, N_AGENTS, N_AGENTS), device=state.device)

        for i in range(N_AGENTS):
            other_oh_base = construct_other_actions_onehot(actions_tensor, i)
            q_base_all = self.critic(state, other_oh_base)
            q_base = q_base_all.gather(1, actions_tensor[i].unsqueeze(1)).squeeze(1)

            for j in range(N_AGENTS):
                if j == i:
                    continue
                sender_idx = j if j < i else j - 1
                start_col = sender_idx * ACTION_DIM
                end_col = start_col + ACTION_DIM

                other_oh_alt = other_oh_base.clone()
                other_oh_alt[:, start_col:end_col] = 0.0

                q_alt_all = self.critic(state, other_oh_alt)
                q_alt = q_alt_all.gather(1, actions_tensor[i].unsqueeze(1)).squeeze(1)

                influence_matrix[:, i, j] = torch.abs(q_base - q_alt)

        mask = torch.eye(N_AGENTS, device=state.device).bool()
        influence_matrix.masked_fill_(mask, 0.0)

        row_sums = influence_matrix.sum(dim=-1, keepdim=True) + 1e-8
        return influence_matrix / row_sums


# ──────────────────────────────────────────────────────────────────────
# Hierarchical Agents (CHARM, MAHIRO, ROMA, LRS)
# ──────────────────────────────────────────────────────────────────────


class ContinuousHierarchicalAgent(nn.Module):
    """
    Used by CHARM and MAHIRO.
    Manager outputs a 2D continuous goal (x, y).
    Worker outputs a discrete action conditioned on the goal.
    """

    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        # Manager networks
        self.manager_critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.manager_actor_mean = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 2), std=0.01),
        )
        self.manager_actor_logstd = nn.Parameter(torch.zeros(1, 2))

        # Worker networks
        self.worker_critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4 + 2, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.worker_actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM + 2, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )

    def get_manager_value(self, state, role_id):
        return self.manager_critic(torch.cat([state, role_id], dim=1))

    def get_manager_action_and_value(self, obs, state, role_id, action=None):
        x = torch.cat([torch.flatten(obs, start_dim=1), role_id], dim=1)
        action_mean = self.manager_actor_mean(x)
        action_logstd = self.manager_actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.manager_critic(torch.cat([state, role_id], dim=1)),
        )

    def get_worker_value(self, state, role_id, goal):
        return self.worker_critic(torch.cat([state, role_id, goal], dim=1))

    def get_worker_action_and_value(
        self, obs, state, role_id, goal, action_mask, action=None
    ):
        x = torch.cat([torch.flatten(obs, start_dim=1), role_id, goal], dim=1)
        logits = self.worker_actor(x)
        masked_logits = logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.worker_critic(torch.cat([state, role_id, goal], dim=1)),
        )


class DiscreteHierarchicalAgent(nn.Module):
    """
    Used by ROMA and LRS.
    Manager outputs a discrete role/phase (1 of K).
    Worker outputs a discrete action conditioned on the role/phase.
    """

    def __init__(self, state_dim, num_roles=4, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.num_roles = num_roles

        # Manager networks
        self.manager_critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.manager_actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_roles), std=0.01),
        )

        # Worker networks
        self.worker_critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4 + num_roles, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.worker_actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM + num_roles, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )

    def get_manager_value(self, state, role_id):
        return self.manager_critic(torch.cat([state, role_id], dim=1))

    def get_manager_action_and_value(self, obs, state, role_id, action=None):
        x = torch.cat([torch.flatten(obs, start_dim=1), role_id], dim=1)
        logits = self.manager_actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.manager_critic(torch.cat([state, role_id], dim=1)),
        )

    def get_worker_value(self, state, role_id, goal_discrete):
        goal_onehot = torch.nn.functional.one_hot(
            goal_discrete, num_classes=self.num_roles
        ).float()
        return self.worker_critic(torch.cat([state, role_id, goal_onehot], dim=1))

    def get_worker_action_and_value(
        self, obs, state, role_id, goal_discrete, action_mask, action=None
    ):
        goal_onehot = torch.nn.functional.one_hot(
            goal_discrete, num_classes=self.num_roles
        ).float()
        x = torch.cat([torch.flatten(obs, start_dim=1), role_id, goal_onehot], dim=1)
        logits = self.worker_actor(x)
        masked_logits = logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.worker_critic(torch.cat([state, role_id, goal_onehot], dim=1)),
        )


# ──────────────────────────────────────────────────────────────────────
# SCOPE Agent (Dynamic Skill-Pool Co-op)
# ──────────────────────────────────────────────────────────────────────


class ScopeExpert(nn.Module):
    def __init__(self, state_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, ACTION_DIM), std=0.01),
        )
        # Critic predicts value of state given this expert is acting
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )


class ScopeAgent(nn.Module):
    """
    SCOPE: Scalable Cooperative Option-Pool Evolution.
    Instead of a Manager, Experts vote via their Critic values (Bottom-Up Routing).
    """

    def __init__(self, state_dim, max_experts=10, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.max_experts = max_experts
        self.experts = nn.ModuleList(
            [ScopeExpert(state_dim, hidden_dim) for _ in range(max_experts)]
        )

    def get_action_and_value(
        self,
        obs,
        state,
        role_id,
        action_mask,
        active_experts,
        action=None,
        expert_idx=None,
    ):
        """
        active_experts: integer representing how many experts are currently spawned.
        Returns: action, log_prob, entropy, value, chosen_expert
        """
        B = obs.shape[0]
        x_actor = torch.cat(
            [torch.flatten(obs, start_dim=1).float(), role_id.float()], dim=1
        )
        x_critic = torch.cat([state, role_id.float()], dim=1)

        # Get outputs from all active experts
        all_logits = []
        all_values = []
        for k in range(active_experts):
            all_logits.append(self.experts[k].actor(x_actor))
            all_values.append(self.experts[k].critic(x_critic))

        # [B, active_experts, ACTION_DIM]
        logits_stack = torch.stack(all_logits, dim=1)
        # [B, active_experts]
        values_stack = torch.stack(all_values, dim=1).squeeze(-1)

        if expert_idx is None:
            # Bottom-Up Routing: select expert with max value
            chosen_expert = torch.argmax(values_stack, dim=1)  # [B]
        else:
            chosen_expert = expert_idx

        # Gather chosen logits and values
        batch_indices = torch.arange(B, device=obs.device)
        chosen_logits = logits_stack[batch_indices, chosen_expert]  # [B, ACTION_DIM]
        chosen_values = values_stack[batch_indices, chosen_expert].unsqueeze(
            -1
        )  # [B, 1]

        masked_logits = chosen_logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            chosen_values,
            chosen_expert,
        )


class ScopeTopDownAgent(nn.Module):
    """
    SCOPE Ablation: Top-Down Router.
    Uses a standard Manager network to pick the expert, rather than bottom-up voting.
    """

    def __init__(self, state_dim, max_experts=10, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.max_experts = max_experts

        # Manager network
        self.manager_actor = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, max_experts), std=0.01),
        )
        self.manager_critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        self.experts = nn.ModuleList(
            [ScopeExpert(state_dim, hidden_dim) for _ in range(max_experts)]
        )

    def get_action_and_value(
        self,
        obs,
        state,
        role_id,
        action_mask,
        active_experts,
        action=None,
        expert_idx=None,
    ):
        B = obs.shape[0]
        x_actor = torch.cat(
            [torch.flatten(obs, start_dim=1).float(), role_id.float()], dim=1
        )
        x_critic = torch.cat([state, role_id.float()], dim=1)

        # Top-Down Manager routing
        manager_logits = self.manager_actor(x_critic)  # [B, max_experts]

        # Mask out inactive experts
        mask = torch.ones(self.max_experts, device=obs.device).bool()
        mask[:active_experts] = False
        manager_logits = manager_logits.masked_fill(mask, -1e9)

        manager_probs = Categorical(logits=manager_logits)

        chosen_expert = manager_probs.sample() if expert_idx is None else expert_idx

        manager_logprob = manager_probs.log_prob(chosen_expert)
        manager_entropy = manager_probs.entropy()
        manager_value = self.manager_critic(x_critic)

        # Get outputs from all active experts
        all_logits = []
        for k in range(active_experts):
            all_logits.append(self.experts[k].actor(x_actor))

        logits_stack = torch.stack(all_logits, dim=1)

        # Gather chosen expert's logits
        batch_indices = torch.arange(B, device=obs.device)
        chosen_logits = logits_stack[batch_indices, chosen_expert]

        masked_logits = chosen_logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)

        if action is None:
            action = probs.sample()

        # Total logprob is logprob of manager + logprob of expert
        total_logprob = manager_logprob + probs.log_prob(action)
        total_entropy = manager_entropy + probs.entropy()

        return action, total_logprob, total_entropy, manager_value, chosen_expert
