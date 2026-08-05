# COMA (Counterfactual Multi-Agent Policy Gradients) Implementation Guide

## 1. Overview & Theoretical Objective
Implement the COMA algorithm (Foerster et al., AAAI 2018) for the HEIST benchmark. COMA addresses multi-agent credit assignment using a centralized critic $Q(s, \mathbf{a}_{-i}, a_i)$ that outputs Q-values for all possible actions of agent $i$ given global state $s$ and the joint actions of all other agents $\mathbf{a}_{-i}$.

### Counterfactual Advantage Formula
For agent $i$, the counterfactual baseline marginalizes over agent $i$'s legal action space while holding all other agents' actions $\mathbf{a}_{-i}$ fixed:
$$A_i(s, \mathbf{a}) = Q_i(s, \mathbf{a}_{-i}, a_i) - \sum_{a_i'} \pi_i(a_i' \mid o_i) Q_i(s, \mathbf{a}_{-i}, a_i')$$

---

## 2. Model Architecture (`src/model.py`)

### A. Centralized COMA Critic (`ComaCritic`)
The critic evaluates Q-values for all $\vert{}A_i\vert{}$ actions of agent $i$ given global state $s$ and the one-hot encoded joint actions of all other agents $\mathbf{a}_{-i}$.

```python
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, std=None, bias_const=0.0):
    if std is None:
        std = (2.0 ** 0.5)
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ComaCritic(nn.Module):
    """COMA Centralized Critic.
    
    Inputs:
        state: Global state vector [B, state_dim]
        other_actions_onehot: One-hot joint actions of all OTHER agents [B, (N_AGENTS-1) * ACTION_DIM]
    Returns:
        Q-values for ALL possible actions of the focal agent [B, ACTION_DIM]
    """
    def __init__(self, state_dim, n_agents=4, action_dim=6, hidden_dim=64):
        super().__init__()
        other_actions_dim = (n_agents - 1) * action_dim
        in_dim = state_dim + other_actions_dim
        
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=1.0)
        )

    def forward(self, state, other_actions_onehot):
        x = torch.cat([state, other_actions_onehot], dim=-1)
        return self.net(x)

```

### B. COMA Agent Class (`ComaAgent`)

Wraps the decentralized actor and centralized critic.

```python
class ComaAgent(nn.Module):
    def __init__(self, state_dim, action_dim=6, hidden_dim=64):
        super().__init__()
        # Decentralized Actor (consumes local obs + role one-hot)
        from model import LOCAL_INPUT_DIM
        self.actor = nn.Sequential(
            layer_init(nn.Linear(LOCAL_INPUT_DIM, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.critic = ComaCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    def get_action_and_probs(self, obs, role_id, action_mask, action=None):
        x = torch.cat((torch.flatten(obs, start_dim=1).float(), role_id.float()), dim=1)
        logits = self.actor(x)
        masked_logits = logits.masked_fill(action_mask != 1, -1e9)
        probs = Categorical(logits=masked_logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.probs, probs.entropy()

```

---

## 3. Counterfactual Advantage Calculation Logic

Implement this function to compute the counterfactual baseline and advantage for a specific agent $i$:

```python
def compute_counterfactual_advantage(policy_probs, q_values_all, taken_actions, action_mask):
    """
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
    masked_q = torch.where(action_mask == 1, q_values_all, torch.zeros_like(q_values_all))
    
    # 2. Extract Q-value of taken action
    q_taken = q_values_all.gather(1, taken_actions.unsqueeze(1)).squeeze(1)
    
    # 3. Compute Counterfactual Baseline: \sum_{a_i'} pi_i(a_i' | o_i) * Q_i(s, a_{-i}, a_i')
    baseline = torch.sum(policy_probs * masked_q, dim=-1)
    
    # 4. Counterfactual Advantage
    advantage = q_taken - baseline
    return advantage, q_taken

```

---

## 4. Trainer Outline (`src/train_coma.py`)

### Step-by-Step Training Algorithm:

1. **Rollout Collection:**
* At each environment step $t$, collect observations $o_t$, global state $s_t$, executed actions $\mathbf{a}_t$, action masks, and rewards $r_t$.


2. **Construct One-Hot Joint Actions for Critic:**
* For agent $i$, construct $\mathbf{a}_{-i}$ by concatenating the one-hot vectors of all agents except $i$.


3. **Compute TD Targets for Critic:**
* $y_t = r_t + \gamma (1 - \text{terminated}_t) Q_{\text{target}}(s_{t+1}, \mathbf{a}_{-i, t+1}, a_{i, t+1})$


4. **Compute Counterfactual Advantages:**
* Pass $s_t$ and $\mathbf{a}_{-i, t}$ through `critic` to get $Q_i(s_t, \mathbf{a}_{-i, t}, \cdot)$.
* Use `compute_counterfactual_advantage` to get $A_i(s_t, \mathbf{a}_t)$.


5. **Loss Computation:**
* **Actor Loss:** $\mathcal{L}_{\text{actor}} = - \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{i=1}^N \log \pi_i(a_{i,b} \mid o_{i,b}) \cdot \text{detach}(A_{i,b})$
* **Critic Loss:** $\mathcal{L}_{\text{critic}} = \frac{1}{B \cdot N} \sum_{b=1}^B \sum_{i=1}^N \left( Q_i(s_{b}, \mathbf{a}_{-i,b}, a_{i,b}) - y_{i,b} \right)^2$



---

## 5. Verification & Testing (`tests/test_coma_smoke.py`)

Create a unit test to confirm shape consistency and gradient flow:

```python
import torch
from env import AGENTS, HeistEnv
from model import ComaAgent

def test_coma_forward_and_advantage():
    env = HeistEnv({"map_size": (11, 11), "guard_count": 0})
    obs, _ = env.reset(seed=42)
    state_dim = env.state().shape[0]
    
    agent = ComaAgent(state_dim=state_dim)
    
    # Fake batch of size 2
    state = torch.as_tensor(env.state()).unsqueeze(0).repeat(2, 1).float()
    other_actions = torch.zeros(2, 3 * 6).float() # 3 other agents * 6 actions
    
    q_vals = agent.critic(state, other_actions)
    assert q_vals.shape == (2, 6), f"Expected Q-values shape (2, 6), got {q_vals.shape}"

```