import torch
import torch.nn.functional as F

from env import HeistEnv
from model import ACTION_DIM, N_AGENTS, ComaAgent, ComaCritic
from ppo_utils import compute_counterfactual_advantage


def test_coma_critic_shape():
    state_dim = 135
    critic = ComaCritic(state_dim=state_dim, n_agents=4, action_dim=6)

    state = torch.randn(2, state_dim)
    other_actions = torch.zeros(2, (N_AGENTS - 1) * ACTION_DIM)

    q_vals = critic(state, other_actions)
    assert q_vals.shape == (2, 6), f"Expected Q-values shape (2, 6), got {q_vals.shape}"


def test_coma_forward_and_advantage():
    env = HeistEnv({"map_size": (11, 11), "guard_count": 0})
    obs, _ = env.reset(seed=42)
    state_dim = env.state().shape[0]

    agent = ComaAgent(state_dim=state_dim)

    # Fake batch of size 2
    state = torch.as_tensor(env.state()).unsqueeze(0).repeat(2, 1).float()
    other_actions = torch.zeros(2, (N_AGENTS - 1) * ACTION_DIM).float()

    q_vals = agent.critic(state, other_actions)
    assert q_vals.shape == (2, 6), f"Expected Q-values shape (2, 6), got {q_vals.shape}"

    scout_obs = (
        torch.as_tensor(obs["scout"]["observation"]).unsqueeze(0).repeat(2, 1, 1)
    )
    scout_role = torch.as_tensor(obs["scout"]["role_id"]).unsqueeze(0).repeat(2, 1)
    scout_mask = torch.as_tensor(obs["scout"]["action_mask"]).unsqueeze(0).repeat(2, 1)

    act, log_p, probs, ent = agent.get_action_and_probs(
        scout_obs, scout_role, scout_mask
    )
    assert act.shape == (2,)
    assert log_p.shape == (2,)
    assert probs.shape == (2, 6)
    assert ent.shape == (2,)

    adv, q_taken = compute_counterfactual_advantage(probs, q_vals, act, scout_mask)
    assert adv.shape == (2,)
    assert q_taken.shape == (2,)

    # Test gradient flow
    target = torch.randn(2)
    critic_loss = F.mse_loss(q_taken, target)
    actor_loss = -(log_p * adv.detach()).mean()
    loss = actor_loss + critic_loss

    loss.backward()
    for name, param in agent.named_parameters():
        assert param.grad is not None, f"Gradient for {name} should not be None"
