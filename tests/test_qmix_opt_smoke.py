"""
Unit test for optimized QMIX action selection and replay logic.

Validates that:
1. Batched GPU tensor action selection produces valid legal actions.
2. Action masks are strictly enforced under greedy (epsilon=0.0) selection.
3. Replay sampling and QMIX joint optimization step execute without error.

Run with: uv run pytest src/test_qmix_opt_smoke.py
"""

import torch

from constants import OBSERVATION_SIZE
from env import HeistEnv
from model import QMixMixing, QNetwork
from train_qmix import ReplayBuffer, select_actions


def test_select_actions_masking():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 60,
        }
    )
    obs, _ = env.reset(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q_nets = {a: QNetwork().to(device) for a in env.agents}

    # Greedy action selection (epsilon = 0.0)
    actions = select_actions(env, obs, q_nets, epsilon=0.0, device=device)
    for a in env.agents:
        mask = obs[a]["action_mask"]
        act = actions[a]
        assert mask[act] == 1, f"Agent {a} selected illegal action {act}"


def test_qmix_optimization_step():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 60,
        }
    )
    obs, _ = env.reset(seed=123)
    state_dim = env.state().shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_nets = {a: QNetwork().to(device) for a in env.agents}
    target_nets = {a: QNetwork().to(device) for a in env.agents}
    mixing = QMixMixing(len(env.agents), state_dim).to(device)
    target_mixing = QMixMixing(len(env.agents), state_dim).to(device)

    buffer = ReplayBuffer(100, OBSERVATION_SIZE, state_dim)
    for _ in range(10):
        actions = select_actions(env, obs, q_nets, epsilon=0.2, device=device)
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        done = bool(any(terms.values()) or any(truncs.values()))
        buffer.push(
            obs,
            actions,
            rewards,
            any(terms.values()),
            any(truncs.values()),
            env.state(),
            next_obs,
            env.state(),
        )
        obs = next_obs
        if done:
            obs, _ = env.reset()

    assert buffer.size == 10

    batch = buffer.sample(8)
    for key in batch:
        if isinstance(batch[key], dict):
            batch[key] = {a: t.to(device) for a, t in batch[key].items()}
        else:
            batch[key] = batch[key].to(device)

    # Online joint Q
    q_vals = []
    for a in env.agents:
        q = q_nets[a](batch["obs"][a], batch["role"][a])
        q = q.gather(1, batch["actions"][a].unsqueeze(1)).squeeze(1)
        q_vals.append(q)
    q_vals = torch.stack(q_vals, dim=1)
    q_tot = mixing(q_vals, batch["states"])

    # Target joint Q
    with torch.no_grad():
        tq = []
        for a in env.agents:
            q_next = target_nets[a](batch["next_obs"][a], batch["role"][a])
            masked = torch.where(
                batch["next_mask"][a] == 1, q_next, torch.full_like(q_next, -1e9)
            )
            tq.append(masked.max(dim=1).values)
        tq = torch.stack(tq, dim=1)
        q_tot_target = target_mixing(tq, batch["next_states"])
        r = torch.stack([batch["rewards"][a] for a in env.agents], dim=1).mean(dim=1)
        y = r + 0.99 * (1 - batch["terminated"]) * q_tot_target

    loss = torch.nn.functional.mse_loss(q_tot, y)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
