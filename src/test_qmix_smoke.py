"""
QMIX logic smoke test (no training loop).

Validates the replay buffer, masked per-agent Q values, the monotonic
mixing network forward/backward pass, and one full TD gradient step.
Run with:  uv run python src/test_qmix_smoke.py
"""

import numpy as np
import torch

from env import HeistEnv
from model import QNetwork, QMixMixing
from train_qmix import ReplayBuffer, select_actions


def main():
    env = HeistEnv({
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
    })
    obs, _ = env.reset(seed=0)
    state_dim = env.state().shape[0]

    q = {a: QNetwork() for a in env.agents}
    tq = {a: QNetwork() for a in env.agents}
    mixing = QMixMixing(len(env.agents), state_dim)
    tmix = QMixMixing(len(env.agents), state_dim)
    tmix.load_state_dict(mixing.state_dict())

    # --- action selection respects the mask ---
    acts = select_actions(env, obs, q, epsilon=1.0, device="cpu")
    print("random joint action:", acts)

    # --- fill replay buffer ---
    buf = ReplayBuffer(1000, (5, 5), state_dim)
    for i in range(50):
        acts = select_actions(env, obs, q, epsilon=0.5, device="cpu")
        nobs, rewards, terms, truncs, _ = env.step(acts)
        done = bool(any(terms.values()) or any(truncs.values()))
        buf.push(obs, acts, rewards, any(terms.values()), any(truncs.values()),
                 env.state(), nobs, env.state())
        obs = nobs
        if done:
            obs, _ = env.reset(seed=i)
    print("buffer size:", buf.size)

    # --- one gradient step ---
    params = list(mixing.parameters())
    for a in env.agents:
        params += list(q[a].parameters())
    opt = torch.optim.Adam(params, lr=1e-4)

    batch = buf.sample(32)
    qv = []
    for a in env.agents:
        qa = q[a](batch["obs"][a], batch["role"][a]) \
              .gather(1, batch["actions"][a].unsqueeze(1)).squeeze(1)
        qv.append(qa)
    qv = torch.stack(qv, dim=1)
    q_tot = mixing(qv, batch["states"])

    with torch.no_grad():
        tqv = []
        for a in env.agents:
            nq = tq[a](batch["next_obs"][a], batch["role"][a])
            m = torch.where(batch["next_mask"][a] == 1, nq, torch.full_like(nq, -1e9))
            tqv.append(m.max(dim=1).values)
        tqv = torch.stack(tqv, dim=1)
        q_tot_target = tmix(tqv, batch["next_states"])
        rewards = torch.stack([batch["rewards"][a] for a in env.agents], dim=1).mean(dim=1)
        # REV-3: bootstrap uses terminations only; truncations keep the target
        # from the stored (terminal) next state.
        y = rewards + 0.99 * (1 - batch["terminated"]) * q_tot_target

    loss = torch.nn.functional.mse_loss(q_tot, y)
    loss.backward()
    opt.step()
    print(f"q_tot shape={tuple(q_tot.shape)} target shape={tuple(y.shape)}")
    print(f"loss: {loss.item():.4f}")
    print("QMIX gradient step OK")


if __name__ == "__main__":
    main()
