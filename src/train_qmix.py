"""
QMIX trainer for HEIST.

PLAN.md Baseline 3 (value-decomposition): per-agent deep Q networks whose
individual utilities are combined by a monotonic mixing network conditioned
on the global state.  QMIX's monotonicity constraint (dQ_tot/dQ_i >= 0) is
exactly the factorization assumption that PLAN.md predicts breaks under
Causal Credit Dilution, so this is the key diagnostic baseline.

This implementation follows the standard offline QMIX recipe:
  * per-agent Q networks (own + target)
  * monotonic hypernetwork mixer (own + target)
  * replay buffer of joint transitions
  * epsilon-greedy exploration with linear annealing
  * polyak target updates

Example:
    uv run python src/train_qmix.py --total-steps 2_000_000
"""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from env import HeistEnv, AGENTS, parse_env_config
from constants import (
    OBSERVATION_SIZE, ACTION_SPACE_SIZE as ACTION_DIM, GLOBAL_STATE_DIM, N_AGENTS,
)
from model import QNetwork, QMixMixing


@dataclass
class Args:
    exp_name: str = "qmix"
    seed: int = 0
    cuda: bool = True
    total_steps: int = 2_000_000
    lr: float = 1e-4
    buffer_size: int = 100_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005          # polyak target update
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal_steps: int = 500_000
    learning_starts: int = 10_000
    train_freq: int = 1         # gradient steps per env step
    target_update_freq: int = 200
    eval_every: int = 5_000
    eval_episodes: int = 20
    env_config: str = ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--total-steps", type=int, default=Args.total_steps)
    p.add_argument("--lr", type=float, default=Args.lr)
    p.add_argument("--buffer-size", type=int, default=Args.buffer_size)
    p.add_argument("--batch-size", type=int, default=Args.batch_size)
    p.add_argument("--gamma", type=float, default=Args.gamma)
    p.add_argument("--tau", type=float, default=Args.tau)
    p.add_argument("--eps-start", type=float, default=Args.eps_start)
    p.add_argument("--eps-end", type=float, default=Args.eps_end)
    p.add_argument("--eps-anneal-steps", type=int, default=Args.eps_anneal_steps)
    p.add_argument("--learning-starts", type=int, default=Args.learning_starts)
    p.add_argument("--train-freq", type=int, default=Args.train_freq)
    p.add_argument("--target-update-freq", type=int, default=Args.target_update_freq)
    p.add_argument("--eval-every", type=int, default=Args.eval_every)
    p.add_argument("--eval-episodes", type=int, default=Args.eval_episodes)
    p.add_argument("--env-config", type=str, default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Replay buffer of joint transitions
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, state_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = {a: np.zeros((capacity, *obs_shape), dtype=np.int32) for a in AGENTS}
        self.gs = {a: np.zeros((capacity, GLOBAL_STATE_DIM), dtype=np.int32) for a in AGENTS}
        self.mask = {a: np.zeros((capacity, ACTION_DIM), dtype=np.int8) for a in AGENTS}
        self.role = {a: np.zeros((capacity, N_AGENTS), dtype=np.int8) for a in AGENTS}
        self.actions = {a: np.zeros(capacity, dtype=np.int64) for a in AGENTS}
        self.rewards = {a: np.zeros(capacity, dtype=np.float32) for a in AGENTS}
        self.next_obs = {a: np.zeros((capacity, *obs_shape), dtype=np.int32) for a in AGENTS}
        self.next_gs = {a: np.zeros((capacity, GLOBAL_STATE_DIM), dtype=np.int32) for a in AGENTS}
        self.next_mask = {a: np.zeros((capacity, ACTION_DIM), dtype=np.int8) for a in AGENTS}
        # REV-3: store terminations and truncations separately (QMIX bootstraps
        # the target on truncation from the stored terminal next state).
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def push(self, obs, actions, rewards, terminations, truncations, states, next_obs, next_states):
        for a in AGENTS:
            self.obs[a][self.pos] = obs[a]["observation"]
            self.gs[a][self.pos] = obs[a]["global_state"]
            self.mask[a][self.pos] = obs[a]["action_mask"]
            self.role[a][self.pos] = obs[a]["role_id"]
            self.actions[a][self.pos] = actions[a]
            self.rewards[a][self.pos] = rewards[a]
            self.next_obs[a][self.pos] = next_obs[a]["observation"]
            self.next_gs[a][self.pos] = next_obs[a]["global_state"]
            self.next_mask[a][self.pos] = next_obs[a]["action_mask"]
        self.terminated[self.pos] = float(terminations)
        self.truncated[self.pos] = float(truncations)
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": {a: torch.tensor(self.obs[a][idx], dtype=torch.float32) for a in AGENTS},
            "gs": {a: torch.tensor(self.gs[a][idx], dtype=torch.float32) for a in AGENTS},
            "mask": {a: torch.tensor(self.mask[a][idx], dtype=torch.long) for a in AGENTS},
            "role": {a: torch.tensor(self.role[a][idx], dtype=torch.float32) for a in AGENTS},
            "actions": {a: torch.tensor(self.actions[a][idx], dtype=torch.long) for a in AGENTS},
            "rewards": {a: torch.tensor(self.rewards[a][idx], dtype=torch.float32) for a in AGENTS},
            "next_obs": {a: torch.tensor(self.next_obs[a][idx], dtype=torch.float32) for a in AGENTS},
            "next_gs": {a: torch.tensor(self.next_gs[a][idx], dtype=torch.float32) for a in AGENTS},
            "next_mask": {a: torch.tensor(self.next_mask[a][idx], dtype=torch.long) for a in AGENTS},
            "terminated": torch.tensor(self.terminated[idx], dtype=torch.float32),
            "truncated": torch.tensor(self.truncated[idx], dtype=torch.float32),
            "states": torch.tensor(self.states[idx], dtype=torch.float32),
            "next_states": torch.tensor(self.next_states[idx], dtype=torch.float32),
        }
        return batch


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------
def select_actions(env, obs, q_nets, epsilon, device):
    """Epsilon-greedy joint action selection respecting action masks."""
    actions = {}
    for a in AGENTS:
        mask = obs[a]["action_mask"]
        legal = np.argwhere(mask == 1).ravel()
        if len(legal) == 0:
            actions[a] = int(np.random.randint(ACTION_DIM))
            continue
        if np.random.rand() < epsilon:
            actions[a] = int(np.random.choice(legal))
            continue
        obs_t = torch.tensor(obs[a]["observation"], dtype=torch.float32, device=device).unsqueeze(0)
        gs_t = torch.tensor(obs[a]["global_state"], dtype=torch.float32, device=device).unsqueeze(0)
        role_t = torch.tensor(obs[a]["role_id"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = q_nets[a](obs_t, gs_t, role_t).squeeze(0)
        q = q.cpu().numpy()
        q[~mask.astype(bool)] = -1e9
        actions[a] = int(np.argmax(q))
    return actions


def train(args: Args):
    run_name = f"{args.exp_name}_s{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env_config = parse_env_config(args.env_config)
    env = HeistEnv(env_config)
    obs, _ = env.reset(seed=args.seed)
    state_dim = env.state().shape[0]

    # networks
    q_nets = {a: QNetwork().to(device) for a in AGENTS}
    target_nets = {a: QNetwork().to(device) for a in AGENTS}
    for a in AGENTS:
        target_nets[a].load_state_dict(q_nets[a].state_dict())

    mixing = QMixMixing(n_agents=len(AGENTS), state_dim=state_dim).to(device)
    target_mixing = QMixMixing(n_agents=len(AGENTS), state_dim=state_dim).to(device)
    target_mixing.load_state_dict(mixing.state_dict())

    params = list(mixing.parameters())
    for a in AGENTS:
        params += list(q_nets[a].parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    replay = ReplayBuffer(args.buffer_size, OBSERVATION_SIZE, state_dim)

    # ------------------------------------------------------------------
    def epsilon_at(step):
        frac = min(step / args.eps_anneal_steps, 1.0)
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    def eval_policies(step):
        from evaluate import evaluate_policies
        metrics = evaluate_policies(
            {a: q_nets[a] for a in AGENTS}, env, episodes=args.eval_episodes,
            seed=args.seed + 1_000_000, algo="qmix", device=device,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step)
        print(f"  eval@{step}: win_rate={metrics['win_rate']:.3f} "
              f"return={metrics['mean_return']:.3f} len={metrics['mean_length']:.1f}")

    # ------------------------------------------------------------------
    global_step = 0
    episode = 0
    total_reward = 0.0

    while global_step < args.total_steps:
        obs, _ = env.reset(seed=args.seed + episode)
        states = env.state()
        episode += 1
        ep_reward = 0.0
        done = False

        while not done and global_step < args.total_steps:
            epsilon = epsilon_at(global_step)
            actions = select_actions(env, obs, q_nets, epsilon, device)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done = bool(any(terms.values()) or any(truncs.values()))
            next_states = env.state()

            replay.push(obs, actions, rewards, any(terms.values()),
                        any(truncs.values()), states, next_obs, next_states)

            obs = next_obs
            states = next_states
            ep_reward += sum(rewards.values()) / len(AGENTS)
            global_step += 1
            total_reward += sum(rewards.values()) / len(AGENTS)

            # ---------------- gradient step ----------------
            if replay.size > args.learning_starts and global_step % args.train_freq == 0:
                batch = replay.sample(args.batch_size)
                for key in batch:
                    if isinstance(batch[key], dict):
                        batch[key] = {a: t.to(device) for a, t in batch[key].items()}
                    else:
                        batch[key] = batch[key].to(device)

                # online joint Q
                q_vals = []
                for a in AGENTS:
                    q = q_nets[a](batch["obs"][a], batch["gs"][a], batch["role"][a])
                    q = q.gather(1, batch["actions"][a].unsqueeze(1)).squeeze(1)
                    q_vals.append(q)
                q_vals = torch.stack(q_vals, dim=1)  # [B, n]
                q_tot = mixing(q_vals, batch["states"])

                # target joint Q: max over valid next actions
                with torch.no_grad():
                    tq = []
                    for a in AGENTS:
                        q_next = target_nets[a](batch["next_obs"][a], batch["next_gs"][a],
                                                batch["role"][a])
                        masked = torch.where(
                            batch["next_mask"][a] == 1, q_next,
                            torch.full_like(q_next, -1e9))
                        tq.append(masked.max(dim=1).values)
                    tq = torch.stack(tq, dim=1)  # [B, n]
                    q_tot_target = target_mixing(tq, batch["next_states"])
                    rewards = torch.stack([batch["rewards"][a] for a in AGENTS], dim=1).mean(dim=1)
                    # REV-3: bootstrapping uses terminations only.  For a
                    # TRUNCATED transition the stored next state IS the true
                    # terminal state (the env is not auto-reset in QMIX), so
                    # the target correctly bootstraps q_tot_target there.
                    y = rewards + args.gamma * (1 - batch["terminated"]) * q_tot_target

                loss = nn.functional.mse_loss(q_tot, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 10.0)
                optimizer.step()

                # polyak target update
                for a in AGENTS:
                    for tp, p in zip(target_nets[a].parameters(), q_nets[a].parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                for tp, p in zip(target_mixing.parameters(), mixing.parameters()):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

                writer.add_scalar("losses/q_loss", loss.item(), global_step)

            writer.add_scalar("charts/epsilon", epsilon, global_step)
            if global_step % 1000 == 0:
                writer.add_scalar("charts/episode_return", ep_reward, global_step)
                print(f"step={global_step} eps={epsilon:.3f} ep_reward={ep_reward:.2f} "
                      f"buffer={replay.size}")

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                for a in AGENTS:
                    torch.save(q_nets[a].state_dict(), f"checkpoints/{run_name}/{a}_q.pt")
                torch.save(mixing.state_dict(), f"checkpoints/{run_name}/mixing.pt")
                eval_policies(global_step)
                # eval resets the live env, invalidating the current training
                # episode.  Break out so the outer loop does a fresh reset.
                done = True

    # final save
    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
    for a in AGENTS:
        torch.save(q_nets[a].state_dict(), f"checkpoints/{run_name}/{a}_q.pt")
    torch.save(mixing.state_dict(), f"checkpoints/{run_name}/mixing.pt")
    writer.close()
    print("training done.")


if __name__ == "__main__":
    train(parse_args())
