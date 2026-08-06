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
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from constants import (
    ACTION_SPACE_SIZE as ACTION_DIM,
)
from constants import (
    N_AGENTS,
    OBSERVATION_SIZE,
)
from env import AGENTS, HeistEnv, parse_env_config
from model import QMixMixing, QNetwork
from ppo_utils import write_completion


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
    tau: float = 0.005  # polyak target update
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_anneal_steps: int = 500_000
    learning_starts: int = 10_000
    train_freq: int = 1  # gradient steps per env step
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
    p.add_argument(
        "--use-rnd", action="store_true", help="enable RND exploration bonus"
    )
    p.add_argument(
        "--rnd-coef", type=float, default=0.05, help="RND reward coefficient"
    )
    p.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="custom checkpoint path for transfer",
    )
    args = p.parse_args()
    if args.train_freq < 1:
        p.error("--train-freq must be positive")
    if args.target_update_freq < 1:
        p.error("--target-update-freq must be positive")
    return args


# ---------------------------------------------------------------------------
# Replay buffer of joint transitions
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, state_dim):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self.obs = {a: np.zeros((capacity, *obs_shape), dtype=np.int32) for a in AGENTS}
        self.mask = {a: np.zeros((capacity, ACTION_DIM), dtype=np.int8) for a in AGENTS}
        # REV-7 (REVISION_PLAN.md §6): no per-agent global_state; the QMIX
        # Q-networks see local view + role one-hot, the mixing net sees
        # env.state() (still stored in self.states / self.next_states).
        self.role = {a: np.zeros((capacity, N_AGENTS), dtype=np.int8) for a in AGENTS}
        self.actions = {a: np.zeros(capacity, dtype=np.int64) for a in AGENTS}
        self.rewards = {a: np.zeros(capacity, dtype=np.float32) for a in AGENTS}
        self.next_obs = {
            a: np.zeros((capacity, *obs_shape), dtype=np.int32) for a in AGENTS
        }
        self.next_mask = {
            a: np.zeros((capacity, ACTION_DIM), dtype=np.int8) for a in AGENTS
        }
        # REV-3: store terminations and truncations separately (QMIX bootstraps
        # the target on truncation from the stored terminal next state).
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.truncated = np.zeros(capacity, dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)

    def push(
        self,
        obs,
        actions,
        rewards,
        terminations,
        truncations,
        states,
        next_obs,
        next_states,
    ):
        for a in AGENTS:
            self.obs[a][self.pos] = obs[a]["observation"]
            self.mask[a][self.pos] = obs[a]["action_mask"]
            self.role[a][self.pos] = obs[a]["role_id"]
            self.actions[a][self.pos] = actions[a]
            self.rewards[a][self.pos] = rewards[a]
            self.next_obs[a][self.pos] = next_obs[a]["observation"]
            self.next_mask[a][self.pos] = next_obs[a]["action_mask"]
        self.terminated[self.pos] = float(terminations)
        self.truncated[self.pos] = float(truncations)
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        def tensor(array, dtype):
            # Advanced NumPy indexing already materializes a contiguous batch;
            # from_numpy avoids an additional CPU-side copy before transfer.
            return torch.from_numpy(array).to(dtype=dtype)

        batch = {
            "obs": {a: tensor(self.obs[a][idx], torch.float32) for a in AGENTS},
            "mask": {a: tensor(self.mask[a][idx], torch.long) for a in AGENTS},
            "role": {a: tensor(self.role[a][idx], torch.float32) for a in AGENTS},
            "actions": {a: tensor(self.actions[a][idx], torch.long) for a in AGENTS},
            "rewards": {a: tensor(self.rewards[a][idx], torch.float32) for a in AGENTS},
            "next_obs": {
                a: tensor(self.next_obs[a][idx], torch.float32) for a in AGENTS
            },
            "next_mask": {
                a: tensor(self.next_mask[a][idx], torch.long) for a in AGENTS
            },
            "terminated": tensor(self.terminated[idx], torch.float32),
            "truncated": tensor(self.truncated[idx], torch.float32),
            "states": tensor(self.states[idx], torch.float32),
            "next_states": tensor(self.next_states[idx], torch.float32),
        }
        return batch


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------
def select_actions(env, obs, q_nets, epsilon, device):
    """Epsilon-greedy joint action selection respecting action masks."""
    actions = {}
    obs_batch = torch.tensor(
        np.stack([obs[a]["observation"] for a in AGENTS]),
        dtype=torch.float32,
        device=device,
    )
    role_batch = torch.tensor(
        np.stack([obs[a]["role_id"] for a in AGENTS]),
        dtype=torch.float32,
        device=device,
    )
    mask_batch = torch.tensor(
        np.stack([obs[a]["action_mask"] for a in AGENTS]),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        q_list = []
        for i, a in enumerate(AGENTS):
            q_i = q_nets[a](obs_batch[i : i + 1], role_batch[i : i + 1])
            q_list.append(q_i.squeeze(0))
        q_all = torch.stack(q_list, dim=0)  # [n_agents, ACTION_DIM]
        q_masked = torch.where(mask_batch == 1, q_all, torch.full_like(q_all, -1e9))
        greedy_actions = q_masked.argmax(dim=-1).cpu().numpy()

    for i, a in enumerate(AGENTS):
        mask = obs[a]["action_mask"]
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            actions[a] = int(np.random.randint(ACTION_DIM))
        elif np.random.rand() < epsilon:
            actions[a] = int(np.random.choice(legal))
        else:
            actions[a] = int(greedy_actions[i])
    return actions


def train(args: Args):
    import re

    run_name = (
        args.exp_name
        if re.search(r"_s\d+", args.exp_name)
        else f"{args.exp_name}_s{args.seed}"
    )
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
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

    # Stage-to-stage policy transfer
    from ppo_utils import get_previous_stage_checkpoint, load_matching_weights

    prev_ckpt = get_previous_stage_checkpoint(run_name, args.exp_name)
    if prev_ckpt:
        print(
            f"  [Transfer] Loading previous stage checkpoint from {prev_ckpt}"
        )
        for a in AGENTS:
            load_matching_weights(
                q_nets[a], os.path.join(prev_ckpt, f"{a}_q.pt"), device
            )
            target_nets[a].load_state_dict(q_nets[a].state_dict())

    if args.load_checkpoint:
        print(f"  [Transfer] Loading custom checkpoint from {args.load_checkpoint}")
        for a in AGENTS:
            load_matching_weights(q_nets[a], f"{args.load_checkpoint}/{a}_q.pt", device)
            target_nets[a].load_state_dict(q_nets[a].state_dict())

    rnd_module = None
    if args.use_rnd:
        from exploration import RNDModule

        rnd_module = RNDModule(obs_dim=25, device=device)
        print("  [Exploration] RND Module initialized.")

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
            {a: q_nets[a] for a in AGENTS},
            env,
            episodes=args.eval_episodes,
            seed=args.seed + 1_000_000,
            algo="qmix",
            device=device,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step)
        print(
            f"  eval@{step}: win_rate={metrics['win_rate']:.3f} "
            f"return={metrics['mean_return']:.3f} len={metrics['mean_length']:.1f}"
        )

        # Periodic CAI and Counterfactual evaluation to track credit metrics during training (every 50k steps or final step)
        if (
            step > 0
            and (
                (
                    step % 50000 < 50
                )  # QMIX step updates are env steps (single env), so check if within a small window
                or (step >= args.total_steps - 50)
            )
        ):
            try:
                from evaluate import counterfactual_importance, credit_attribution_index

                cai = credit_attribution_index(
                    {a: q_nets[a] for a in AGENTS},
                    env,
                    episodes=args.eval_episodes,
                    seed=args.seed + 1_000_000,
                    device=device,
                )
                for agent_name, val in cai.items():
                    writer.add_scalar(f"eval/cai/{agent_name}", val, step)

                imp = counterfactual_importance(
                    {a: q_nets[a] for a in AGENTS},
                    env,
                    episodes=args.eval_episodes,
                    seed=args.seed + 1_000_000,
                    device=device,
                )
                for agent_name, val in imp.items():
                    writer.add_scalar(f"eval/importance/{agent_name}", val, step)
            except Exception as e:
                print(
                    f"  [Diagnostics] Warning: failed to calculate CAI/importance metrics: {e}"
                )

    # ------------------------------------------------------------------
    start_time = time.time()
    start_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_time))
    print(
        f"[{start_utc}] Training started: run_name={run_name}, total_steps={args.total_steps}"
    )
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

            if rnd_module is not None:
                obs_stacked = torch.stack(
                    [torch.tensor(obs[a]["observation"], device=device) for a in AGENTS]
                )
                r_int = rnd_module.compute_reward(obs_stacked)
                for idx_a, a in enumerate(AGENTS):
                    rewards[a] = float(rewards[a] + args.rnd_coef * r_int[idx_a].item())
                rnd_module.update(obs_stacked)

            replay.push(
                obs,
                actions,
                rewards,
                any(terms.values()),
                any(truncs.values()),
                states,
                next_obs,
                next_states,
            )

            obs = next_obs
            states = next_states
            ep_reward += sum(rewards.values()) / len(AGENTS)
            global_step += 1
            total_reward += sum(rewards.values()) / len(AGENTS)

            # ---------------- gradient step ----------------
            if (
                replay.size > args.learning_starts
                and global_step % args.train_freq == 0
            ):
                batch = replay.sample(args.batch_size)
                for key in batch:
                    if isinstance(batch[key], dict):
                        batch[key] = {a: t.to(device) for a, t in batch[key].items()}
                    else:
                        batch[key] = batch[key].to(device)

                # online joint Q
                q_vals = []
                for a in AGENTS:
                    q = q_nets[a](batch["obs"][a], batch["role"][a])
                    q = q.gather(1, batch["actions"][a].unsqueeze(1)).squeeze(1)
                    q_vals.append(q)
                q_vals = torch.stack(q_vals, dim=1)  # [B, n]
                q_tot = mixing(q_vals, batch["states"])

                # target joint Q: max over valid next actions
                with torch.no_grad():
                    tq = []
                    for a in AGENTS:
                        q_next = target_nets[a](batch["next_obs"][a], batch["role"][a])
                        masked = torch.where(
                            batch["next_mask"][a] == 1,
                            q_next,
                            torch.full_like(q_next, -1e9),
                        )
                        tq.append(masked.max(dim=1).values)
                    tq = torch.stack(tq, dim=1)  # [B, n]
                    q_tot_target = target_mixing(tq, batch["next_states"])
                    rewards = torch.stack(
                        [batch["rewards"][a] for a in AGENTS], dim=1
                    ).mean(dim=1)
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

                # Apply Polyak updates at the configured cadence.  Compound
                # tau across skipped gradient steps so the target lag remains
                # comparable to the prior every-step implementation.
                if global_step % args.target_update_freq == 0:
                    update_tau = 1 - (1 - args.tau) ** args.target_update_freq
                    for a in AGENTS:
                        for tp, p in zip(
                            target_nets[a].parameters(),
                            q_nets[a].parameters(),
                            strict=True,
                        ):
                            tp.data.copy_(
                                update_tau * p.data + (1 - update_tau) * tp.data
                            )
                    for tp, p in zip(
                        target_mixing.parameters(), mixing.parameters(), strict=True
                    ):
                        tp.data.copy_(update_tau * p.data + (1 - update_tau) * tp.data)

                writer.add_scalar("losses/q_loss", loss.item(), global_step)

            writer.add_scalar("charts/epsilon", epsilon, global_step)
            if global_step % 1000 == 0:
                writer.add_scalar("charts/episode_return", ep_reward, global_step)
                print(
                    f"step={global_step} eps={epsilon:.3f} ep_reward={ep_reward:.2f} "
                    f"buffer={replay.size}"
                )

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                for a in AGENTS:
                    torch.save(
                        q_nets[a].state_dict(), f"checkpoints/{run_name}/{a}_q.pt"
                    )
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
    eval_policies(global_step)
    write_completion(run_name, "qmix", args.total_steps, global_step)
    writer.close()
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    train(parse_args())
