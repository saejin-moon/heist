"""
MAPPO (Multi-Agent PPO) trainer for HEIST.

PLAN.md Baseline 2: all agents share ONE policy network, but the critic is
centralized and consumes the full `env.state()` vector (grid + positions +
phase flags).  The actor still only sees the local, fog-masked observation,
so partial observability is removed only for value estimation.  This is the
standard MAPPO cooperative baseline that QMIX should outperform on
credit-assignment-heavy tasks.

Example:
    uv run python src/train_mappo.py --total-timesteps 2_000_000
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from env import AGENTS, parse_env_config
from constants import (
    OBSERVATION_SIZE, ACTION_SPACE_SIZE as ACTION_DIM, N_AGENTS,
)
from model import MappoAgent
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "mappo"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    eval_every: int = 20
    eval_episodes: int = 20
    env_config: str = ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--torch-deterministic", action="store_true", default=True)
    p.add_argument("--save-model", action="store_true", default=True)
    p.add_argument("--no-save-model", action="store_true", help="disable checkpointing")
    p.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    p.add_argument("--learning-rate", type=float, default=Args.learning_rate)
    p.add_argument("--num-envs", type=int, default=Args.num_envs)
    p.add_argument("--num-steps", type=int, default=Args.num_steps)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=Args.gamma)
    p.add_argument("--gae-lambda", type=float, default=Args.gae_lambda)
    p.add_argument("--num-minibatches", type=int, default=Args.num_minibatches)
    p.add_argument("--update-epochs", type=int, default=Args.update_epochs)
    p.add_argument("--clip-coef", type=float, default=Args.clip_coef)
    p.add_argument("--ent-coef", type=float, default=Args.ent_coef)
    p.add_argument("--vf-coef", type=float, default=Args.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=Args.max_grad_norm)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=Args.eval_every)
    p.add_argument("--eval-episodes", type=int, default=Args.eval_episodes)
    p.add_argument("--env-config", type=str, default="")
    return p.parse_args()


def train(args: Args):
    run_name = f"{args.exp_name}_s{args.seed}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device: {device}")
    args.save_model = not getattr(args, "no_save_model", False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_config = parse_env_config(args.env_config)
    vec_env = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)
    next_obs, next_state = vec_env.reset(seed=args.seed)
    state_dim = vec_env.state_dim
    # REV-3: termination vs truncation tracked separately; the boundary of the
    # rollout bootstraps truncated envs from the true terminal state.
    next_terminations = torch.zeros(args.num_envs, device=device)
    next_truncations = torch.zeros(args.num_envs, device=device)
    last_infos = [{} for _ in range(args.num_envs)]

    # shared policy: one actor + centralized critic
    # REV-2 (REVISION_PLAN.md §2): this single network controls all four roles;
    # it needs the env-issued role one-hot (REV-1) to avoid aliasing.
    policy = MappoAgent(state_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_h, obs_w = OBSERVATION_SIZE
    buffers = {}
    for a in AGENTS:
        buffers[a] = {
            "obs": torch.zeros((args.num_steps, args.num_envs, obs_h, obs_w), device=device),
            "action_mask": torch.zeros((args.num_steps, args.num_envs, ACTION_DIM), device=device),
            # REV-7 (REVISION_PLAN.md §6): no global_state buffer; the MAPPO
            # actor sees local view + role only, the centralized critic sees
            # env.state() from the shared state_buffer.
            "role_id": torch.zeros((args.num_steps, args.num_envs, N_AGENTS), device=device),
            "actions": torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device),
            "logprobs": torch.zeros((args.num_steps, args.num_envs), device=device),
            "rewards": torch.zeros((args.num_steps, args.num_envs), device=device),
            "terminated": torch.zeros((args.num_steps, args.num_envs), device=device),
            "truncated": torch.zeros((args.num_steps, args.num_envs), device=device),
            # REV-3: bootstrap value for envs done this step (V(terminal state)
            # for truncations, 0 for terminations)
            "bootstrap": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
        }
    state_buffer = torch.zeros((args.num_steps, args.num_envs, state_dim), device=device)

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    print(f"num_updates: {num_updates}")

    def eval_policies(step):
        from evaluate import evaluate_policies
        metrics = evaluate_policies(
            {a: policy for a in AGENTS}, vec_env.envs[0],
            episodes=args.eval_episodes, seed=args.seed + 1_000_000,
            algo="mappo", device=device,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step)
        print(f"  eval@{step}: win_rate={metrics['win_rate']:.3f} "
              f"return={metrics['mean_return']:.3f} len={metrics['mean_length']:.1f}")

    start_time = time.time()
    global_step = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for g in optimizer.param_groups:
                g["lr"] = lr_now

        # ---------------- rollout ----------------
        for step in range(args.num_steps):
            global_step += 1
            state_t = torch.tensor(next_state, device=device)
            state_buffer[step] = state_t
            actions_dict = {}
            for a in AGENTS:
                with torch.no_grad():
                    obs_t = torch.tensor(next_obs[a]["observation"], device=device)
                    role_t = torch.tensor(next_obs[a]["role_id"], device=device)
                    mask_t = torch.tensor(next_obs[a]["action_mask"], device=device)
                    action, logprob, _, value = policy.get_action_and_value(
                        obs_t, role_t, mask_t, state_t)
                buffers[a]["obs"][step] = obs_t
                buffers[a]["role_id"][step] = role_t
                buffers[a]["action_mask"][step] = mask_t
                buffers[a]["actions"][step] = action
                buffers[a]["logprobs"][step] = logprob
                buffers[a]["values"][step] = value.flatten()
                actions_dict[a] = action.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = vec_env.step(actions_dict)
            next_terminations = torch.tensor(terminations["scout"], device=device).float()
            next_truncations = torch.tensor(truncations["scout"], device=device).float()
            last_infos = infos
            for a in AGENTS:
                buffers[a]["rewards"][step] = torch.tensor(rewards[a], device=device)
                term_t = torch.tensor(terminations[a], device=device).float()
                trunc_t = torch.tensor(truncations[a], device=device).float()
                buffers[a]["terminated"][step] = term_t
                buffers[a]["truncated"][step] = trunc_t
                # REV-3: store critic(terminal_state) for envs truncated here.
                with torch.no_grad():
                    t_idx = trunc_t.bool().nonzero(as_tuple=False).flatten()
                    if t_idx.numel():
                        t_states = torch.stack([torch.tensor(
                            infos[int(i)]["terminal_state"], device=device) for i in t_idx])
                        val = policy.get_value(t_states).flatten()
                        buffers[a]["bootstrap"][step][t_idx] = val
            next_state = vec_env.state

        # ---------------- GAE + returns (centralized critic) ----------------
        with torch.no_grad():
            next_value = policy.get_value(
                torch.tensor(next_state, device=device)).flatten()
            # REV-3: truncated envs bootstrap to critic(terminal_state), not 0.
            t_idx = next_truncations.bool().nonzero(as_tuple=False).flatten()
            if t_idx.numel():
                t_states = torch.stack([torch.tensor(
                    last_infos[int(i)]["terminal_state"], device=device) for i in t_idx])
                next_value[t_idx] = policy.get_value(t_states).flatten()
            # terminated envs bootstrap to 0
            next_value = next_value * (1.0 - next_terminations)
        advantages = {}
        returns = {}
        for a in AGENTS:
            adv = torch.zeros_like(buffers[a]["rewards"])
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminations
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffers[a]["terminated"][t + 1]
                    nextvalues = torch.where(
                        buffers[a]["truncated"][t + 1].bool(),
                        buffers[a]["bootstrap"][t + 1],
                        buffers[a]["values"][t + 1],
                    )
                delta = buffers[a]["rewards"][t] + args.gamma * nextvalues * nextnonterminal \
                        - buffers[a]["values"][t]
                adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            advantages[a] = adv
            returns[a] = adv + buffers[a]["values"]

        # ---------------- policy update (concatenated across agents) ---------
        b_obs = torch.cat([buffers[a]["obs"].reshape(-1, obs_h, obs_w) for a in AGENTS])
        b_mask = torch.cat([buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in AGENTS])
        b_role = torch.cat([buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in AGENTS])
        b_actions = torch.cat([buffers[a]["actions"].reshape(-1) for a in AGENTS])
        b_logprobs = torch.cat([buffers[a]["logprobs"].reshape(-1) for a in AGENTS])
        b_advantages = torch.cat([advantages[a].reshape(-1) for a in AGENTS])
        b_returns = torch.cat([returns[a].reshape(-1) for a in AGENTS])
        # centralized critic sees the shared state, replicated per agent block
        # (each agent's flattened block shares the same (step, env) states)
        b_states = torch.cat([state_buffer.reshape(-1, state_dim) for _ in AGENTS])

        for epoch in range(args.update_epochs):
            b_inds = np.arange(b_obs.shape[0])
            np.random.shuffle(b_inds)
            minibatch_size = b_obs.shape[0] // args.num_minibatches

            for start in range(0, b_obs.shape[0], minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb], b_role[mb], b_mask[mb], b_states[mb], b_actions[mb])

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

                adv = b_advantages[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = ((newvalue.flatten() - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # ---------------- logging ----------------
        sps = int(global_step / (time.time() - start_time))
        avg_reward = np.mean([buffers[a]["rewards"].mean().item() for a in AGENTS])
        writer.add_scalar("charts/global_step", global_step, global_step)
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("charts/mean_reward", avg_reward, global_step)
        writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)
        print(f"update={update} step={global_step} sps={sps} mean_reward={avg_reward:.4f}")

        if update % args.eval_every == 0:
            if args.save_model:
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")
            eval_policies(global_step)

    vec_env.close()
    writer.close()
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")
    print("training done.")


if __name__ == "__main__":
    train(parse_args())
