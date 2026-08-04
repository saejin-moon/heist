"""
IPPO (Independent PPO) baseline trainer for HEIST.

PLAN.md Baseline 1: each agent runs an independent PPO policy (its own
actor-critic network and its own optimizer) on its own observation stream.
Agents only interact through the shared reward and the environment's causal
gates.  This is the standard IPPO setup and serves as the weak baseline
against which MAPPO and QMIX are compared.

Two modes:
  * --shared:  parameter sharing variant.  All four agents share ONE
               policy/optimizer and learn from all four data streams
               concatenated (CleanRL ppo_parallel style).
  * default:   fully independent per-agent policies.

Example:
    uv run python src/train_ippo.py --total-timesteps 2_000_000
    uv run python src/train_ippo.py --shared --total-timesteps 2_000_000
"""

import argparse
import os
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
from env import AGENTS, parse_env_config
from model import HeistAgent
from vec_env import VectorEnv


@dataclass
class Args:
    # experiment
    exp_name: str = "ippo"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "heist"
    wandb_entity: str | None = None
    save_model: bool = True
    # algorithm
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
    shared: bool = False
    # evaluation
    eval_every: int = 20
    eval_episodes: int = 20
    # environment override (json path or inline python-ish dict)
    env_config: str = ""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--torch-deterministic", action="store_true", default=True)
    p.add_argument("--track", action="store_true")
    p.add_argument("--save-model", action="store_true", default=True)
    p.add_argument("--no-save-model", action="store_true", help="disable checkpointing")
    p.add_argument("--wandb-project-name", type=str, default=Args.wandb_project_name)
    p.add_argument("--wandb-entity", type=str, default=None)
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
    p.add_argument("--shared", action="store_true")
    p.add_argument("--eval-every", type=int, default=Args.eval_every)
    p.add_argument("--eval-episodes", type=int, default=Args.eval_episodes)
    p.add_argument("--env-config", type=str, default="")
    return p.parse_args()


def log_summary(writer, global_step, metrics):
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)


def train(args: Args):
    run_name = f"{args.exp_name}_s{args.seed}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"device: {device}")

    args.save_model = not getattr(args, "no_save_model", False)

    # deterministic seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_config = parse_env_config(args.env_config)

    # ------------------------------------------------------------------
    # policies: one per agent (independent) or a single shared network
    # ------------------------------------------------------------------
    if args.shared:
        # REV-2 (REVISION_PLAN.md §2): one shared network for all four roles;
        # role identity must arrive via the env-issued role one-hot once the
        # observation contract is updated (REV-1).
        shared_policy = HeistAgent().to(device)
        policies = {a: shared_policy for a in AGENTS}
        optimizers = {
            "all": torch.optim.Adam(
                shared_policy.parameters(), lr=args.learning_rate, eps=1e-5
            )
        }
    else:
        policies = {a: HeistAgent().to(device) for a in AGENTS}
        optimizers = {
            a: torch.optim.Adam(p.parameters(), lr=args.learning_rate, eps=1e-5)
            for a, p in policies.items()
        }

    vec_env = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)
    next_obs, next_state = vec_env.reset(seed=args.seed)
    # REV-3: termination vs truncation tracked separately; the boundary of the
    # rollout bootstraps truncated envs from the true terminal observation.
    next_terminations = torch.zeros(args.num_envs, device=device)
    next_truncations = torch.zeros(args.num_envs, device=device)
    last_infos = [{} for _ in range(args.num_envs)]

    # ------------------------------------------------------------------
    # rollout buffers (per agent)
    # ------------------------------------------------------------------
    obs_h, obs_w = OBSERVATION_SIZE
    buffers = {}
    for a in AGENTS:
        buffers[a] = {
            "obs": torch.zeros(
                (args.num_steps, args.num_envs, obs_h, obs_w), device=device
            ),
            "action_mask": torch.zeros(
                (args.num_steps, args.num_envs, ACTION_DIM), device=device
            ),
            # REV-7 (REVISION_PLAN.md §6): no global_state buffer; baselines
            # navigate from local view + role one-hot only.
            "role_id": torch.zeros(
                (args.num_steps, args.num_envs, N_AGENTS), device=device
            ),
            "actions": torch.zeros(
                (args.num_steps, args.num_envs), dtype=torch.long, device=device
            ),
            "logprobs": torch.zeros((args.num_steps, args.num_envs), device=device),
            "rewards": torch.zeros((args.num_steps, args.num_envs), device=device),
            "terminated": torch.zeros((args.num_steps, args.num_envs), device=device),
            "truncated": torch.zeros((args.num_steps, args.num_envs), device=device),
            # value to bootstrap with for envs done at this step (REV-3):
            # V(terminal obs) for truncations, 0 for terminations
            "bootstrap": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
        }

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    print(f"num_updates: {num_updates}")

    def eval_policies(step):
        """Run a quick greedy evaluation of current policies."""
        from evaluate import evaluate_policies

        metrics = evaluate_policies(
            {a: policies[a] for a in AGENTS},
            vec_env.envs[0],
            episodes=args.eval_episodes,
            seed=args.seed + 1_000_000,
            algo="ippo",
            device=device,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step)
        print(
            f"  eval@{step}: win_rate={metrics['win_rate']:.3f} "
            f"return={metrics['mean_return']:.3f} len={metrics['mean_length']:.1f}"
        )

    start_time = time.time()
    global_step = 0

    for update in range(1, num_updates + 1):
        # anneal lr
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for opt in optimizers.values():
                for g in opt.param_groups:
                    g["lr"] = lr_now

        # ---------------- rollout ----------------
        for step in range(args.num_steps):
            global_step += 1
            actions_dict = {}
            for a in AGENTS:
                with torch.no_grad():
                    obs_t = torch.tensor(next_obs[a]["observation"], device=device)
                    role_t = torch.tensor(next_obs[a]["role_id"], device=device)
                    mask_t = torch.tensor(next_obs[a]["action_mask"], device=device)
                    action, logprob, _, value = policies[a].get_action_and_value(
                        obs_t, role_t, mask_t
                    )
                buffers[a]["obs"][step] = obs_t
                buffers[a]["role_id"][step] = role_t
                buffers[a]["action_mask"][step] = mask_t
                buffers[a]["actions"][step] = action
                buffers[a]["logprobs"][step] = logprob
                buffers[a]["values"][step] = value.flatten()
                actions_dict[a] = action.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = vec_env.step(
                actions_dict
            )
            next_terminations = torch.tensor(
                terminations["scout"], device=device
            ).float()
            next_truncations = torch.tensor(truncations["scout"], device=device).float()
            last_infos = infos
            for a in AGENTS:
                buffers[a]["rewards"][step] = torch.tensor(rewards[a], device=device)
                term_t = torch.tensor(terminations[a], device=device).float()
                trunc_t = torch.tensor(truncations[a], device=device).float()
                buffers[a]["terminated"][step] = term_t
                buffers[a]["truncated"][step] = trunc_t
                # REV-3: store the bootstrap value for envs done this step.
                with torch.no_grad():
                    t_idx = trunc_t.bool().nonzero(as_tuple=False).flatten()
                    if t_idx.numel():
                        t_obs = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(i)]["terminal_observation"][a][
                                        "observation"
                                    ],
                                    device=device,
                                )
                                for i in t_idx
                            ]
                        )
                        t_role = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(i)]["terminal_observation"][a]["role_id"],
                                    device=device,
                                )
                                for i in t_idx
                            ]
                        )
                        val = policies[a].get_value(t_obs, t_role).flatten()
                        buffers[a]["bootstrap"][step][t_idx] = val

        # ---------------- GAE + returns ----------------
        returns = {}
        advantages = {}
        for a in AGENTS:
            with torch.no_grad():
                next_value = (
                    policies[a]
                    .get_value(
                        torch.tensor(next_obs[a]["observation"], device=device),
                        torch.tensor(next_obs[a]["role_id"], device=device),
                    )
                    .flatten()
                )
                # REV-3: truncated envs bootstrap to value(terminal obs), not 0.
                t_idx = next_truncations.bool().nonzero(as_tuple=False).flatten()
                if t_idx.numel():
                    t_obs = torch.stack(
                        [
                            torch.tensor(
                                last_infos[int(i)]["terminal_observation"][a][
                                    "observation"
                                ],
                                device=device,
                            )
                            for i in t_idx
                        ]
                    )
                    t_role = torch.stack(
                        [
                            torch.tensor(
                                last_infos[int(i)]["terminal_observation"][a][
                                    "role_id"
                                ],
                                device=device,
                            )
                            for i in t_idx
                        ]
                    )
                    next_value[t_idx] = policies[a].get_value(t_obs, t_role).flatten()
                # terminated envs bootstrap to 0
                next_value = next_value * (1.0 - next_terminations)
            adv = torch.zeros_like(buffers[a]["rewards"])
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_terminations
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffers[a]["terminated"][t + 1]
                    # REV-3: truncations bootstrap from the stored terminal value
                    nextvalues = torch.where(
                        buffers[a]["truncated"][t + 1].bool(),
                        buffers[a]["bootstrap"][t + 1],
                        buffers[a]["values"][t + 1],
                    )
                delta = (
                    buffers[a]["rewards"][t]
                    + args.gamma * nextvalues * nextnonterminal
                    - buffers[a]["values"][t]
                )
                adv[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            advantages[a] = adv
            returns[a] = adv + buffers[a]["values"]

        # ---------------- policy update ----------------
        # each agent updates its own policy on its own data (IPPO);
        # when --shared, all data is concatenated into one batch.
        groups = [("all", AGENTS)] if args.shared else [(a, [a]) for a in AGENTS]

        for group_name, agent_list in groups:
            for _ in range(args.update_epochs):
                b_obs = torch.cat(
                    [buffers[a]["obs"].reshape(-1, obs_h, obs_w) for a in agent_list]
                )
                b_mask = torch.cat(
                    [
                        buffers[a]["action_mask"].reshape(-1, ACTION_DIM)
                        for a in agent_list
                    ]
                )
                b_role = torch.cat(
                    [buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in agent_list]
                )
                b_actions = torch.cat(
                    [buffers[a]["actions"].reshape(-1) for a in agent_list]
                )
                b_logprobs = torch.cat(
                    [buffers[a]["logprobs"].reshape(-1) for a in agent_list]
                )
                b_advantages = torch.cat(
                    [advantages[a].reshape(-1) for a in agent_list]
                )
                b_returns = torch.cat([returns[a].reshape(-1) for a in agent_list])
                b_inds = np.arange(b_obs.shape[0])
                np.random.shuffle(b_inds)
                minibatch_size = b_obs.shape[0] // args.num_minibatches

                for start in range(0, b_obs.shape[0], minibatch_size):
                    end = start + minibatch_size
                    mb = b_inds[start:end]
                    policy = policies[agent_list[0]]
                    _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                        b_obs[mb], b_role[mb], b_mask[mb], b_actions[mb]
                    )

                    logratio = newlogprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()

                    adv = b_advantages[mb]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = ((newvalue.flatten() - b_returns[mb]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                    )

                    optimizers[group_name].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        policies[agent_list[0]].parameters(), args.max_grad_norm
                    )
                    optimizers[group_name].step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---------------- logging ----------------
        if update % 1 == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_reward = np.mean([buffers[a]["rewards"].mean().item() for a in AGENTS])
            writer.add_scalar("charts/global_step", global_step, global_step)
            writer.add_scalar("charts/sps", sps, global_step)
            writer.add_scalar("charts/mean_reward", avg_reward, global_step)
            writer.add_scalar(
                "charts/lr",
                optimizers[list(optimizers)[0]].param_groups[0]["lr"],
                global_step,
            )
            print(
                f"update={update} step={global_step} sps={sps} mean_reward={avg_reward:.4f}"
            )

        if args.save_model and update % args.eval_every == 0:
            os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
            for a, p in policies.items():
                torch.save(p.state_dict(), f"checkpoints/{run_name}/{a}.pt")
            eval_policies(global_step)

    vec_env.close()
    writer.close()
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        for a, p in policies.items():
            torch.save(p.state_dict(), f"checkpoints/{run_name}/{a}.pt")
    print("training done.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
