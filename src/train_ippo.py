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
from ppo_utils import compute_gae, write_completion
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

    save_model: bool = True
    load_checkpoint: str = ""
    no_cuda: bool = False
    use_rnd: bool = False
    rnd_coef: float = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--torch-deterministic", action="store_true", default=True)
    p.add_argument("--track", action="store_true")
    p.add_argument("--save-model", action="store_true", default=True)
    p.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="disable checkpointing",
    )
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
    return p.parse_args()


def log_summary(writer, global_step, metrics):
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)


def train(args: Args):
    import re

    run_name = (
        args.exp_name
        if re.search(r"_s\d+", args.exp_name)
        else f"{args.exp_name}_s{args.seed}"
    )
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
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    writer = SummaryWriter(f"runs/{run_name}", flush_secs=30)

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

    # Stage-to-stage policy transfer
    from ppo_utils import get_previous_stage_checkpoint, load_matching_weights

    prev_ckpt = get_previous_stage_checkpoint(run_name, args.exp_name)
    if prev_ckpt:
        print(f"  [Transfer] Loading previous stage checkpoint from {prev_ckpt}")
        if args.shared:
            load_matching_weights(
                shared_policy, os.path.join(prev_ckpt, "scout.pt"), device
            )
        else:
            for a in AGENTS:
                load_matching_weights(
                    policies[a], os.path.join(prev_ckpt, f"{a}.pt"), device
                )

    if args.load_checkpoint:
        print(f"  [Transfer] Loading custom checkpoint from {args.load_checkpoint}")
        if args.shared:
            load_matching_weights(
                shared_policy, f"{args.load_checkpoint}/scout.pt", device
            )
        else:
            for a in AGENTS:
                load_matching_weights(
                    policies[a], f"{args.load_checkpoint}/{a}.pt", device
                )

    rnd_module = None
    if args.use_rnd:
        from exploration import RNDModule

        rnd_module = RNDModule(
            obs_dim=OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1], device=device
        )
        print("  [Exploration] RND Module initialized.")

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

        # Periodic CAI and Counterfactual evaluation to track credit metrics during training (every 50k steps or final step)
        if step > 0 and (
            (step % 50000 < (args.num_steps * args.num_envs))
            or (step >= args.total_timesteps - (args.num_steps * args.num_envs))
        ):
            try:
                from evaluate import counterfactual_importance, credit_attribution_index

                cai = credit_attribution_index(
                    {a: policies[a] for a in AGENTS},
                    vec_env.envs[0],
                    episodes=args.eval_episodes,
                    seed=args.seed + 1_000_000,
                    device=device,
                )
                for agent_name, val in cai.items():
                    writer.add_scalar(f"eval/cai/{agent_name}", val, step)

                imp = counterfactual_importance(
                    {a: policies[a] for a in AGENTS},
                    vec_env.envs[0],
                    episodes=args.eval_episodes,
                    seed=args.seed + 1_000_000,
                    device=device,
                )
                if isinstance(imp, dict):
                    if "baseline_win_rate" in imp:
                        writer.add_scalar(
                            "eval/importance/baseline_win_rate",
                            imp["baseline_win_rate"],
                            step,
                        )
                    importance_dict = imp.get("importance", {})
                    if isinstance(importance_dict, dict):
                        for agent_name, val in importance_dict.items():
                            writer.add_scalar(
                                f"eval/importance/{agent_name}", val, step
                            )
            except Exception as e:
                print(
                    f"  [Diagnostics] Warning: failed to calculate CAI/importance metrics: {e}"
                )

    start_time = time.time()
    start_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(start_time))
    print(
        f"[{start_utc}] Training started: run_name={run_name}, total_timesteps={args.total_timesteps}"
    )
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
            global_step += args.num_envs
            actions_dict = {}
            stacked = next_obs["_stacked"]
            obs_all = torch.as_tensor(stacked["observation"], device=device)
            role_all = torch.as_tensor(stacked["role_id"], device=device)
            mask_all = torch.as_tensor(stacked["action_mask"], device=device)
            with torch.no_grad():
                if args.shared:
                    flat_obs = obs_all.flatten(0, 1)
                    flat_role = role_all.flatten(0, 1)
                    flat_mask = mask_all.flatten(0, 1)
                    all_actions, all_logprobs, _, all_values = (
                        shared_policy.get_action_and_value(
                            flat_obs, flat_role, flat_mask
                        )
                    )
                    all_actions = all_actions.view(len(AGENTS), args.num_envs)
                    all_logprobs = all_logprobs.view(len(AGENTS), args.num_envs)
                    all_values = all_values.view(len(AGENTS), args.num_envs)
                for i, a in enumerate(AGENTS):
                    if not args.shared:
                        action, logprob, _, value = policies[a].get_action_and_value(
                            obs_all[i], role_all[i], mask_all[i]
                        )
                    else:
                        action, logprob, value = (
                            all_actions[i],
                            all_logprobs[i],
                            all_values[i],
                        )
                    buffers[a]["obs"][step] = obs_all[i]
                    buffers[a]["role_id"][step] = role_all[i]
                    buffers[a]["action_mask"][step] = mask_all[i]
                    buffers[a]["actions"][step] = action
                    buffers[a]["logprobs"][step] = logprob
                    buffers[a]["values"][step] = value.flatten()
                    actions_dict[a] = action.cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = vec_env.step(
                actions_dict
            )
            next_terminations = torch.as_tensor(
                terminations["scout"], device=device
            ).float()
            next_truncations = torch.as_tensor(
                truncations["scout"], device=device
            ).float()
            last_infos = infos
            r_int_tensor = None
            if rnd_module is not None:
                r_int_tensor = rnd_module.compute_reward(obs_all)
            for idx_a, a in enumerate(AGENTS):
                r_step = torch.as_tensor(rewards[a], device=device)
                if r_int_tensor is not None:
                    r_step = r_step + args.rnd_coef * r_int_tensor[idx_a]
                buffers[a]["rewards"][step] = r_step
                term_t = torch.as_tensor(terminations[a], device=device).float()
                trunc_t = torch.as_tensor(truncations[a], device=device).float()
                buffers[a]["terminated"][step] = term_t
                buffers[a]["truncated"][step] = trunc_t
                # REV-3: store the bootstrap value for envs done this step.
                with torch.no_grad():
                    t_idx = trunc_t.bool().nonzero(as_tuple=False).flatten()
                    if t_idx.numel():
                        # terminal_observation is packed per-agent with a
                        # leading env dim of 1 (vec_env._pack([o])); strip it.
                        t_obs = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(i)]["terminal_observation"][a][
                                        "observation"
                                    ][0],
                                    device=device,
                                )
                                for i in t_idx
                            ]
                        )
                        t_role = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(i)]["terminal_observation"][a]["role_id"][
                                        0
                                    ],
                                    device=device,
                                )
                                for i in t_idx
                            ]
                        )
                        val = policies[a].get_value(t_obs, t_role).flatten()
                        buffers[a]["bootstrap"][step][t_idx] = val

        # ---------------- GAE + returns ----------------
        next_values = []
        for agent_index, a in enumerate(AGENTS):
            with torch.no_grad():
                next_value = (
                    policies[a]
                    .get_value(
                        torch.as_tensor(
                            next_obs["_stacked"]["observation"][agent_index],
                            device=device,
                        ),
                        torch.as_tensor(
                            next_obs["_stacked"]["role_id"][agent_index],
                            device=device,
                        ),
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
                                ][0],
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
                                ][0],
                                device=device,
                            )
                            for i in t_idx
                        ]
                    )
                    next_value[t_idx] = policies[a].get_value(t_obs, t_role).flatten()
                # terminated envs bootstrap to 0
                next_value = next_value * (1.0 - next_terminations)
            next_values.append(next_value)
        stacked_advantages, stacked_returns = compute_gae(
            torch.stack([buffers[a]["rewards"] for a in AGENTS]),
            torch.stack([buffers[a]["values"] for a in AGENTS]),
            torch.stack([buffers[a]["terminated"] for a in AGENTS]),
            torch.stack([buffers[a]["truncated"] for a in AGENTS]),
            torch.stack([buffers[a]["bootstrap"] for a in AGENTS]),
            torch.stack(next_values),
            next_terminations.unsqueeze(0),
            args.gamma,
            args.gae_lambda,
        )
        advantages = {a: stacked_advantages[i] for i, a in enumerate(AGENTS)}
        returns = {a: stacked_returns[i] for i, a in enumerate(AGENTS)}

        # ---------------- policy update ----------------
        # each agent updates its own policy on its own data (IPPO);
        # when --shared, all data is concatenated into one batch.
        groups = [("all", AGENTS)] if args.shared else [(a, [a]) for a in AGENTS]

        for group_name, agent_list in groups:
            b_obs = torch.cat(
                [buffers[a]["obs"].reshape(-1, obs_h, obs_w) for a in agent_list]
            )
            b_mask = torch.cat(
                [buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in agent_list]
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
            b_advantages = torch.cat([advantages[a].reshape(-1) for a in agent_list])
            b_returns = torch.cat([returns[a].reshape(-1) for a in agent_list])
            minibatch_size = b_obs.shape[0] // args.num_minibatches
            for _ in range(args.update_epochs):
                b_inds = torch.randperm(b_obs.shape[0], device=device)

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
                    if rnd_module is not None:
                        rnd_module.update(b_obs[mb])

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

        if args.save_model and (update % args.eval_every == 0 or update == num_updates):
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
        write_completion(
            run_name,
            "ippo",
            args.total_timesteps,
            num_updates * args.num_steps * args.num_envs,
        )
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    args = parse_args()
    train(args)
