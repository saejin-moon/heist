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

from constants import (
    ACTION_SPACE_SIZE as ACTION_DIM,
)
from constants import (
    N_AGENTS,
    OBSERVATION_SIZE,
)
from env import AGENTS, parse_env_config
from model import MappoAgent
from ppo_utils import compute_gae, write_completion
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
    car_coef: float = 0.0
    eval_every: int = 20
    eval_episodes: int = 20
    env_config: str = ""

    cir_coef: float = 0.0
    target_kl: float = 0.0
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
    p.add_argument("--save-model", action="store_true", default=True)
    p.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="disable checkpointing",
    )
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
    p.add_argument(
        "--car-coef", type=float, default=0.0, help="CAR intrinsic reward coefficient."
    )
    p.add_argument(
        "--cir-coef",
        type=float,
        default=0.0,
        help="CIR coefficient for routing advantages.",
    )
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


def train(args: Args):
    import re

    run_name = (
        args.exp_name
        if re.search(r"_s\d+", args.exp_name)
        else f"{args.exp_name}_s{args.seed}"
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

    # Stage-to-stage policy transfer
    from ppo_utils import get_previous_stage_checkpoint, load_matching_weights

    prev_ckpt = get_previous_stage_checkpoint(run_name, args.exp_name)
    if prev_ckpt:
        print(f"  [Transfer] Loading previous stage checkpoint from {prev_ckpt}")
        load_matching_weights(policy, os.path.join(prev_ckpt, "policy.pt"), device)

    if args.load_checkpoint:
        print(f"  [Transfer] Loading custom checkpoint from {args.load_checkpoint}")
        load_matching_weights(policy, f"{args.load_checkpoint}/policy.pt", device)

    rnd_module = None
    if args.use_rnd:
        from exploration import RNDModule

        rnd_module = RNDModule(
            obs_dim=OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1], device=device
        )
        print("  [Exploration] RND Module initialized.")

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
            # REV-7 (REVISION_PLAN.md §6): no global_state buffer; the MAPPO
            # actor sees local view + role only, the centralized critic sees
            # env.state() from the shared state_buffer.
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
            # REV-3: bootstrap value for envs done this step (V(terminal state)
            # for truncations, 0 for terminations)
            "bootstrap": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
        }
    state_buffer = torch.zeros(
        (args.num_steps, args.num_envs, state_dim), device=device
    )

    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    print(f"num_updates: {num_updates}")

    def eval_policies(step):
        from evaluate import evaluate_policies

        metrics = evaluate_policies(
            {a: policy for a in AGENTS},
            vec_env.envs[0],
            episodes=args.eval_episodes,
            seed=args.seed + 1_000_000,
            algo="mappo",
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
                    {a: policy for a in AGENTS},
                    vec_env.envs[0],
                    episodes=args.eval_episodes,
                    seed=args.seed + 1_000_000,
                    device=device,
                )
                for agent_name, val in cai.items():
                    writer.add_scalar(f"eval/cai/{agent_name}", val, step)

                imp = counterfactual_importance(
                    {a: policy for a in AGENTS},
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
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            for g in optimizer.param_groups:
                g["lr"] = lr_now

        # ---------------- rollout ----------------
        buf_influence = torch.zeros(
            (args.num_steps, args.num_envs, N_AGENTS, N_AGENTS), device=device
        )
        for step in range(args.num_steps):
            global_step += args.num_envs
            state_t = torch.as_tensor(next_state, device=device)
            state_buffer[step] = state_t
            if getattr(args, "cir_coef", 0.0) > 0.0:
                with torch.no_grad():
                    buf_influence[step] = policy.get_influence_matrix(state_t)
            stacked = next_obs["_stacked"]
            obs_all = torch.as_tensor(stacked["observation"], device=device)
            role_all = torch.as_tensor(stacked["role_id"], device=device)
            mask_all = torch.as_tensor(stacked["action_mask"], device=device)
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    state_t.repeat(len(AGENTS), 1),
                )
            actions = action.view(len(AGENTS), args.num_envs)
            logprobs = logprob.view(len(AGENTS), args.num_envs)
            values = value.view(len(AGENTS), args.num_envs)
            actions_dict = {}
            for i, a in enumerate(AGENTS):
                buffers[a]["obs"][step] = obs_all[i]
                buffers[a]["role_id"][step] = role_all[i]
                buffers[a]["action_mask"][step] = mask_all[i]
                buffers[a]["actions"][step] = actions[i]
                buffers[a]["logprobs"][step] = logprobs[i]
                buffers[a]["values"][step] = values[i]
                actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = vec_env.step(
                actions_dict
            )
            # --- CAR: Counterfactual Affordance Reward ---
            if getattr(args, "car_coef", 0.0) > 0.0:
                new_masks = next_obs["_stacked"]["action_mask"]
                old_masks = mask_all.cpu().numpy()
                delta_masks = new_masks.sum(axis=-1) - old_masks.sum(axis=-1)
                for env_idx in range(args.num_envs):
                    for idx_a, a in enumerate(AGENTS):
                        if infos[env_idx].get(a, {}).get("car_unlocked", False):
                            bonus = 0.0
                            for idx_j in range(len(AGENTS)):
                                if idx_j != idx_a:
                                    bonus += max(
                                        0.0, float(delta_masks[idx_j, env_idx])
                                    )
                            rewards[a][env_idx] += args.car_coef * bonus

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
                # REV-3: store critic(terminal_state) for envs truncated here.
                with torch.no_grad():
                    t_idx = trunc_t.bool().nonzero(as_tuple=False).flatten()
                    if t_idx.numel():
                        t_states = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(i)]["terminal_state"], device=device
                                )
                                for i in t_idx
                            ]
                        )
                        val = policy.get_value(t_states).flatten()
                        buffers[a]["bootstrap"][step][t_idx] = val
            next_state = vec_env.state

        # ---------------- GAE + returns (centralized critic) ----------------
        with torch.no_grad():
            next_value = policy.get_value(
                torch.as_tensor(next_state, device=device)
            ).flatten()
            # REV-3: truncated envs bootstrap to critic(terminal_state), not 0.
            t_idx = next_truncations.bool().nonzero(as_tuple=False).flatten()
            if t_idx.numel():
                t_states = torch.stack(
                    [
                        torch.tensor(
                            last_infos[int(i)]["terminal_state"], device=device
                        )
                        for i in t_idx
                    ]
                )
                next_value[t_idx] = policy.get_value(t_states).flatten()
            # terminated envs bootstrap to 0
            next_value = next_value * (1.0 - next_terminations)
        stacked_advantages, stacked_returns = compute_gae(
            torch.stack([buffers[a]["rewards"] for a in AGENTS]),
            torch.stack([buffers[a]["values"] for a in AGENTS]),
            torch.stack([buffers[a]["terminated"] for a in AGENTS]),
            torch.stack([buffers[a]["truncated"] for a in AGENTS]),
            torch.stack([buffers[a]["bootstrap"] for a in AGENTS]),
            next_value.unsqueeze(0).expand(len(AGENTS), -1),
            next_terminations.unsqueeze(0),
            args.gamma,
            args.gae_lambda,
        )

        # --- CIR: Causal Influence Routing ---
        if getattr(args, "cir_coef", 0.0) > 0.0:
            adv_t = stacked_advantages.permute(
                1, 2, 0
            )  # [num_steps, num_envs, N_AGENTS]
            routed_adv = torch.einsum("sten,ste->stn", buf_influence, adv_t)
            adv_t_new = (1.0 - args.cir_coef) * adv_t + (args.cir_coef * routed_adv)
            stacked_advantages = adv_t_new.permute(2, 0, 1)
            stacked_returns = stacked_advantages + torch.stack(
                [buffers[a]["values"] for a in AGENTS]
            )

        advantages = {a: stacked_advantages[i] for i, a in enumerate(AGENTS)}
        returns = {a: stacked_returns[i] for i, a in enumerate(AGENTS)}

        # ---------------- policy update (concatenated across agents) ---------
        b_obs = torch.cat([buffers[a]["obs"].reshape(-1, obs_h, obs_w) for a in AGENTS])
        b_mask = torch.cat(
            [buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in AGENTS]
        )
        b_role = torch.cat(
            [buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in AGENTS]
        )
        b_actions = torch.cat([buffers[a]["actions"].reshape(-1) for a in AGENTS])
        b_logprobs = torch.cat([buffers[a]["logprobs"].reshape(-1) for a in AGENTS])
        b_advantages = torch.cat([advantages[a].reshape(-1) for a in AGENTS])
        b_returns = torch.cat([returns[a].reshape(-1) for a in AGENTS])
        # centralized critic sees the shared state, replicated per agent block
        # (each agent's flattened block shares the same (step, env) states)
        b_states = torch.cat([state_buffer.reshape(-1, state_dim) for _ in AGENTS])

        for _ in range(args.update_epochs):
            b_inds = torch.randperm(b_obs.shape[0], device=device)
            minibatch_size = b_obs.shape[0] // args.num_minibatches

            for start in range(0, b_obs.shape[0], minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb], b_role[mb], b_mask[mb], b_states[mb], b_actions[mb]
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
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                if rnd_module is not None:
                    rnd_module.update(b_obs[mb])

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---------------- logging ----------------
        sps = int(global_step / (time.time() - start_time))
        avg_reward = np.mean([buffers[a]["rewards"].mean().item() for a in AGENTS])
        win_rate = (
            (buffers[list(AGENTS)[0]]["rewards"] > 5.0).any(dim=0).float().mean().item()
            if hasattr(buffers[list(AGENTS)[0]]["rewards"], "dim")
            else 0.0
        )
        writer.add_scalar("charts/global_step", global_step, global_step)
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("charts/mean_reward", avg_reward, global_step)
        writer.add_scalar("charts/lr", optimizer.param_groups[0]["lr"], global_step)
        print(
            f"update={update} step={global_step} sps={sps} win_rate={win_rate:.3f} mean_reward={avg_reward:.4f}"
        )

        if update % args.eval_every == 0 or update == num_updates:
            if args.save_model:
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")
            eval_policies(global_step)

    vec_env.close()
    writer.close()
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")
        write_completion(
            run_name,
            "mappo",
            args.total_timesteps,
            num_updates * args.num_steps * args.num_envs,
        )
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    train(parse_args())
