"""
TarMAC communication trainer for HEIST (REV-7, Phase B/C).

Uses a shared CommAgent with differentiable TarMAC message passing.
All agents share the same encoder + communication + policy, with role
one-hot distinguishing them.

Design:
  - Rollout collects joint transitions: at each step, all agents'
    observations are passed through the shared CommAgent in one batch,
    enabling TarMAC message passing before action selection.
  - GAE is computed per-agent (like IPPO) since rewards and values are
    per-agent.
  - The emergent-language diagnostic (Phase C) is logged periodically.

Example:
    uv run python src/train_comm.py --total-steps 2048
"""

import argparse
import os
import time

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
from model import CommAgent
from ppo_utils import compute_gae, write_completion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default="comm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--anneal-lr", action="store_true")
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=2048)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--env-config", type=str, default="")
    p.add_argument(
        "--centralized",
        action="store_true",
        help="MAPPO-style centralized critic: value estimates come from "
        "env.state() (shared across agents) instead of local obs+messages. "
        "Actor still uses TarMAC messages (FUTURE_PLANS.md \u00a75).",
    )
    p.add_argument(
        "--comm-diagnostic-every",
        type=int,
        default=4096,
        help="Run message-outcome correlation diagnostic every N steps.",
    )
    args = p.parse_args()
    if args.num_minibatches < 1:
        p.error("--num-minibatches must be positive")
    if args.num_steps * args.num_envs % args.num_minibatches:
        p.error("num-envs * num-steps must be divisible by --num-minibatches")
    return args


def _stack_obs(next_obs, agent_list=AGENTS):
    """Convert per-agent dicts to the lists CommAgent expects.

    next_obs[a] arrays are [num_envs, H, W] / [num_envs, A] etc.; CommAgent
    flattens with start_dim=1 so they map directly to [B, 25] per agent.
    """
    stacked = next_obs["_stacked"]
    obs = torch.as_tensor(stacked["observation"], device=device)
    role = torch.as_tensor(stacked["role_id"], device=device)
    mask = torch.as_tensor(stacked["action_mask"], device=device)
    obs_list = [obs[i] for i in range(len(agent_list))]
    role_list = [role[i] for i in range(len(agent_list))]
    mask_list = [mask[i] for i in range(len(agent_list))]
    return obs_list, role_list, mask_list


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}_s{args.seed}"
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()),
    )
    print(f"device: {device} run={run_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_config = parse_env_config(args.env_config)
    from vec_env import VectorEnv

    vec_env = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)
    next_obs, next_state = vec_env.reset(seed=args.seed)
    next_terminations = torch.zeros(args.num_envs, device=device)
    next_truncations = torch.zeros(args.num_envs, device=device)
    last_infos = [{} for _ in range(args.num_envs)]

    state_dim = vec_env.state_dim
    policy = CommAgent(state_dim=state_dim, centralized=args.centralized).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    obs_h, obs_w = OBSERVATION_SIZE
    n = len(AGENTS)
    # Joint buffers: all agents stored together at each (step, env)
    buf_obs = torch.zeros(
        (args.num_steps, args.num_envs, n, obs_h, obs_w), device=device
    )
    buf_role = torch.zeros((args.num_steps, args.num_envs, n, N_AGENTS), device=device)
    buf_mask = torch.zeros(
        (args.num_steps, args.num_envs, n, ACTION_DIM), device=device
    )
    buf_actions = torch.zeros(
        (args.num_steps, args.num_envs, n), dtype=torch.long, device=device
    )
    buf_logprobs = torch.zeros((args.num_steps, args.num_envs, n), device=device)
    buf_rewards = torch.zeros((args.num_steps, args.num_envs, n), device=device)
    buf_values = torch.zeros((args.num_steps, args.num_envs, n), device=device)
    buf_terminated = torch.zeros((args.num_steps, args.num_envs), device=device)
    buf_truncated = torch.zeros((args.num_steps, args.num_envs), device=device)
    buf_bootstrap = torch.zeros((args.num_steps, args.num_envs, n), device=device)
    # centralized critic consumes env.state() at every (step, env)
    buf_state = torch.zeros((args.num_steps, args.num_envs, state_dim), device=device)

    num_updates = args.total_steps // (args.num_steps * args.num_envs)
    print(f"num_updates: {num_updates}")

    start_time = time.time()
    global_step = 0

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            for g in optimizer.param_groups:
                g["lr"] = frac * args.learning_rate

        # ---------------- rollout ----------------
        for step in range(args.num_steps):
            global_step += 1
            state_t = torch.as_tensor(next_state, device=device)
            if args.centralized:
                buf_state[step] = state_t
            obs_list, role_list, mask_list = _stack_obs(next_obs)
            with torch.no_grad():
                actions, logprobs, _, values = policy.get_action_and_value(
                    obs_list,
                    role_list,
                    mask_list,
                    state=(state_t if args.centralized else None),
                )
            # actions/logprobs/values: [num_envs, n_agents]

            for i, _a in enumerate(AGENTS):
                buf_obs[step, :, i] = obs_list[i]
                buf_role[step, :, i] = role_list[i]
                buf_mask[step, :, i] = mask_list[i]
                buf_actions[step, :, i] = actions[:, i]
                buf_logprobs[step, :, i] = logprobs[:, i]
                buf_values[step, :, i] = values[:, i]

            actions_dict = {
                a: actions[:, i].cpu().numpy().astype(np.int64)
                for i, a in enumerate(AGENTS)
            }
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

            for i, a in enumerate(AGENTS):
                buf_rewards[step, :, i] = torch.as_tensor(rewards[a], device=device)
            buf_terminated[step] = torch.as_tensor(
                terminations["scout"], device=device
            ).float()
            buf_truncated[step] = torch.as_tensor(
                truncations["scout"], device=device
            ).float()

            # bootstrap values for truncated envs (REV-3: truncations bootstrap
            # to V(terminal_observation), terminations to 0).
            t_idx = buf_truncated[step].bool().nonzero(as_tuple=False).flatten()
            if t_idx.numel():
                with torch.no_grad():
                    if args.centralized:
                        # MAPPO-style: bootstrap from critic(terminal_state)
                        t_states = torch.stack(
                            [
                                torch.tensor(
                                    infos[int(env_i)]["terminal_state"], device=device
                                )
                                for env_i in t_idx.cpu().tolist()
                            ]
                        )
                        bv = policy.critic(t_states).expand(-1, n).clone()
                        buf_bootstrap[step][t_idx] = bv
                    else:
                        for env_i in t_idx.cpu().tolist():
                            term_obs = infos[int(env_i)]["terminal_observation"]
                            obs_t = [
                                torch.tensor(term_obs[a]["observation"], device=device)
                                for a in AGENTS
                            ]
                            role_t = [
                                torch.tensor(term_obs[a]["role_id"], device=device)
                                for a in AGENTS
                            ]
                            mask_t = [
                                torch.tensor(term_obs[a]["action_mask"], device=device)
                                for a in AGENTS
                            ]
                            _, _, _, bv = policy.get_action_and_value(
                                obs_t, role_t, mask_t
                            )
                            buf_bootstrap[step, env_i] = bv[0]

        # ---------------- GAE + returns ----------------
        with torch.no_grad():
            obs_list, role_list, mask_list = _stack_obs(next_obs)
            next_state_t = torch.as_tensor(next_state, device=device)
            _, _, _, next_values = policy.get_action_and_value(
                obs_list,
                role_list,
                mask_list,
                state=(next_state_t if args.centralized else None),
            )  # [num_envs, n]

            # bootstrap for truncated envs at rollout end
            t_idx = next_truncations.bool().nonzero(as_tuple=False).flatten().tolist()
            if args.centralized and t_idx:
                t_states = torch.stack(
                    [
                        torch.tensor(
                            last_infos[int(env_i)]["terminal_state"], device=device
                        )
                        for env_i in t_idx
                    ]
                )
                next_values[t_idx] = policy.critic(t_states).expand(-1, n).clone()
            elif not args.centralized:
                for env_i in t_idx:
                    term_obs = last_infos[int(env_i)]["terminal_observation"]
                    obs_t = [
                        torch.tensor(term_obs[a]["observation"], device=device)
                        for a in AGENTS
                    ]
                    role_t = [
                        torch.tensor(term_obs[a]["role_id"], device=device)
                        for a in AGENTS
                    ]
                    mask_t = [
                        torch.tensor(term_obs[a]["action_mask"], device=device)
                        for a in AGENTS
                    ]
                    _, _, _, bv = policy.get_action_and_value(obs_t, role_t, mask_t)
                    next_values[env_i] = bv[0]  # [n], on device

            # terminated envs bootstrap to 0
            next_values = next_values * (1.0 - next_terminations).unsqueeze(1)

        stacked_advantages, stacked_returns = compute_gae(
            buf_rewards.permute(2, 0, 1),
            buf_values.permute(2, 0, 1),
            buf_terminated.unsqueeze(0).expand(n, -1, -1),
            buf_truncated.unsqueeze(0).expand(n, -1, -1),
            buf_bootstrap.permute(2, 0, 1),
            next_values.transpose(0, 1),
            next_terminations.unsqueeze(0),
            args.gamma,
            args.gae_lambda,
        )

        # ---------------- policy update ----------------
        # flatten to (T * num_envs * n_agents, ...) for minibatch SGD
        B = args.num_steps * args.num_envs
        flat_obs = buf_obs.reshape(B, n, obs_h, obs_w)
        flat_role = buf_role.reshape(B, n, N_AGENTS)
        flat_mask = buf_mask.reshape(B, n, ACTION_DIM)
        flat_actions = buf_actions.reshape(B, n)
        flat_logprobs = buf_logprobs.reshape(B, n)
        flat_adv = stacked_advantages.permute(1, 2, 0).reshape(B, n)
        flat_returns = stacked_returns.permute(1, 2, 0).reshape(B, n)
        # centralized critic consumes one state per (step, env), aligned with B
        flat_state = buf_state.reshape(B, state_dim)

        # all agents in a (step, env) share the same transition, so we can
        # reshape to (B * n, ...) or (B, n, ...) and process jointly.  The
        # CommAgent expects [batch, n_agents, ...], so keep B-dim and process
        # (B, n, ...) in minibatches.

        minibatch_size = B // args.num_minibatches

        for _ in range(args.update_epochs):
            b_inds = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                mb_obs = flat_obs[mb]  # [mb, n, H, W]
                mb_role = flat_role[mb]  # [mb, n, N]
                mb_mask = flat_mask[mb]  # [mb, n, A]
                mb_actions = flat_actions[mb]  # [mb, n]
                mb_logprobs_old = flat_logprobs[mb]  # [mb, n]

                # stack into lists for CommAgent.get_action_and_value
                obs_list_mb = [mb_obs[:, i] for i in range(n)]
                role_list_mb = [mb_role[:, i] for i in range(n)]
                mask_list_mb = [mb_mask[:, i] for i in range(n)]

                mb_state = flat_state[mb] if args.centralized else None

                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    obs_list_mb,
                    role_list_mb,
                    mask_list_mb,
                    state=mb_state,
                    action=mb_actions,
                )

                logratio = newlogprob - mb_logprobs_old
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

                adv_mb = flat_adv[mb]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = ((newvalue - flat_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---------------- logging ----------------
        sps = int(global_step / (time.time() - start_time))
        avg_reward = buf_rewards.mean().item()
        writer.add_scalar("charts/sps", sps, global_step)
        writer.add_scalar("charts/mean_reward", avg_reward, global_step)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfrac.item(), global_step)

        if update % 1 == 0:
            print(
                f"update={update} step={global_step} sps={sps} "
                f"mean_reward={avg_reward:.4f}"
            )

        # ---------------- periodic eval + diagnostics ----------------
        if update % max(args.eval_every // (args.num_steps * args.num_envs), 1) == 0:
            if args.save_model:
                os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
                torch.save(policy.state_dict(), f"checkpoints/{run_name}/comm.pt")
            from evaluate import evaluate_comm_policies

            metrics = evaluate_comm_policies(
                policy,
                vec_env.envs[0],
                episodes=args.eval_episodes,
                seed=args.seed + 1_000_000,
                device=device,
            )
            for k, v in metrics.items():
                writer.add_scalar(f"eval/{k}", v, global_step)
            print(
                f"  eval@{global_step}: win_rate={metrics['win_rate']:.3f} "
                f"return={metrics['mean_return']:.3f} "
                f"len={metrics['mean_length']:.1f}"
            )

        # Phase C message-outcome correlation diagnostic
        if (
            args.comm_diagnostic_every > 0
            and global_step % args.comm_diagnostic_every == 0
        ):
            from evaluate import message_outcome_correlation

            diag = message_outcome_correlation(
                policy,
                vec_env.envs[0],
                episodes=10,
                seed=args.seed + 2_000_000,
                device=device,
            )
            writer.add_scalar(
                "diag/max_terminal_corr", diag["max_terminal_message_corr"], global_step
            )
            writer.add_scalar(
                "diag/mean_terminal_corr",
                diag["mean_terminal_message_corr"],
                global_step,
            )
            # log mean attention weight matrix
            attn = diag["mean_attention"]
            for i, a in enumerate(AGENTS):
                for j, b in enumerate(AGENTS):
                    writer.add_scalar(
                        f"attn/{a}_to_{b}", float(attn[i, j]), global_step
                    )
            print(
                f"  diag@{global_step}: max_terminal_corr="
                f"{diag['max_terminal_message_corr']:.4f} "
                f"mean_terminal_corr="
                f"{diag['mean_terminal_message_corr']:.4f}"
            )

    vec_env.close()
    writer.close()
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(policy.state_dict(), f"checkpoints/{run_name}/comm.pt")
        write_completion(
            run_name,
            "comm",
            args.total_steps,
            num_updates * args.num_steps * args.num_envs,
        )
    print("training done.")
