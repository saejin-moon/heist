"""
COMA (Counterfactual Multi-Agent Policy Gradients) trainer for HEIST.

Implements the COMA algorithm (Foerster et al., AAAI 2018).
Centralized critic evaluates Q(s, a_{-i}, a_i) for all possible actions of
agent i given global state s and joint actions of all other agents a_{-i}.

Counterfactual advantage:
    A_i(s, a) = Q_i(s, a_{-i}, a_i) - sum_{a_i'} pi_i(a_i' | o_i) Q_i(s, a_{-i}, a_i')
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from constants import (
    ACTION_SPACE_SIZE as ACTION_DIM,
)
from constants import (
    N_AGENTS,
    OBSERVATION_SIZE,
)
from env import AGENTS, parse_env_config
from model import ComaAgent, ComaCritic
from ppo_utils import (
    compute_counterfactual_advantage,
    get_previous_stage_checkpoint,
    load_matching_weights,
    write_completion,
)
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "coma"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 300_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    num_minibatches: int = 4
    update_epochs: int = 4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    car_coef: float = 0.0
    eval_every: int = 20
    eval_episodes: int = 20
    env_config: str = ""

    cir_coef: float = 0.0
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
    p.add_argument("--total-steps", type=int, dest="total_timesteps")
    p.add_argument("--learning-rate", type=float, default=Args.learning_rate)
    p.add_argument("--num-envs", type=int, default=Args.num_envs)
    p.add_argument("--num-steps", type=int, default=Args.num_steps)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=Args.gamma)
    p.add_argument("--num-minibatches", type=int, default=Args.num_minibatches)
    p.add_argument("--update-epochs", type=int, default=Args.update_epochs)
    p.add_argument("--ent-coef", type=float, default=Args.ent_coef)
    p.add_argument("--vf-coef", type=float, default=Args.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=Args.max_grad_norm)
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


def construct_other_actions_onehot(actions_tensor, focal_agent_idx):
    """
    Args:
        actions_tensor: [N_AGENTS, B] integer actions
        focal_agent_idx: int index of focal agent i
    Returns:
        [B, (N_AGENTS - 1) * ACTION_DIM] float one-hot encoding of other agents' actions
    """
    other_list = []

    for k in range(N_AGENTS):
        if k != focal_agent_idx:
            oh = F.one_hot(actions_tensor[k], num_classes=ACTION_DIM).float()  # [B, 6]
            other_list.append(oh)
    return torch.cat(other_list, dim=-1)  # [B, 18]


def train(args: Args):  # noqa: C901

    import re

    run_name = (
        args.exp_name
        if re.search(r"_s\d+", args.exp_name)
        else f"{args.exp_name}_s{args.seed}"
    )

    ckpt_dir = Path("checkpoints") / run_name
    marker = ckpt_dir / "complete.json"
    if marker.is_file():
        try:
            data = json.loads(marker.read_text())
            if data.get("completed_steps", 0) >= args.total_timesteps:
                print(f"[COMA] Run '{run_name}' already complete. Skipping.")
                return
        except Exception:
            pass

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

    policy = ComaAgent(state_dim).to(device)

    target_critic = ComaCritic(state_dim).to(device)
    target_critic.load_state_dict(policy.critic.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    # Checkpoint loading / stage transfer / resuming
    policy_pt = ckpt_dir / "policy.pt"
    if policy_pt.is_file() and args.save_model:
        print(f"  [Resume] Loading existing checkpoint from {policy_pt}")
        load_matching_weights(policy, str(policy_pt), device)
        target_critic.load_state_dict(policy.critic.state_dict())
    else:
        prev_ckpt = get_previous_stage_checkpoint(run_name, args.exp_name)
        if prev_ckpt:
            print(f"  [Transfer] Loading previous stage checkpoint from {prev_ckpt}")
            load_matching_weights(policy, os.path.join(prev_ckpt, "policy.pt"), device)
            target_critic.load_state_dict(policy.critic.state_dict())

    if args.load_checkpoint:
        print(f"  [Transfer] Loading custom checkpoint from {args.load_checkpoint}")
        load_matching_weights(policy, f"{args.load_checkpoint}/policy.pt", device)
        target_critic.load_state_dict(policy.critic.state_dict())

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
            "bootstrap": torch.zeros((args.num_steps, args.num_envs), device=device),
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
            algo="coma",
            device=device,
        )
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step)
        print(
            f"  eval@{step}: win_rate={metrics['win_rate']:.3f} "
            f"return={metrics['mean_return']:.3f} len={metrics['mean_length']:.1f}"
        )

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
                writer.add_scalar(
                    "eval/importance/baseline_win_rate",
                    imp["baseline_win_rate"],
                    step,
                )
                for agent_name, val in imp["importance"].items():
                    writer.add_scalar(f"eval/importance/{agent_name}", val, step)

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
            stacked = next_obs["_stacked"]
            obs_all = torch.as_tensor(stacked["observation"], device=device)
            role_all = torch.as_tensor(stacked["role_id"], device=device)
            mask_all = torch.as_tensor(stacked["action_mask"], device=device)

            with torch.no_grad():
                action, logprob, _, _ = policy.get_action_and_probs(
                    obs_all.flatten(0, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                )
            actions = action.view(len(AGENTS), args.num_envs)
            logprobs = logprob.view(len(AGENTS), args.num_envs)
            if getattr(args, "cir_coef", 0.0) > 0.0:
                with torch.no_grad():
                    buf_influence[step] = policy.get_influence_matrix(state_t, actions)

            actions_dict = {}
            for i, a in enumerate(AGENTS):
                buffers[a]["obs"][step] = obs_all[i]
                buffers[a]["role_id"][step] = role_all[i]
                buffers[a]["action_mask"][step] = mask_all[i]
                buffers[a]["actions"][step] = actions[i]
                buffers[a]["logprobs"][step] = logprobs[i]
                actions_dict[a] = actions[i].cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = vec_env.step(
                actions_dict
            )

            # CAR intrinsic reward bonus
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

                # Bootstrap value for envs truncated here
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
                        # Average max Q across illegal-masked target critic
                        zero_oh = torch.zeros(
                            t_states.shape[0],
                            (N_AGENTS - 1) * ACTION_DIM,
                            device=device,
                        )
                        val = target_critic(t_states, zero_oh).max(dim=-1)[0]
                        buffers[a]["bootstrap"][step][t_idx] = val
            next_state = vec_env.state

        # ---------------- TD Targets for Critic ----------------
        # Compute Q-targets per agent for all timesteps
        targets = {
            a: torch.zeros((args.num_steps, args.num_envs), device=device)
            for a in AGENTS
        }

        with torch.no_grad():
            next_state_t = torch.as_tensor(next_state, device=device)
            stacked_next = next_obs["_stacked"]
            next_obs_all = torch.as_tensor(stacked_next["observation"], device=device)
            next_role_all = torch.as_tensor(stacked_next["role_id"], device=device)
            next_mask_all = torch.as_tensor(stacked_next["action_mask"], device=device)
            next_act, _, _, _ = policy.get_action_and_probs(
                next_obs_all.flatten(0, 1),
                next_role_all.flatten(0, 1),
                next_mask_all.flatten(0, 1),
            )
            next_actions_all = next_act.view(N_AGENTS, args.num_envs)

            for step in range(args.num_steps):
                if step < args.num_steps - 1:
                    s_next = state_buffer[step + 1]
                    step_actions_next = torch.stack(
                        [buffers[a]["actions"][step + 1] for a in AGENTS]
                    )
                else:
                    s_next = next_state_t
                    step_actions_next = next_actions_all

                for idx_a, a in enumerate(AGENTS):
                    other_oh_next = construct_other_actions_onehot(
                        step_actions_next, idx_a
                    )
                    q_next_all = target_critic(s_next, other_oh_next)
                    a_next = step_actions_next[idx_a]
                    q_next_taken = q_next_all.gather(1, a_next.unsqueeze(1)).squeeze(1)

                    r = buffers[a]["rewards"][step]
                    term = buffers[a]["terminated"][step]
                    trunc = buffers[a]["truncated"][step]
                    boot = buffers[a]["bootstrap"][step]

                    y = r + args.gamma * (1.0 - term) * (
                        (1.0 - trunc) * q_next_taken + trunc * boot
                    )
                    targets[a][step] = y

        # ---------------- COMA Policy & Critic Update ----------------
        b_states = state_buffer.reshape(-1, state_dim)  # [B, state_dim]
        B = b_states.shape[0]

        # Assemble per-agent flattened rollout tensors
        b_obs = {a: buffers[a]["obs"].reshape(-1, obs_h, obs_w) for a in AGENTS}
        b_role = {a: buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in AGENTS}
        b_mask = {a: buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in AGENTS}
        b_actions = {a: buffers[a]["actions"].reshape(-1) for a in AGENTS}
        b_targets = {a: targets[a].reshape(-1) for a in AGENTS}

        # Reshape actions for each agent step into tensor [N, B] to easily make one-hot
        all_b_actions = torch.stack([b_actions[a] for a in AGENTS])  # [N, B]

        b_other_oh = {}
        for idx_a, a in enumerate(AGENTS):
            b_other_oh[a] = construct_other_actions_onehot(all_b_actions, idx_a)

        # Precompute counterfactual advantages across batch
        b_adv = {}
        with torch.no_grad():
            for _idx_a, a in enumerate(AGENTS):
                q_vals_full = policy.get_value(b_states, b_other_oh[a])
                _, _, probs_full, _ = policy.get_action_and_probs(
                    b_obs[a], b_role[a], b_mask[a], action=b_actions[a]
                )
                adv_full, _ = compute_counterfactual_advantage(
                    probs_full, q_vals_full, b_actions[a], b_mask[a]
                )
                b_adv[a] = adv_full

            if getattr(args, "cir_coef", 0.0) > 0.0:
                adv_tensor = torch.stack(
                    [b_adv[a].view(args.num_steps, args.num_envs) for a in AGENTS],
                    dim=-1,
                )
                routed_adv = torch.einsum("sten,ste->stn", buf_influence, adv_tensor)
                adv_tensor_new = (
                    1.0 - args.cir_coef
                ) * adv_tensor + args.cir_coef * routed_adv
                for idx_a, a in enumerate(AGENTS):
                    b_adv[a] = adv_tensor_new[:, :, idx_a].reshape(-1)

        for _ in range(args.update_epochs):
            b_inds = torch.randperm(B, device=device)
            minibatch_size = B // args.num_minibatches

            for start in range(0, B, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                tot_actor_loss = 0.0
                tot_critic_loss = 0.0
                tot_entropy_loss = 0.0

                for _idx_a, a in enumerate(AGENTS):
                    obs_mb = b_obs[a][mb]
                    role_mb = b_role[a][mb]
                    mask_mb = b_mask[a][mb]
                    act_mb = b_actions[a][mb]
                    other_oh_mb = b_other_oh[a][mb]
                    state_mb = b_states[mb]
                    target_mb = b_targets[a][mb]

                    _, log_prob, probs, entropy = policy.get_action_and_probs(
                        obs_mb, role_mb, mask_mb, action=act_mb
                    )
                    q_vals_all = policy.get_value(state_mb, other_oh_mb)

                    adv_unrouted, q_taken = compute_counterfactual_advantage(
                        probs, q_vals_all, act_mb, mask_mb
                    )
                    adv = (
                        b_adv[a][mb]
                        if getattr(args, "cir_coef", 0.0) > 0.0
                        else adv_unrouted
                    )

                    actor_loss = -(log_prob * adv.detach()).mean()
                    critic_loss = F.mse_loss(q_taken, target_mb)

                    tot_actor_loss = tot_actor_loss + actor_loss
                    tot_critic_loss = tot_critic_loss + critic_loss
                    tot_entropy_loss = tot_entropy_loss + entropy.mean()

                total_loss = (
                    (tot_actor_loss / N_AGENTS)
                    + args.vf_coef * (tot_critic_loss / N_AGENTS)
                    - args.ent_coef * (tot_entropy_loss / N_AGENTS)
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                # Soft update target critic
                with torch.no_grad():
                    for p, tp in zip(
                        policy.critic.parameters(),
                        target_critic.parameters(),
                        strict=False,
                    ):
                        tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

                if rnd_module is not None:
                    rnd_module.update(torch.cat([b_obs[a][mb] for a in AGENTS], dim=0))

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
            "coma",
            args.total_timesteps,
            num_updates * args.num_steps * args.num_envs,
        )
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    train(parse_args())
