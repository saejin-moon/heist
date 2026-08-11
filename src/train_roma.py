"""
ROMA: Role-Oriented Multi-Agent RL Baseline
Implements a two-timescale PPO where the Manager sets discrete roles (1 of K).
The Worker's primitive policy is conditioned on this latent role.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch

from constants import ACTION_SPACE_SIZE as ACTION_DIM
from constants import N_AGENTS, OBSERVATION_SIZE
from env import AGENTS, parse_env_config
from model import DiscreteHierarchicalAgent
from ppo_utils import compute_gae_simple, write_completion
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "roma"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 250
    macro_step: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    roma_kl_coef: float = 0.05
    env_config: str = ""
    save_model: bool = True
    eval_every: int = 10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    p.add_argument("--env-config", type=str, default=Args.env_config)
    p.add_argument("--macro-step", type=int, default=Args.macro_step)
    p.add_argument("--eval-every", type=int, default=Args.eval_every)
    p.add_argument("--no-save-model", action="store_false", dest="save_model")
    args, _ = p.parse_known_args()
    return Args(**vars(args))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"roma_s{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_config = parse_env_config(args.env_config)
    envs = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)

    next_obs, next_state = envs.reset(seed=args.seed)
    state_dim = envs.state_dim

    agent = DiscreteHierarchicalAgent(state_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    from ppo_utils import get_previous_stage_checkpoint, load_matching_weights
    prev_ckpt = get_previous_stage_checkpoint(run_name, args.exp_name)
    if prev_ckpt:
        print(f"  [Transfer] Loading previous stage checkpoint from {prev_ckpt}")
        load_matching_weights(agent, os.path.join(prev_ckpt, "roma.pt"), device)

    # Worker Buffers
    w_obs = torch.zeros(
        (
            args.num_steps,
            N_AGENTS,
            args.num_envs,
            OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1],
        )
    ).to(device)
    w_roles = torch.zeros((args.num_steps, N_AGENTS, args.num_envs, 4)).to(device)
    w_states = torch.zeros((args.num_steps, args.num_envs, state_dim)).to(device)
    w_masks = torch.zeros((args.num_steps, N_AGENTS, args.num_envs, ACTION_DIM)).to(
        device
    )
    w_actions = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    w_logprobs = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    w_rewards = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    w_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    w_values = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)

    # Manager Buffers
    m_steps = args.num_steps // args.macro_step
    m_obs = torch.zeros(
        (m_steps, N_AGENTS, args.num_envs, OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1])
    ).to(device)
    m_roles = torch.zeros((m_steps, N_AGENTS, args.num_envs, 4)).to(device)
    m_states = torch.zeros((m_steps, args.num_envs, state_dim)).to(device)
    m_actions = torch.zeros((m_steps, N_AGENTS, args.num_envs)).to(
        device
    )  # discrete roles
    m_logprobs = torch.zeros((m_steps, N_AGENTS, args.num_envs)).to(device)
    m_rewards = torch.zeros((m_steps, N_AGENTS, args.num_envs)).to(device)
    m_dones = torch.zeros((m_steps, args.num_envs)).to(device)
    m_values = torch.zeros((m_steps, N_AGENTS, args.num_envs)).to(device)

    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)

    current_goals = None
    next_done = torch.zeros(args.num_envs).to(device)

    for update in range(1, num_updates + 1):
        macro_reward_acc = torch.zeros((N_AGENTS, args.num_envs)).to(device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs

            state_t = torch.as_tensor(next_state, dtype=torch.float32, device=device)
            stacked = next_obs["_stacked"]
            obs_all = torch.as_tensor(
                stacked["observation"], dtype=torch.float32, device=device
            )
            role_all = torch.as_tensor(
                stacked["role_id"], dtype=torch.float32, device=device
            )
            mask_all = torch.as_tensor(
                stacked["action_mask"], dtype=torch.float32, device=device
            )

            w_obs[step] = obs_all.flatten(2)
            w_roles[step] = role_all
            w_masks[step] = mask_all
            w_states[step] = state_t
            w_dones[step] = next_done

            # 1. Manager Step
            if step % args.macro_step == 0:
                m_step = step // args.macro_step
                m_obs[m_step] = obs_all.flatten(2)
                m_roles[m_step] = role_all
                m_states[m_step] = state_t
                m_dones[m_step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_manager_action_and_value(
                        obs_all.flatten(0, 1),
                        state_t.repeat(N_AGENTS, 1),
                        role_all.flatten(0, 1),
                    )
                    current_goals = action.view(N_AGENTS, args.num_envs)
                    m_actions[m_step] = current_goals
                    m_logprobs[m_step] = logprob.view(N_AGENTS, args.num_envs)
                    m_values[m_step] = value.view(N_AGENTS, args.num_envs)

                    if m_step > 0:
                        m_rewards[m_step - 1] = macro_reward_acc.clone()
                    macro_reward_acc.zero_()

            # 2. Worker Step
            with torch.no_grad():
                g = current_goals.view(N_AGENTS * args.num_envs).long()
                # ROMA conditions the worker on the one-hot encoded role.
                # Here DiscreteHierarchicalAgent does an internal embedding lookup.
                w_action, w_logprob, _, w_value = agent.get_worker_action_and_value(
                    obs_all.flatten(0, 1),
                    state_t.repeat(N_AGENTS, 1),
                    role_all.flatten(0, 1),
                    g,
                    mask_all.flatten(0, 1),
                )
                w_actions[step] = w_action.view(N_AGENTS, args.num_envs)
                w_logprobs[step] = w_logprob.view(N_AGENTS, args.num_envs)
                w_values[step] = w_value.view(N_AGENTS, args.num_envs)

            # 3. Environment Step
            actions_dict = {
                a: w_actions[step, i].cpu().numpy() for i, a in enumerate(AGENTS)
            }

            next_obs, rewards, terminations, truncations, infos = envs.step(
                actions_dict
            )
            next_state = envs.state
            next_done = torch.logical_or(
                torch.tensor(terminations, dtype=torch.float32),
                torch.tensor(truncations, dtype=torch.float32),
            ).to(device)

            for e in range(args.num_envs):
                for i, a in enumerate(AGENTS):
                    base_reward = rewards[e][a]
                    w_rewards[step, i, e] = base_reward
                    macro_reward_acc[i, e] += base_reward

        m_rewards[-1] = macro_reward_acc.clone()

        with torch.no_grad():
            g = current_goals.view(N_AGENTS * args.num_envs).long()
            next_obs_all = torch.as_tensor(
                next_obs["_stacked"]["observation"], dtype=torch.float32, device=device
            )
            next_role_all = torch.as_tensor(
                next_obs["_stacked"]["role_id"], dtype=torch.float32, device=device
            )
            next_state_t = torch.as_tensor(
                next_state, dtype=torch.float32, device=device
            )
            _, _, _, next_w_value = agent.get_worker_action_and_value(
                next_obs_all.flatten(0, 1),
                next_state_t.repeat(N_AGENTS, 1),
                next_role_all.flatten(0, 1),
                g,
            )
            next_w_value = next_w_value.view(N_AGENTS, args.num_envs)
            w_advantages, w_returns = compute_gae_simple(
                w_rewards, w_values, next_w_value, w_dones, args.gamma, args.gae_lambda
            )

            _, _, _, next_m_value = agent.get_manager_action_and_value(
                next_obs_all.flatten(0, 1),
                next_state_t.repeat(N_AGENTS, 1),
                next_role_all.flatten(0, 1),
            )

            next_m_value = next_m_value.view(N_AGENTS, args.num_envs)

            m_advantages, m_returns = compute_gae_simple(
                m_rewards, m_values, next_m_value, m_dones, args.gamma, args.gae_lambda
            )

            # Flatten Worker Batch

            b_w_obs = w_obs.flatten(0, 2)

            b_w_roles = w_roles.flatten(0, 2)

            b_w_states = w_states.unsqueeze(1).repeat(1, N_AGENTS, 1, 1).flatten(0, 2)

            b_w_masks = w_masks.flatten(0, 2)

            b_w_actions = w_actions.flatten(0, 2)

            b_w_logprobs = w_logprobs.flatten(0, 2)

            b_w_advantages = w_advantages.flatten(0, 2)

            b_w_returns = w_returns.flatten(0, 2)

            b_current_goals = (
                m_actions.repeat_interleave(args.macro_step, dim=0).flatten(0, 2).long()
            )

            # Flatten Manager Batch

            b_m_obs = m_obs.flatten(0, 2)

            b_m_roles = m_roles.flatten(0, 2)

            b_m_states = m_states.unsqueeze(1).repeat(1, N_AGENTS, 1, 1).flatten(0, 2)

            b_m_actions = m_actions.flatten(0, 2)

            b_m_logprobs = m_logprobs.flatten(0, 2)

            b_m_advantages = m_advantages.flatten(0, 2)

            b_m_returns = m_returns.flatten(0, 2)

            # Worker Optimization

            w_inds = np.arange(args.num_envs * args.num_steps * N_AGENTS)

            w_minibatch_size = len(w_inds) // 4

            for _ in range(4):
                np.random.shuffle(w_inds)

                for start in range(0, len(w_inds), w_minibatch_size):
                    end = start + w_minibatch_size

                    mb = w_inds[start:end]

                    _, newlogprob, entropy, newvalue = (
                        agent.get_worker_action_and_value(
                            b_w_obs[mb],
                            b_w_states[mb],
                            b_w_roles[mb],
                            b_current_goals[mb],
                            b_w_masks[mb],
                            action=b_w_actions[mb],
                        )
                    )

                    logratio = newlogprob - b_w_logprobs[mb]

                    ratio = logratio.exp()

                    mb_adv = b_w_advantages[mb]

                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss = torch.max(
                        -mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                    ).mean()

                    v_loss = 0.5 * ((newvalue - b_w_returns[mb]) ** 2).mean()

                    loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                    optimizer.zero_grad()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        agent.parameters(), args.max_grad_norm
                    )

                    optimizer.step()

            # Manager Optimization

            m_inds = np.arange(
                args.num_envs * (args.num_steps // args.macro_step) * N_AGENTS
            )

            if len(m_inds) > 0:
                m_minibatch_size = max(1, len(m_inds) // 4)

                for _ in range(4):
                    np.random.shuffle(m_inds)

                    for start in range(0, len(m_inds), m_minibatch_size):
                        end = start + m_minibatch_size

                        mb = m_inds[start:end]

                        _, newlogprob, entropy, newvalue = (
                            agent.get_manager_action_and_value(
                                b_m_obs[mb],
                                b_m_states[mb],
                                b_m_roles[mb],
                                action=b_m_actions[mb],
                            )
                        )

                        logratio = newlogprob - b_m_logprobs[mb]

                        ratio = logratio.exp()

                        mb_adv = b_m_advantages[mb]

                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        pg_loss = torch.max(
                            -mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                        ).mean()

                        v_loss = 0.5 * ((newvalue - b_m_returns[mb]) ** 2).mean()

                        loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                        optimizer.zero_grad()

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(
                            agent.parameters(), args.max_grad_norm
                        )

                        optimizer.step()

        if update % args.eval_every == 0 or update == num_updates:
            print(f"[{run_name}] Update {update}/{num_updates}")

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"runs/{run_name}/roma.pt")
        write_completion(f"runs/{run_name}", args)


if __name__ == "__main__":
    main()
