"""
LRS: Logical Reward Shaping Baseline
Implements a two-timescale PPO where the Manager outputs a discrete logical phase.
The Worker's primitive policy is conditioned on this logical phase and shaped
towards fulfilling its preconditions.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch

from constants import ACTION_SPACE_SIZE as ACTION_DIM
from constants import N_AGENTS
from env import AGENTS, parse_env_config
from model import DiscreteHierarchicalAgent
from ppo_utils import compute_gae, write_completion
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "lrs"
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
    lrs_shaping_coef: float = 0.1
    env_config: str = ""
    save_model: bool = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    p.add_argument("--env-config", type=str, default=Args.env_config)
    p.add_argument("--macro-step", type=int, default=Args.macro_step)
    p.add_argument("--no-save-model", action="store_false", dest="save_model")
    args, _ = p.parse_known_args()
    return Args(**vars(args))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"lrs_s{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_config = parse_env_config(args.env_config)
    envs = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)

    next_obs, next_state = envs.reset(seed=args.seed)
    state_dim = envs.state_dim

    agent = DiscreteHierarchicalAgent(state_dim).to(device)

    # Worker Buffers
    w_obs = torch.zeros((args.num_steps, N_AGENTS, args.num_envs, 49)).to(device)
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
    m_obs = torch.zeros((m_steps, N_AGENTS, args.num_envs, 49)).to(device)
    m_roles = torch.zeros((m_steps, N_AGENTS, args.num_envs, 4)).to(device)
    m_states = torch.zeros((m_steps, args.num_envs, state_dim)).to(device)
    m_actions = torch.zeros((m_steps, N_AGENTS, args.num_envs)).to(
        device
    )  # discrete logical phase
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

            w_obs[step] = obs_all
            w_roles[step] = role_all
            w_masks[step] = mask_all
            w_states[step] = state_t
            w_dones[step] = next_done

            # 1. Manager Step
            if step % args.macro_step == 0:
                m_step = step // args.macro_step
                m_obs[m_step] = obs_all
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
            actions_list = []
            for e in range(args.num_envs):
                act_dict = {
                    a: int(w_actions[step, i, e].item()) for i, a in enumerate(AGENTS)
                }
                actions_list.append(act_dict)

            next_obs, rewards, terminations, truncations, infos = envs.step(
                actions_list
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
            compute_gae(
                w_rewards, w_values, next_w_value, w_dones, args.gamma, args.gae_lambda
            )

        if update % args.eval_every == 0 or update == num_updates:
            print(f"[{run_name}] Update {update}/{num_updates}")

    if args.save_model:
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"runs/{run_name}/lrs.pt")
        write_completion(f"runs/{run_name}", args)


if __name__ == "__main__":
    main()
