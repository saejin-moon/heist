"""
CO-OP: Scalable Cooperative Option-Pool Evolution.
Implements a dynamic Bottom-Up Confidence Routing mechanism.
"""

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from constants import ACTION_SPACE_SIZE as ACTION_DIM
from constants import N_AGENTS, OBSERVATION_SIZE
from env import AGENTS, parse_env_config
from model import CoopAgent, CoopTopDownAgent
from ppo_utils import compute_gae_simple, write_completion
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "coop"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 2_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 250
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    tau_spawn: float = -0.1
    max_experts: int = 6
    burn_in_steps: int = 500_000
    complexity_penalty: float = 0.01
    prune_episodes: int = 50
    env_config: str = ""
    save_model: bool = True
    load_checkpoint: str = ""
    ablation_fixed: bool = False
    ablation_no_car: bool = False
    ablation_top_down: bool = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=Args.exp_name)
    p.add_argument("--seed", type=int, default=Args.seed)
    p.add_argument("--total-timesteps", type=int, default=Args.total_timesteps)
    p.add_argument("--env-config", type=str, default=Args.env_config)
    p.add_argument("--no-save-model", action="store_false", dest="save_model")
    p.add_argument("--load-checkpoint", type=str, default="")
    p.add_argument("--ablation-fixed", action="store_true")
    p.add_argument("--ablation-no-car", action="store_true")
    p.add_argument("--ablation-top-down", action="store_true")
    args, _ = p.parse_known_args()
    return Args(**vars(args))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"{args.exp_name}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    env_config = parse_env_config(args.env_config)
    envs = VectorEnv(args.num_envs, config=env_config, base_seed=args.seed)

    next_obs, next_state = envs.reset(seed=args.seed)
    state_dim = envs.state_dim

    if args.ablation_top_down:
        agent = CoopTopDownAgent(state_dim, max_experts=args.max_experts).to(device)
    else:
        agent = CoopAgent(state_dim, max_experts=args.max_experts).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.load_checkpoint and os.path.exists(f"{args.load_checkpoint}/coop.pt"):
        agent.load_state_dict(
            torch.load(f"{args.load_checkpoint}/coop.pt", map_location=device)
        )
        print(f"Loaded checkpoint from {args.load_checkpoint}")

    active_experts = (
        args.max_experts if args.ablation_fixed else 2
    )  # Start with 2 experts unless fixed

    # Buffers
    obs_t = torch.zeros(
        (
            args.num_steps,
            N_AGENTS,
            args.num_envs,
            OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1],
        )
    ).to(device)
    roles_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs, 4)).to(device)
    states_t = torch.zeros((args.num_steps, args.num_envs, state_dim)).to(device)
    masks_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs, ACTION_DIM)).to(
        device
    )

    actions_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    logprobs_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    rewards_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    dones_t = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_t = torch.zeros((args.num_steps, N_AGENTS, args.num_envs)).to(device)
    expert_idx_t = torch.zeros(
        (args.num_steps, N_AGENTS, args.num_envs), dtype=torch.long
    ).to(device)
    alarms_t = torch.zeros((args.num_steps, args.num_envs)).to(device)
    car_unlocked_t = torch.zeros(
        (args.num_steps, N_AGENTS, args.num_envs), dtype=torch.bool
    ).to(device)

    expert_last_used_episode = torch.zeros(
        args.max_experts, dtype=torch.long, device=device
    )

    global_step = 0
    global_episode_count = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)

    next_done = torch.zeros(args.num_envs).to(device)
    start_time = time.time()

    for update in range(1, num_updates + 1):
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

            obs_t[step] = obs_all.flatten(2)
            roles_t[step] = role_all
            masks_t[step] = mask_all
            states_t[step] = state_t
            dones_t[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, chosen_expert = agent.get_action_and_value(
                    obs_all.flatten(0, 1),
                    state_t.repeat(N_AGENTS, 1),
                    role_all.flatten(0, 1),
                    mask_all.flatten(0, 1),
                    active_experts,
                )

                # Check for spawning
                max_conf = torch.max(value)
                if (
                    max_conf < args.tau_spawn
                    and active_experts < args.max_experts
                    and not args.ablation_fixed
                    and global_step >= args.burn_in_steps
                ):
                    active_experts += 1
                    print(
                        f"Spawning Expert {active_experts} at step {global_step} (min_conf={max_conf.item():.3f})!"
                    )

                actions_t[step] = action.view(N_AGENTS, args.num_envs)
                logprobs_t[step] = logprob.view(N_AGENTS, args.num_envs)
                values_t[step] = value.view(N_AGENTS, args.num_envs)
                expert_idx_t[step] = chosen_expert.view(N_AGENTS, args.num_envs)

                # Update expert usage and check for pruning
                chosen_expert_unique = chosen_expert.unique()
                expert_last_used_episode[chosen_expert_unique] = global_episode_count

                if not args.ablation_fixed:
                    for k in range(active_experts - 1, -1, -1):
                        if (
                            active_experts > 1
                            and (global_episode_count - expert_last_used_episode[k])
                            > args.prune_episodes
                        ):
                            print(
                                f"Pruning Expert {k} at episode {global_episode_count} (unused for {global_episode_count - expert_last_used_episode[k]} episodes)!"
                            )

                            if k < active_experts - 1:
                                # Shift weights and optimizer state to keep active experts contiguous
                                src = active_experts - 1
                                agent.experts[k].load_state_dict(
                                    agent.experts[src].state_dict()
                                )

                                for p_target, p_source in zip(
                                    agent.experts[k].parameters(),
                                    agent.experts[src].parameters(),
                                    strict=False,
                                ):
                                    if p_source in optimizer.state:
                                        state_src = optimizer.state[p_source]
                                        optimizer.state[p_target] = {
                                            "step": state_src.get(
                                                "step", torch.tensor(0.0)
                                            ),
                                            "exp_avg": state_src["exp_avg"].clone(),
                                            "exp_avg_sq": state_src[
                                                "exp_avg_sq"
                                            ].clone(),
                                        }
                                        if "max_exp_avg_sq" in state_src:
                                            optimizer.state[p_target][
                                                "max_exp_avg_sq"
                                            ] = state_src["max_exp_avg_sq"].clone()
                                    else:
                                        if p_target in optimizer.state:
                                            del optimizer.state[p_target]

                                expert_last_used_episode[k] = expert_last_used_episode[
                                    src
                                ]
                                expert_idx_t[expert_idx_t == src] = k

                            active_experts -= 1

            actions_list = []
            for e in range(args.num_envs):
                act_dict = {
                    a: int(actions_t[step, i, e].item()) for i, a in enumerate(AGENTS)
                }
                actions_list.append(act_dict)

            # Structural Affordance Detection (Pre-step state)
            old_mask_sum = mask_all.view(args.num_envs, N_AGENTS, ACTION_DIM).sum(
                dim=(1, 2)
            )

            next_obs, rewards, terminations, truncations, infos = envs.step(
                actions_list
            )

            alarms_step = [
                infos[e].get("scout", {}).get("alarm", 0.0)
                if isinstance(infos, list) and e < len(infos)
                else 0.0
                for e in range(args.num_envs)
            ]
            alarms_t[step] = torch.tensor(
                alarms_step, dtype=torch.float32, device=device
            )

            next_state = envs.state
            next_done = torch.logical_or(
                torch.tensor(terminations, dtype=torch.float32),
                torch.tensor(truncations, dtype=torch.float32),
            ).to(device)
            global_episode_count += int(next_done.sum().item())

            # Structural Affordance Detection (Post-step state)
            new_mask_all = torch.as_tensor(
                next_obs["_stacked"]["action_mask"], dtype=torch.float32, device=device
            )
            new_mask_sum = new_mask_all.view(args.num_envs, N_AGENTS, ACTION_DIM).sum(
                dim=(1, 2)
            )
            affordance_triggered = new_mask_sum > old_mask_sum  # [num_envs]

            # Assign CAR structurally
            for e in range(args.num_envs):
                for i, a in enumerate(AGENTS):
                    base_reward = rewards[e][a]
                    # The affordance was triggered globally, and THIS agent took the INTERACT action (5)
                    is_unlocked = (
                        affordance_triggered[e].item()
                        and int(actions_t[step, i, e].item()) == 5
                    )
                    car_unlocked_t[step, i, e] = is_unlocked
                    car_reward = 0.0
                    if is_unlocked and not args.ablation_no_car:
                        car_reward = 1.0  # Dense causal affordance routing reward to the winning expert
                    rewards_t[step, i, e] = (
                        base_reward
                        + car_reward
                        - (args.complexity_penalty * active_experts)
                    )

        with torch.no_grad():
            next_obs_all = torch.as_tensor(
                next_obs["_stacked"]["observation"], dtype=torch.float32, device=device
            )
            next_role_all = torch.as_tensor(
                next_obs["_stacked"]["role_id"], dtype=torch.float32, device=device
            )
            next_state_t = torch.as_tensor(
                next_state, dtype=torch.float32, device=device
            )

            _, _, _, next_value, _ = agent.get_action_and_value(
                next_obs_all.flatten(0, 1),
                next_state_t.repeat(N_AGENTS, 1),
                next_role_all.flatten(0, 1),
                mask_all.flatten(0, 1),
                active_experts,
            )
            next_value = next_value.view(N_AGENTS, args.num_envs)

            # Apply Asymmetric Failure Shielding & Macro Weighting from MARC
            win = (rewards_t > 5.0).any(dim=1).any(dim=0)  # [B]
            macro_alarm_factor = torch.exp(-1.5 * (alarms_t / 100.0))  # [T, B]
            macro_outcome = torch.where(win, 1.0, -0.5)  # [B]

            alarm_fac = macro_alarm_factor.unsqueeze(1)  # [T, 1, B]
            out_fac = macro_outcome.unsqueeze(0).unsqueeze(0)  # [1, 1, B]
            unshielded_omega = out_fac * alarm_fac  # [T, 1, B]

            not_win = (~win).unsqueeze(0).unsqueeze(0)  # [1, 1, B]
            shield_mask = not_win & car_unlocked_t  # [T, N_AGENTS, B]

            unshielded_omega = unshielded_omega.expand(-1, N_AGENTS, -1)
            alarm_fac = alarm_fac.expand(-1, N_AGENTS, -1)

            omega_t = torch.where(shield_mask, alarm_fac, unshielded_omega)

            # Permute from [T, N_AGENTS, B] to [T, N_AGENTS, B] -> rewards_t is [T, N_AGENTS, B]
            rewards_t = rewards_t * omega_t

            advantages, returns = compute_gae_simple(
                rewards_t, values_t, next_value, dones_t, args.gamma, args.gae_lambda
            )

        # Flatten the batch
        b_obs = obs_t.flatten(0, 2)
        b_roles = roles_t.flatten(0, 2)
        b_states = states_t.unsqueeze(1).repeat(1, N_AGENTS, 1, 1).flatten(0, 2)
        b_masks = masks_t.flatten(0, 2)
        b_actions = actions_t.flatten(0, 2)
        b_logprobs = logprobs_t.flatten(0, 2)
        b_advantages = advantages.flatten(0, 2)
        b_returns = returns.flatten(0, 2)
        b_experts = expert_idx_t.flatten(0, 2)

        # Optimize policy & value networks
        b_inds = np.arange(args.num_envs * args.num_steps * N_AGENTS)
        minibatch_size = len(b_inds) // args.num_minibatches

        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_states[mb_inds],
                    b_roles[mb_inds],
                    b_masks[mb_inds],
                    active_experts,
                    action=b_actions[mb_inds],
                    expert_idx=b_experts[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if update % 5 == 0 or update == num_updates:
            sps = int(global_step / (time.time() - start_time))
            mean_reward = rewards_t.sum(0).mean().item()
            win_rate = (
                (rewards_t[:, 0, :] > 5.0).any(dim=0).float().mean().item()
                if hasattr(rewards_t, "dim")
                else 0.0
            )
            print(
                f"[{run_name}] update={update} step={global_step} sps={sps} win_rate={win_rate:.3f} mean_reward={mean_reward:.3f} experts={active_experts}"
            )

    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"checkpoints/{run_name}/coop.pt")
        write_completion(f"checkpoints/{run_name}", args)


if __name__ == "__main__":
    main()
