"""
MARC (Micro-Macro Asymmetric Retroactive Causal-chain) trainer for HEIST.

Implements the MARC algorithm:
1. Micro Credit (\\mu_{i, t}): Local action interaction impact via Action Affordance Deltas (\\Delta Mask_j(s_t, a_{i,t})).
2. Macro Weighting (\\Omega_t): Multiplicative global alarm penalty exp(-\alpha A_t / A_max) and outcome factor.
3. Asymmetric Failure Shielding: Negative credit targets direct failure triggers while shielding upstream enablers.
4. Backward Retroactive Pass (t = T -> 0): Propagates Micro-Macro causal advantages backward over trajectory.
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
    ALARM_MAX,
    N_AGENTS,
    OBSERVATION_SIZE,
)
from env import AGENTS, parse_env_config
from ppo_utils import (
    get_previous_stage_checkpoint,
    load_matching_weights,
    write_completion,
)
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "marc"
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 300_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    env_config: str = ""
    save_model: bool = True
    eval_every: int = 10
    alpha_alarm: float = 1.5
    gamma_causal: float = 0.95
    affordance_coef: float = 0.5
    use_rnd: bool = True
    rnd_coef: float = 0.05
    use_ckpt: bool = False
    from_stage: str = ""


def parse_args() -> Args:
    args = Args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=args.exp_name)
    parser.add_argument("--seed", type=int, default=args.seed)
    parser.add_argument("--total-timesteps", type=int, default=args.total_timesteps)
    parser.add_argument("--learning-rate", type=float, default=args.learning_rate)
    parser.add_argument("--num-envs", type=int, default=args.num_envs)
    parser.add_argument("--num-steps", type=int, default=args.num_steps)
    parser.add_argument("--env-config", type=str, default=args.env_config)
    parser.add_argument("--eval-every", type=int, default=args.eval_every)
    parser.add_argument("--alpha-alarm", type=float, default=args.alpha_alarm)
    parser.add_argument("--gamma-causal", type=float, default=args.gamma_causal)
    parser.add_argument("--affordance-coef", type=float, default=args.affordance_coef)
    parser.add_argument(
        "--no-shielding",
        action="store_true",
        dest="no_shielding",
        default=False,
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        dest="no_macro",
        default=False,
    )
    parser.add_argument(
        "--no-rnd",
        action="store_false",
        dest="use_rnd",
        default=True,
    )
    parser.add_argument("--rnd-coef", type=float, default=args.rnd_coef)
    parser.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        default=True,
    )
    parser.add_argument(
        "--use-ckpt",
        action="store_true",
        dest="use_ckpt",
        default=False,
    )
    parser.add_argument("--from-stage", type=str, default="")
    parsed = parser.parse_args()
    for k, v in vars(parsed).items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def compute_marc_advantages(
    rewards,
    values,
    dones,
    truncs,
    alarms,
    affordance_deltas,
    gamma,
    gae_lambda,
    alpha_alarm,
    gamma_causal,
    affordance_coef,
    no_shielding=False,
    no_macro=False,
):
    r"""Computes Micro-Macro Asymmetric Retroactive Causal-chain (MARC) advantages.

    Args:
        rewards: [N_AGENTS, T, B]
        values: [N_AGENTS, T, B]
        dones: [N_AGENTS, T, B]
        truncs: [N_AGENTS, T, B]
        alarms: [T, B] Alarm values A_t in [0, 100]
        affordance_deltas: [N_AGENTS, T, B] Continuous mask expansion sum
        no_shielding: If True, disables asymmetric failure shielding.
        no_macro: If True, disables macro weighting (\Omega_t = 1.0).
    """
    num_agents, num_steps, num_envs = rewards.shape
    base_adv = torch.zeros_like(rewards)
    marc_adv = torch.zeros_like(rewards)

    # 1. Standard GAE base computation
    for step in range(num_steps - 1, -1, -1):
        if step == num_steps - 1:
            next_val = torch.zeros_like(values[:, step])
        else:
            next_val = values[:, step + 1]
        nonterminal = 1.0 - dones[:, step]
        delta = rewards[:, step] + gamma * next_val * nonterminal - values[:, step]
        if step == num_steps - 1:
            base_adv[:, step] = delta
        else:
            base_adv[:, step] = (
                delta + gamma * gae_lambda * nonterminal * base_adv[:, step + 1]
            )

    # 2. Vectorized Micro-Macro Retroactive Backward Pass
    # Win condition per parallel env: True if any agent at any step received +10 win reward (>5.0)
    win = (rewards > 5.0).any(dim=1).any(dim=0)  # [B]

    if no_macro:
        omega_t = torch.ones_like(rewards)
    else:
        macro_alarm_factor = torch.exp(-alpha_alarm * (alarms / ALARM_MAX))  # [T, B]
        macro_outcome = torch.where(win, 1.0, -0.5)  # [B]

        alarm_fac = macro_alarm_factor.unsqueeze(0)  # [1, T, B]
        out_fac = macro_outcome.unsqueeze(0).unsqueeze(0)  # [1, 1, B]
        unshielded_omega = out_fac * alarm_fac  # [1, T, B]

        if not no_shielding:
            not_win = (~win).unsqueeze(0).unsqueeze(0)  # [1, 1, B]
            shield_mask = not_win & (affordance_deltas > 0)  # [N_AGENTS, T, B]
            omega_t = torch.where(shield_mask, alarm_fac, unshielded_omega)
        else:
            omega_t = unshielded_omega

    # Micro credit: Base GAE + Affordance boost
    affordance_boost = (
        affordance_deltas * affordance_coef if affordance_coef > 0.0 else 0.0
    )
    micro_credit = base_adv + affordance_boost
    immediate_marc = micro_credit * omega_t  # [N_AGENTS, T, B]

    # Backward Causal Trace Propagation
    retro_trace = torch.zeros(num_agents, num_envs, device=rewards.device)
    for step in range(num_steps - 1, -1, -1):
        retro_trace = (
            immediate_marc[:, step]
            + gamma_causal * (1.0 - dones[:, step]) * retro_trace
        )
        marc_adv[:, step] = retro_trace

    return marc_adv, marc_adv + values


def train(args):
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu"
    )

    env_cfg = parse_env_config(args.env_config) if args.env_config else {}
    vec_env = VectorEnv(args.num_envs, config=env_cfg, base_seed=args.seed)

    dummy_env = vec_env.envs[0]
    dummy_obs, _ = dummy_env.reset()
    state_dim = dummy_env.state().shape[0]
    from model import MappoAgent

    policy = MappoAgent(state_dim).to(device)

    if args.use_ckpt:
        ckpt_dir = get_previous_stage_checkpoint(run_name, exp_name=args.exp_name)
        if ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "policy.pt")
            load_matching_weights(policy, ckpt_path, device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    rnd_module = None
    if args.use_rnd:
        from exploration import RNDModule

        rnd_module = RNDModule(
            obs_dim=OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1], device=device
        )

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    def make_agent_dict(factory):
        return {a: factory() for a in AGENTS}

    buffers = make_agent_dict(
        lambda: {
            "obs": torch.zeros(
                (args.num_steps, args.num_envs, *OBSERVATION_SIZE),
                device=device,
            ),
            "role_id": torch.zeros(
                (args.num_steps, args.num_envs, N_AGENTS), device=device
            ),
            "action_mask": torch.zeros(
                (args.num_steps, args.num_envs, ACTION_DIM), device=device
            ),
            "actions": torch.zeros(
                (args.num_steps, args.num_envs),
                dtype=torch.long,
                device=device,
            ),
            "log_probs": torch.zeros((args.num_steps, args.num_envs), device=device),
            "rewards": torch.zeros((args.num_steps, args.num_envs), device=device),
            "dones": torch.zeros((args.num_steps, args.num_envs), device=device),
            "truncs": torch.zeros((args.num_steps, args.num_envs), device=device),
            "values": torch.zeros((args.num_steps, args.num_envs), device=device),
            "affordance_deltas": torch.zeros(
                (args.num_steps, args.num_envs),
                dtype=torch.float32,
                device=device,
            ),
        }
    )
    buffer_states = torch.zeros(
        (args.num_steps, args.num_envs, state_dim), device=device
    )
    buffer_alarms = torch.zeros((args.num_steps, args.num_envs), device=device)

    obs_dict, info_dict = vec_env.reset()
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            state_t = torch.tensor(
                np.array(vec_env.call("state")),
                dtype=torch.float32,
                device=device,
            )
            buffer_states[step] = state_t

            with torch.no_grad():
                val_t = policy.get_value(state_t).squeeze(-1)

            step_actions = {}
            with torch.no_grad():
                for a in AGENTS:
                    o_t = torch.tensor(
                        np.array(obs_dict[a]["observation"]),
                        dtype=torch.float32,
                        device=device,
                    )
                    r_t = torch.tensor(
                        np.array(obs_dict[a]["role_id"]),
                        dtype=torch.float32,
                        device=device,
                    )
                    m_t = torch.tensor(
                        np.array(obs_dict[a]["action_mask"]),
                        dtype=torch.float32,
                        device=device,
                    )
                    act, lp, _, _ = policy.get_action_and_value(o_t, r_t, m_t, state_t)
                    step_actions[a] = act

                    buffers[a]["obs"][step] = o_t
                    buffers[a]["role_id"][step] = r_t
                    buffers[a]["action_mask"][step] = m_t
                    buffers[a]["actions"][step] = act
                    buffers[a]["log_probs"][step] = lp
                    buffers[a]["values"][step] = val_t

            joint_actions_np = {a: step_actions[a].cpu().numpy() for a in AGENTS}
            next_obs, rewards, dones, truncs, infos = vec_env.step(joint_actions_np)

            # Store global alarm per parallel environment
            alarms_step = [
                infos[env_idx].get("scout", {}).get("alarm", 0.0)
                if isinstance(infos, list) and env_idx < len(infos)
                else 0.0
                for env_idx in range(args.num_envs)
            ]
            buffer_alarms[step] = torch.tensor(
                alarms_step, dtype=torch.float32, device=device
            )

            # Compute affordance deltas
            delta_masks = np.zeros((len(AGENTS), args.num_envs), dtype=np.float32)
            for j, a_j in enumerate(AGENTS):
                new_m = np.sum(next_obs[a_j]["action_mask"], axis=-1)
                old_m = np.sum(obs_dict[a_j]["action_mask"], axis=-1)
                delta_masks[j] = new_m - old_m

            for idx_a, a in enumerate(AGENTS):
                r_tensor = torch.tensor(rewards[a], dtype=torch.float32, device=device)
                if rnd_module is not None:
                    rnd_r = rnd_module.compute_reward(buffers[a]["obs"][step])
                    r_tensor = r_tensor + args.rnd_coef * rnd_r

                buffers[a]["rewards"][step] = r_tensor
                buffers[a]["dones"][step] = torch.tensor(
                    dones[a], dtype=torch.float32, device=device
                )
                buffers[a]["truncs"][step] = torch.tensor(
                    truncs[a], dtype=torch.float32, device=device
                )

                affordance_val = np.zeros(args.num_envs, dtype=np.float32)
                for env_idx in range(args.num_envs):
                    if (
                        isinstance(infos, list)
                        and env_idx < len(infos)
                        and infos[env_idx].get(a, {}).get("car_unlocked", False)
                    ):
                        for idx_j in range(len(AGENTS)):
                            if idx_j != idx_a:
                                affordance_val[env_idx] += max(
                                    0.0, float(delta_masks[idx_j, env_idx])
                                )

                buffers[a]["affordance_deltas"][step] = torch.tensor(
                    affordance_val, dtype=torch.float32, device=device
                )

            obs_dict = next_obs

        # Compute MARC Micro-Macro Retroactive Advantages
        with torch.no_grad():
            raw_rewards = torch.stack([buffers[a]["rewards"] for a in AGENTS], dim=0)
            raw_values = torch.stack([buffers[a]["values"] for a in AGENTS], dim=0)
            raw_dones = torch.stack([buffers[a]["dones"] for a in AGENTS], dim=0)
            raw_truncs = torch.stack([buffers[a]["truncs"] for a in AGENTS], dim=0)
            raw_affordance = torch.stack(
                [buffers[a]["affordance_deltas"] for a in AGENTS], dim=0
            )

            stacked_adv, stacked_returns = compute_marc_advantages(
                raw_rewards,
                raw_values,
                raw_dones,
                raw_truncs,
                buffer_alarms,
                raw_affordance,
                args.gamma,
                args.gae_lambda,
                args.alpha_alarm,
                args.gamma_causal,
                args.affordance_coef,
                no_shielding=args.no_shielding,
                no_macro=args.no_macro,
            )

            b_advs = {a: stacked_adv[i].reshape(-1) for i, a in enumerate(AGENTS)}
            b_returns = {
                a: stacked_returns[i].reshape(-1) for i, a in enumerate(AGENTS)
            }

        b_states = buffer_states.reshape(-1, state_dim)
        b_obs = {a: buffers[a]["obs"].reshape(-1, *OBSERVATION_SIZE) for a in AGENTS}
        b_role = {a: buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in AGENTS}
        b_mask = {a: buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in AGENTS}
        b_actions = {a: buffers[a]["actions"].reshape(-1) for a in AGENTS}
        b_log_probs = {a: buffers[a]["log_probs"].reshape(-1) for a in AGENTS}
        b_values = {a: buffers[a]["values"].reshape(-1) for a in AGENTS}

        b_inds = np.arange(batch_size)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                tot_policy_loss = 0.0
                tot_value_loss = 0.0
                tot_entropy_loss = 0.0

                state_mb = b_states[mb]
                new_value = policy.get_value(state_mb).squeeze(-1)

                for a in AGENTS:
                    obs_mb = b_obs[a][mb]
                    role_mb = b_role[a][mb]
                    mask_mb = b_mask[a][mb]
                    act_mb = b_actions[a][mb]
                    old_lp_mb = b_log_probs[a][mb]
                    adv_mb = b_advs[a][mb]
                    ret_mb = b_returns[a][mb]

                    if args.norm_adv:
                        adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    _, newlogprob, entropy, _ = policy.get_action_and_value(
                        obs_mb, role_mb, mask_mb, state_mb, action=act_mb
                    )
                    logratio = newlogprob - old_lp_mb
                    ratio = logratio.exp()

                    pg_loss1 = -adv_mb * ratio
                    pg_loss2 = -adv_mb * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    if args.clip_vloss:
                        v_loss_unclipped = (new_value - ret_mb) ** 2
                        v_clipped = b_values[a][mb] + torch.clamp(
                            new_value - b_values[a][mb],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret_mb) ** 2
                        v_loss = (
                            0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                        )
                    else:
                        v_loss = 0.5 * ((new_value - ret_mb) ** 2).mean()

                    tot_policy_loss = tot_policy_loss + pg_loss
                    tot_value_loss = tot_value_loss + v_loss
                    tot_entropy_loss = tot_entropy_loss + entropy.mean()

                loss = (
                    (tot_policy_loss / N_AGENTS)
                    + args.vf_coef * (tot_value_loss / N_AGENTS)
                    - args.ent_coef * (tot_entropy_loss / N_AGENTS)
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                if rnd_module is not None:
                    rnd_module.update(torch.cat([b_obs[a][mb] for a in AGENTS], dim=0))

        sps = int(global_step / (time.time() - start_time))
        avg_reward = np.mean([buffers[a]["rewards"].mean().item() for a in AGENTS])
        print(
            f"update={update} step={global_step} sps={sps} mean_reward={avg_reward:.4f}"
        )

        if (update % args.eval_every == 0 or update == num_updates) and args.save_model:
            os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
            torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")

    vec_env.close()
    writer.close()
    if args.save_model:
        os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
        torch.save(policy.state_dict(), f"checkpoints/{run_name}/policy.pt")
        write_completion(
            run_name,
            "marc",
            args.total_timesteps,
            num_updates * args.num_steps * args.num_envs,
        )
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    train(parse_args())
