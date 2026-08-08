"""
ATE (Average Treatment Effect Contrastive PPO) trainer for HEIST.

Computes Average Treatment Effect (ATE) advantage against explicit WAIT null action (a_4 = WAIT):
    A_i^{ATE}(s, a_i) = Q_i(s, a_{-i}, a_i) - Q_i(s, a_{-i}, a_{WAIT})
"""

import argparse
import os
import time
from dataclasses import dataclass

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
    compute_ate_advantage,
    get_previous_stage_checkpoint,
    load_matching_weights,
    write_completion,
)
from vec_env import VectorEnv


@dataclass
class Args:
    exp_name: str = "ate"
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
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    env_config: str = ""
    save_model: bool = True
    eval_every: int = 10
    cir_coef: float = 0.0
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
    parser.add_argument("--cir-coef", type=float, default=args.cir_coef)
    parser.add_argument(
        "--use-rnd",
        action="store_true",
        default=False,
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

    policy = ComaAgent(state_dim=state_dim, hidden_dim=64).to(device)
    target_critic = ComaCritic(state_dim=state_dim, hidden_dim=64).to(device)
    target_critic.load_state_dict(policy.critic.state_dict())

    if args.use_ckpt:
        ckpt_dir = get_previous_stage_checkpoint(run_name, exp_name=args.exp_name)
        if ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "policy.pt")
            load_matching_weights(policy, ckpt_path, device)
            target_critic.load_state_dict(policy.critic.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

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
            "other_actions_oh": torch.zeros(
                (args.num_steps, args.num_envs, (N_AGENTS - 1) * ACTION_DIM),
                device=device,
            ),
        }
    )
    buffer_states = torch.zeros(
        (args.num_steps, args.num_envs, state_dim), device=device
    )

    obs_dict, info_dict = vec_env.reset()
    global_step = 0
    start_time = time.time()

    def eval_policies(step):
        pass

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

            step_actions = {}
            step_log_probs = {}

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
                    act, lp, _, _ = policy.get_action_and_probs(o_t, r_t, m_t)
                    step_actions[a] = act
                    step_log_probs[a] = lp

                    buffers[a]["obs"][step] = o_t
                    buffers[a]["role_id"][step] = r_t
                    buffers[a]["action_mask"][step] = m_t
                    buffers[a]["actions"][step] = act
                    buffers[a]["log_probs"][step] = lp

            joint_actions_np = {a: step_actions[a].cpu().numpy() for a in AGENTS}
            next_obs, rewards, dones, truncs, infos = vec_env.step(joint_actions_np)

            for _idx_a, a in enumerate(AGENTS):
                other_oh = []
                for other in AGENTS:
                    if other != a:
                        oh = F.one_hot(step_actions[other], num_classes=ACTION_DIM)
                        other_oh.append(oh)
                buffers[a]["other_actions_oh"][step] = torch.cat(
                    other_oh, dim=1
                ).float()

                r_tensor = torch.tensor(rewards[a], dtype=torch.float32, device=device)
                if rnd_module is not None:
                    rnd_r = rnd_module.compute_reward(buffers[a]["obs"][step])
                    r_tensor = r_tensor + args.rnd_coef * rnd_r

                buffers[a]["rewards"][step] = r_tensor
                buffers[a]["dones"][step] = torch.tensor(
                    dones[a], dtype=torch.float32, device=device
                )

            obs_dict = next_obs

        # Compute Q targets & ATE advantages
        b_states = buffer_states.reshape(-1, state_dim)
        b_obs = {a: buffers[a]["obs"].reshape(-1, *OBSERVATION_SIZE) for a in AGENTS}
        b_role = {a: buffers[a]["role_id"].reshape(-1, N_AGENTS) for a in AGENTS}
        b_mask = {a: buffers[a]["action_mask"].reshape(-1, ACTION_DIM) for a in AGENTS}
        b_actions = {a: buffers[a]["actions"].reshape(-1) for a in AGENTS}
        b_other_oh = {
            a: buffers[a]["other_actions_oh"].reshape(-1, (N_AGENTS - 1) * ACTION_DIM)
            for a in AGENTS
        }
        b_targets = {}
        b_adv = {}

        with torch.no_grad():
            for a in AGENTS:
                q_all = target_critic(b_states, b_other_oh[a])
                adv, q_taken = compute_ate_advantage(
                    q_all, b_actions[a], b_mask[a], null_action_idx=4
                )
                b_targets[a] = q_taken
                b_adv[a] = adv

        b_inds = np.arange(batch_size)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
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

                    _, log_prob, _, entropy = policy.get_action_and_probs(
                        obs_mb, role_mb, mask_mb, action=act_mb
                    )
                    q_vals_all = policy.get_value(state_mb, other_oh_mb)
                    adv_unrouted, q_taken = compute_ate_advantage(
                        q_vals_all, act_mb, mask_mb, null_action_idx=4
                    )
                    adv = adv_unrouted

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

                with torch.no_grad():
                    for p, tp in zip(
                        policy.critic.parameters(),
                        target_critic.parameters(),
                        strict=False,
                    ):
                        tp.data.copy_(0.01 * p.data + 0.99 * tp.data)

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
            "ate",
            args.total_timesteps,
            num_updates * args.num_steps * args.num_envs,
        )
    elapsed = time.time() - start_time
    end_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    print(f"[{end_utc}] training done in {elapsed:.1f}s ({elapsed / 60:.1f} min).")


if __name__ == "__main__":
    train(parse_args())
