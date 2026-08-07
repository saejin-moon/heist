"""
Evaluation and diagnostic metrics for HEIST.

Two families of metrics:

1. Standard benchmark metrics
   - win_rate, mean_return, mean_length, mean_final_alarm
   - per-agent task completion (scout tags, hack, neutralize, loot, extraction)

2. Credit Attribution Index (CAI) diagnostics
   The PLAN.md novelty hook predicts "Causal Credit Dilution": if the
   Extractor fails late in the episode, shared negative reward propagates
   backward and dilutes the credit the Scout/Hacker deserved for executing
   their prerequisites perfectly.

   We operationalize this two ways:

   a) Correlation CAI: across eval episodes, compute the per-agent shaped
      credit received (sum of their task/tag rewards, excluding the shared
      terminal reward) and correlate it with the final terminal outcome.
      If credit flows correctly, upstream agents that complete their phase
      early should strongly predict a win.  Dilution shows up as upstream
      credit becoming uncorrelated with outcome.

   b) Counterfactual importance: run the same seed-set once with the full
      team and once with each agent replaced by a no-op (always wait).
      The drop in win rate measures each agent's causal importance.
      Comparing importance to received credit exposes misattribution.
"""

import numpy as np
import torch

from constants import WAIT
from env import AGENTS


# ---------------------------------------------------------------------------
# Helpers
def _select_action(policy, obs, role, mask, device, greedy=True):
    if isinstance(obs, dict):
        role = obs.get("role_id", role)
        mask = obs.get("action_mask", mask)
        obs = obs.get("observation", obs)
    if hasattr(policy, "parameters"):
        try:
            device = next(policy.parameters()).device
        except StopIteration:
            pass
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    role_t = torch.as_tensor(role, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(mask, dtype=torch.int64, device=device).unsqueeze(0)
    with torch.no_grad():
        if greedy or not hasattr(policy, "get_action_and_value"):
            logits = _actor_logits(policy, obs_t, role_t)
            masked = torch.where(mask_t == 1, logits, torch.full_like(logits, -1e9))
            return int(masked.argmax(dim=-1).item())
        action, _, _, _ = policy.get_action_and_value(obs_t, role_t, mask_t)
        return int(action.item())


def _actor_logits(policy, obs_t, role_t):
    """Extract raw action logits from any model class (HeistAgent, MappoAgent, ComaAgent, QNetwork)."""
    if hasattr(policy, "get_action_and_value"):
        x = torch.cat((torch.flatten(obs_t, start_dim=1), role_t), dim=1)
        if hasattr(policy, "actor"):
            return policy.actor(x)
        elif hasattr(policy, "net"):
            return policy.net(x)
    try:
        return policy(obs_t, role_t)
    except Exception:
        x = torch.cat((torch.flatten(obs_t, start_dim=1), role_t), dim=1)
        if hasattr(policy, "net"):
            return policy.net(x)
        elif hasattr(policy, "actor"):
            return policy.actor(x)
        return policy(x)


def run_episode(env, policies, device, greedy=True, seed=None, noop_agent=None):
    """Roll out one episode.  Returns a dict of per-episode metrics."""
    obs, _ = env.reset(seed=seed)

    total_return = 0.0
    length = 0
    # per-agent shaped credit (excludes shared terminal reward)
    credit = {a: 0.0 for a in AGENTS}
    terminal_reward = 0.0
    win = False

    for _ in range(env.config["max_steps"]):
        actions = {}
        for a in AGENTS:
            if noop_agent == a:
                actions[a] = WAIT
                continue
            mask = obs[a]["action_mask"]
            legal = np.argwhere(mask == 1).ravel()
            if greedy and len(legal) > 0:
                actions[a] = _select_action(
                    policies[a],
                    obs[a]["observation"],
                    obs[a]["role_id"],
                    mask,
                    device,
                    greedy=True,
                )
            else:
                actions[a] = int(np.random.choice(legal))
        obs, rewards, terms, truncs, infos = env.step(actions)
        length += 1
        terminated = bool(any(terms.values()))
        truncated = bool(any(truncs.values()))
        done = terminated or truncated

        # separate shared terminal reward from per-agent shaped credit.
        # Only a real termination carries the +/-10 terminal reward; a
        # truncation (timeout) is neither a win nor a loss here, and the
        # per-step rewards at that boundary must not be treated as terminal.
        if terminated:
            terminal_reward = rewards[AGENTS[0]]
            win = bool(infos[AGENTS[0]].get("win", False))
        elif not done:
            for a in AGENTS:
                credit[a] += rewards[a]

        total_return += sum(rewards.values()) / len(AGENTS)
        if done:
            break

    # Loss mode cause attribution
    if win:
        cause = "win"
    elif env.alarm >= env.config.get("alarm_max", 100.0):
        cause = "alarm_max"
    elif env._check_caught():
        cause = "guard_catch"
    else:
        cause = "timeout"

    return {
        "return": total_return,
        "length": length,
        "terminal_reward": terminal_reward,
        "win": win,
        "credit": credit,
        "alarm": env.alarm,
        "terminal_disabled": env.terminal_disabled,
        "loot_acquired": env.loot_acquired,
        "extraction_triggered": env.extraction_triggered,
        "hack_progress": env.hack_progress,
        "explored_pct": float(np.mean(env.explored_map)),
        "tagged_pois": len(env.tagged_pois),
        "scout_tagged": len(env.tagged_pois) > 0,
        "neutralized_guards": int(np.sum(env.neutralized > 0)),
        "cause": cause,
    }


def evaluate_policies(
    policies, env, episodes=20, seed=0, algo="ippo", device="cpu", greedy=True
):
    """Run `episodes` evaluation episodes and aggregate standard metrics."""
    metrics = {
        "win_rate": 0.0,
        "mean_return": 0.0,
        "mean_length": 0.0,
        "mean_alarm": 0.0,
        "scout_tag_rate": 0.0,
        "terminal_rate": 0.0,
        "loot_rate": 0.0,
        "extraction_rate": 0.0,
        "mean_hack_progress": 0.0,
        "mean_explored_pct": 0.0,
        "mean_tagged_pois": 0.0,
        "mean_neutralized_guards": 0.0,
        "cause_alarm_max_rate": 0.0,
        "cause_guard_catch_rate": 0.0,
        "cause_timeout_rate": 0.0,
        "role_credit_scout": 0.0,
        "role_credit_hacker": 0.0,
        "role_credit_muscle": 0.0,
        "role_credit_extractor": 0.0,
    }
    if episodes <= 0:
        return metrics
    results = [
        run_episode(env, policies, device, greedy=greedy, seed=seed + i)
        for i in range(episodes)
    ]
    
    # Vectorized fast metric calculation
    n_ep = float(episodes)
    metrics["win_rate"] = sum(r["win"] for r in results) / n_ep
    metrics["mean_return"] = float(np.mean([r["return"] for r in results]))
    metrics["mean_length"] = float(np.mean([r["length"] for r in results]))
    metrics["mean_alarm"] = float(np.mean([r["alarm"] for r in results]))
    metrics["scout_tag_rate"] = sum(r["scout_tagged"] for r in results) / n_ep
    metrics["terminal_rate"] = sum(r["terminal_disabled"] for r in results) / n_ep
    metrics["loot_rate"] = sum(r["loot_acquired"] for r in results) / n_ep
    metrics["extraction_rate"] = sum(r["extraction_triggered"] for r in results) / n_ep
    metrics["mean_hack_progress"] = float(np.mean([r["hack_progress"] for r in results]))
    metrics["mean_explored_pct"] = float(np.mean([r.get("explored_pct", 0.0) for r in results]))
    metrics["mean_tagged_pois"] = float(np.mean([r.get("tagged_pois", 0.0) for r in results]))
    metrics["mean_neutralized_guards"] = float(np.mean([r.get("neutralized_guards", 0.0) for r in results]))
    
    # Cause attribution breakdown
    metrics["cause_alarm_max_rate"] = sum(r["cause"] == "alarm_max" for r in results) / n_ep
    metrics["cause_guard_catch_rate"] = sum(r["cause"] == "guard_catch" for r in results) / n_ep
    metrics["cause_timeout_rate"] = sum(r["cause"] == "timeout" for r in results) / n_ep
    
    # Role credit breakdown
    for a in AGENTS:
        metrics[f"role_credit_{a}"] = float(np.mean([r["credit"][a] for r in results]))

    return metrics


def credit_attribution_index(policies, env, episodes=50, seed=0, device="cpu"):
    """Correlation CAI.

    For each agent, the Pearson correlation between its per-episode shaped
    credit and the episode outcome (terminal reward).  High positive values
    mean the agent's contributions predict team success.  Low values for
    upstream agents (scout/hacker) indicate Causal Credit Dilution.
    """
    results = [
        run_episode(env, policies, device, greedy=True, seed=seed + i)
        for i in range(episodes)
    ]
    outcomes = np.array([r["terminal_reward"] for r in results])
    cai = {}
    for a in AGENTS:
        credits = np.array([r["credit"][a] for r in results])
        if outcomes.std() == 0 or credits.std() == 0:
            cai[a] = 0.0
        else:
            cai[a] = float(np.corrcoef(credits, outcomes)[0, 1])
    return cai


def counterfactual_importance(policies, env, episodes=30, seed=0, device="cpu"):
    """Counterfactual importance.

    Baseline win rate vs win rate when each agent is replaced by a no-op
    (always waits).  The drop (baseline - noop) is the agent's causal
    importance for the team.
    """
    base = evaluate_policies(policies, env, episodes=episodes, seed=seed, device=device)
    base_win = base["win_rate"]
    importance = {}
    for a in AGENTS:
        wins = 0
        for i in range(episodes):
            r = run_episode(
                env, policies, device, greedy=True, seed=seed + i, noop_agent=a
            )
            wins += int(r["win"])
        importance[a] = base_win - wins / episodes
    return {"baseline_win_rate": base_win, "importance": importance}


def summarize(policies, env, episodes=50, seed=0, device="cpu"):
    """Print a full evaluation report with CAI diagnostics."""
    print("=" * 64)
    print("HEIST evaluation")
    print("=" * 64)
    metrics = evaluate_policies(
        policies, env, episodes=episodes, seed=seed, device=device
    )
    print(f"win_rate          : {metrics['win_rate']:.3f}")
    print(f"mean_return       : {metrics['mean_return']:.3f}")
    print(f"mean_length       : {metrics['mean_length']:.1f}")
    print(f"mean_alarm        : {metrics['mean_alarm']:.1f}")
    print(f"terminal_rate     : {metrics['terminal_rate']:.3f}")
    print(f"loot_rate         : {metrics['loot_rate']:.3f}")
    print(f"extraction_rate   : {metrics['extraction_rate']:.3f}")
    print(f"mean_hack_progress: {metrics['mean_hack_progress']:.2f}")
    print("-" * 64)
    print("Credit Attribution Index (corr credit vs outcome):")
    cai = credit_attribution_index(
        policies, env, episodes=episodes, seed=seed, device=device
    )
    for a in AGENTS:
        print(f"  {a:>9}: {cai[a]:+.3f}")
    print("-" * 64)
    print("Counterfactual importance (baseline - no-op win rate):")
    imp = counterfactual_importance(
        policies,
        env,
        episodes=max(episodes // 2, 10),
        seed=seed + 10_000,
        device=device,
    )
    print(f"  baseline win rate: {imp['baseline_win_rate']:.3f}")
    for a in AGENTS:
        print(f"  {a:>9}: {imp['importance'][a]:+.3f}")
    print("=" * 64)
    return metrics


# ---------------------------------------------------------------------------
# REV-7 (REVISION_PLAN.md §6) Phase B/C: communication-aware evaluation
# ---------------------------------------------------------------------------
def run_comm_episode(
    policy, env, device, greedy=True, seed=None, record_messages=False
):
    """Roll out one episode with a shared CommAgent (joint forward).

    Every step, all agents' observations are passed to the CommAgent at
    once so TarMAC message passing can happen before action selection.

    Returns (metrics_dict, messages) where messages (only when
    record_messages=True) is a list of per-step dicts with the gated
    messages for each agent and the phase flags at that step.
    """
    obs, _ = env.reset(seed=seed)
    total_return = 0.0
    length = 0
    credit = {a: 0.0 for a in AGENTS}
    terminal_reward = 0.0
    win = False
    messages = []

    for _ in range(env.config["max_steps"]):
        obs_list = [
            torch.tensor(
                obs[a]["observation"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            for a in AGENTS
        ]
        role_list = [
            torch.tensor(
                obs[a]["role_id"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            for a in AGENTS
        ]
        mask_list = [
            torch.tensor(
                obs[a]["action_mask"], dtype=torch.int64, device=device
            ).unsqueeze(0)
            for a in AGENTS
        ]

        with torch.no_grad():
            if greedy:
                logits = _comm_logits(policy, obs_list, role_list)  # [n, A]
                actions = []
                for a, m in enumerate(mask_list):
                    masked = torch.where(
                        m == 1, logits[a], torch.full_like(logits[a], -1e9)
                    )
                    actions.append(masked.argmax(dim=-1))
            else:
                actions, _, _, _ = policy.get_action_and_value(
                    obs_list, role_list, mask_list, state=None, action=None
                )
                actions = [actions[0, a] for a in range(len(AGENTS))]

        if record_messages:
            msg = policy._last_messages_gated  # [1, n, msg_dim]
            messages.append(
                {
                    "messages": msg.squeeze(0).cpu().numpy(),
                    "terminal_disabled": int(env.terminal_disabled),
                    "loot_acquired": int(env.loot_acquired),
                    "extraction_triggered": int(env.extraction_triggered),
                }
            )

        actions_dict = {a: int(actions[i].item()) for i, a in enumerate(AGENTS)}
        obs, rewards, terms, truncs, infos = env.step(actions_dict)
        length += 1
        terminated = bool(any(terms.values()))
        truncated = bool(any(truncs.values()))
        done = terminated or truncated

        if terminated:
            terminal_reward = rewards[AGENTS[0]]
            win = bool(infos[AGENTS[0]].get("win", False))
        elif not done:
            for a in AGENTS:
                credit[a] += rewards[a]

        total_return += sum(rewards.values()) / len(AGENTS)
        if done:
            break

    metrics = {
        "return": total_return,
        "length": length,
        "terminal_reward": terminal_reward,
        "win": win,
        "credit": credit,
        "alarm": env.alarm,
        "terminal_disabled": env.terminal_disabled,
        "loot_acquired": env.loot_acquired,
        "extraction_triggered": env.extraction_triggered,
        "hack_progress": env.hack_progress,
    }
    return metrics, messages


def _comm_logits(policy, obs_list, role_list):
    """Joint forward through a CommAgent, returning per-agent logits."""
    features = policy._joint_features(obs_list, role_list)
    encoded = policy.encoder(features)
    aggregated, messages_gated, attention = policy.comm(encoded)
    policy._last_messages_gated = messages_gated.detach()
    policy._last_attention = attention.detach()
    B, n, _ = features.shape
    agent_input = torch.cat([features, aggregated], dim=-1)
    logits = policy.actor(agent_input.view(B * n, -1))
    return logits.view(B, n, -1)[0]  # [n, ACTION_DIM]


def evaluate_comm_policies(policy, env, episodes=20, seed=0, device="cpu", greedy=True):
    """Standard metrics for a shared CommAgent."""
    metrics = {
        "win_rate": 0.0,
        "mean_return": 0.0,
        "mean_length": 0.0,
        "mean_alarm": 0.0,
        "terminal_rate": 0.0,
        "loot_rate": 0.0,
        "extraction_rate": 0.0,
        "mean_hack_progress": 0.0,
    }
    if episodes <= 0:
        return metrics
    results = [
        run_comm_episode(policy, env, device, greedy=greedy, seed=seed + i)[0]
        for i in range(episodes)
    ]
    metrics["win_rate"] = np.mean([r["win"] for r in results])
    metrics["mean_return"] = np.mean([r["return"] for r in results])
    metrics["mean_length"] = np.mean([r["length"] for r in results])
    metrics["mean_alarm"] = np.mean([r["alarm"] for r in results])
    metrics["terminal_rate"] = np.mean([r["terminal_disabled"] for r in results])
    metrics["loot_rate"] = np.mean([r["loot_acquired"] for r in results])
    metrics["extraction_rate"] = np.mean([r["extraction_triggered"] for r in results])
    metrics["mean_hack_progress"] = np.mean([r["hack_progress"] for r in results])
    return metrics


def message_outcome_correlation(policy, env, episodes=20, seed=0, device="cpu"):
    """REV-7 Phase C emergent-language diagnostic.

    Records every step's per-agent gated messages together with the phase
    flags (terminal_disabled, loot_acquired, extraction_triggered) and
    returns the Pearson correlation of each message dimension with each
    phase flag, plus the mean attention weight between agents.

    If the agents learn to signal "terminal hacked" (the intended message),
    some message dimension should correlate strongly with terminal_disabled.
    """
    all_msgs = []  # [T, n_agents, msg_dim]
    all_flags = []  # [T, 3]
    attn_sum = np.zeros((len(AGENTS), len(AGENTS)))
    attn_count = 0
    for i in range(episodes):
        _, steps = run_comm_episode(
            policy, env, device, greedy=True, seed=seed + i, record_messages=True
        )
        for s in steps:
            all_msgs.append(s["messages"])
            all_flags.append(
                [s["terminal_disabled"], s["loot_acquired"], s["extraction_triggered"]]
            )
            attn = policy._last_attention  # [1, n, n]
            attn_sum += attn.squeeze(0).cpu().numpy()
            attn_count += 1

    msgs = np.array(all_msgs)  # [T, n, d]
    flags = np.array(all_flags)  # [T, 3]
    n_agents, d = msgs.shape[1], msgs.shape[2]

    # per-agent: corr of each message dim with each phase flag
    corr = np.zeros((n_agents, d, 3))
    for a in range(n_agents):
        for dim in range(d):
            for f in range(3):
                x = msgs[:, a, dim]
                y = flags[:, f]
                if x.std() == 0 or y.std() == 0:
                    corr[a, dim, f] = 0.0
                else:
                    corr[a, dim, f] = float(np.corrcoef(x, y)[0, 1])

    diag = {
        "message_phase_corr": corr,  # [n, d, 3]
        "mean_attention": attn_sum / max(attn_count, 1),  # [n, n]
        "n_steps": len(all_msgs),
    }
    # headline scalar: strongest |corr| between any message dim and
    # terminal_disabled across all agents (the intended signal).
    terminal_corr = np.abs(corr[:, :, 0])
    diag["max_terminal_message_corr"] = (
        float(terminal_corr.max()) if terminal_corr.size else 0.0
    )
    diag["mean_terminal_message_corr"] = (
        float(terminal_corr.mean()) if terminal_corr.size else 0.0
    )
    return diag
