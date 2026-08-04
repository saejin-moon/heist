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

from env import AGENTS
from constants import WAIT, INTERACT, ACTION_DELTAS, OBSERVATION_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _select_action(policy, obs, gs, mask, device, greedy=True):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    gs_t = torch.tensor(gs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.int64, device=device).unsqueeze(0)
    with torch.no_grad():
        if greedy:
            logits = _actor_logits(policy, obs_t, gs_t)
            masked = torch.where(mask_t == 1, logits, torch.full_like(logits, -1e9))
            return int(masked.argmax(dim=-1).item())
        action, _, _, _ = policy.get_action_and_value(obs_t, gs_t, mask_t)
        return int(action.item())


def _actor_logits(policy, obs_t, gs_t):
    """Extract raw action logits from any of the model classes.

    HeistAgent / MappoAgent expose `.actor`; QNetwork exposes `.net`.
    """
    x = torch.cat((torch.flatten(obs_t, start_dim=1), gs_t), dim=1)
    if hasattr(policy, "actor"):
        return policy.actor(x)
    return policy.net(x)


def run_episode(env, policies, device, greedy=True, seed=None, noop_agent=None):
    """Roll out one episode.  Returns a dict of per-episode metrics."""
    obs, _ = env.reset(seed=seed)

    total_return = 0.0
    length = 0
    # per-agent shaped credit (excludes shared terminal reward)
    credit = {a: 0.0 for a in AGENTS}
    terminal_reward = 0.0

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
                    policies[a], obs[a]["observation"], obs[a]["global_state"],
                    mask, device, greedy=True)
            else:
                actions[a] = int(np.random.choice(legal))
        obs, rewards, terms, truncs, infos = env.step(actions)
        length += 1
        done = bool(any(terms.values()) or any(truncs.values()))

        # separate shared terminal reward from per-agent shaped credit
        if done:
            terminal_reward = rewards[AGENTS[0]]
        else:
            for a in AGENTS:
                credit[a] += rewards[a]

        total_return += sum(rewards.values()) / len(AGENTS)
        if done:
            break

    return {
        "return": total_return,
        "length": length,
        "terminal_reward": terminal_reward,
        "win": terminal_reward > 0,
        "credit": credit,
        "alarm": env.alarm,
        "terminal_disabled": env.terminal_disabled,
        "loot_acquired": env.loot_acquired,
        "extraction_triggered": env.extraction_triggered,
        "hack_progress": env.hack_progress,
    }


def evaluate_policies(policies, env, episodes=20, seed=0, algo="ippo",
                      device="cpu", greedy=True):
    """Run `episodes` evaluation episodes and aggregate standard metrics."""
    metrics = {
        "win_rate": 0.0, "mean_return": 0.0, "mean_length": 0.0,
        "mean_alarm": 0.0, "terminal_rate": 0.0, "loot_rate": 0.0,
        "extraction_rate": 0.0, "mean_hack_progress": 0.0,
    }
    if episodes <= 0:
        return metrics
    results = [run_episode(env, policies, device, greedy=greedy, seed=seed + i)
               for i in range(episodes)]
    wins = sum(r["win"] for r in results)
    metrics["win_rate"] = wins / episodes
    metrics["mean_return"] = np.mean([r["return"] for r in results])
    metrics["mean_length"] = np.mean([r["length"] for r in results])
    metrics["mean_alarm"] = np.mean([r["alarm"] for r in results])
    metrics["terminal_rate"] = np.mean([r["terminal_disabled"] for r in results])
    metrics["loot_rate"] = np.mean([r["loot_acquired"] for r in results])
    metrics["extraction_rate"] = np.mean([r["extraction_triggered"] for r in results])
    metrics["mean_hack_progress"] = np.mean([r["hack_progress"] for r in results])
    return metrics


def credit_attribution_index(policies, env, episodes=50, seed=0, device="cpu"):
    """Correlation CAI.

    For each agent, the Pearson correlation between its per-episode shaped
    credit and the episode outcome (terminal reward).  High positive values
    mean the agent's contributions predict team success.  Low values for
    upstream agents (scout/hacker) indicate Causal Credit Dilution.
    """
    results = [run_episode(env, policies, device, greedy=True, seed=seed + i)
               for i in range(episodes)]
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
    base = evaluate_policies(policies, env, episodes=episodes, seed=seed,
                             device=device)
    base_win = base["win_rate"]
    importance = {}
    for a in AGENTS:
        wins = 0
        for i in range(episodes):
            r = run_episode(env, policies, device, greedy=True,
                            seed=seed + i, noop_agent=a)
            wins += int(r["win"])
        importance[a] = base_win - wins / episodes
    return {"baseline_win_rate": base_win, "importance": importance}


def summarize(policies, env, episodes=50, seed=0, device="cpu"):
    """Print a full evaluation report with CAI diagnostics."""
    print("=" * 64)
    print("HEIST evaluation")
    print("=" * 64)
    metrics = evaluate_policies(policies, env, episodes=episodes, seed=seed, device=device)
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
    cai = credit_attribution_index(policies, env, episodes=episodes, seed=seed, device=device)
    for a in AGENTS:
        print(f"  {a:>9}: {cai[a]:+.3f}")
    print("-" * 64)
    print("Counterfactual importance (baseline - no-op win rate):")
    imp = counterfactual_importance(policies, env, episodes=max(episodes // 2, 10),
                                    seed=seed + 10_000, device=device)
    print(f"  baseline win rate: {imp['baseline_win_rate']:.3f}")
    for a in AGENTS:
        print(f"  {a:>9}: {imp['importance'][a]:+.3f}")
    print("=" * 64)
    return metrics
