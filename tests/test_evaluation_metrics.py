"""
tests/test_evaluation_metrics.py -- Unit tests for expanded evaluation metrics
including causal funnel rates, loss mode attribution, and per-role credit tracking.
"""

import numpy as np

from curriculum import CURRICULUM
from env import HeistEnv
from evaluate import evaluate_policies
from model import HeistAgent


def test_evaluate_policies_expanded_metrics():
    """Verify evaluate_policies computes causal funnel, loss cause, and role credit keys."""
    env = HeistEnv(CURRICULUM[0])
    policies = {a: HeistAgent() for a in env.possible_agents}

    # Run lightweight evaluation of 2 episodes
    metrics = evaluate_policies(policies, env, episodes=2, seed=42)

    expected_keys = [
        "win_rate",
        "mean_return",
        "mean_length",
        "mean_alarm",
        "scout_tag_rate",
        "terminal_rate",
        "loot_rate",
        "extraction_rate",
        "mean_hack_progress",
        "mean_explored_pct",
        "mean_tagged_pois",
        "mean_neutralized_guards",
        "cause_alarm_max_rate",
        "cause_guard_catch_rate",
        "cause_timeout_rate",
        "role_credit_scout",
        "role_credit_hacker",
        "role_credit_muscle",
        "role_credit_extractor",
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric key: {key}"
        assert isinstance(metrics[key], (float, int, np.floating, np.integer)), (
            f"Metric {key} must be numeric"
        )
