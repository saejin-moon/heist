"""
Smoke tests for exploration modules (RND and CountExplorationModule).
"""

import numpy as np
import torch

from constants import OBSERVATION_SIZE
from exploration import CountExplorationModule, RNDModule


def test_rnd_module_shapes_and_update():
    obs_dim = OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]
    rnd = RNDModule(obs_dim=obs_dim, feature_dim=32, device="cpu")

    # Mock observation batch: shape (4 agents, 8 envs, obs_h, obs_w)
    obs = torch.randint(0, 10, (4, 8, OBSERVATION_SIZE[0], OBSERVATION_SIZE[1]), dtype=torch.int32)
    rewards = rnd.compute_reward(obs)

    assert rewards.shape == (4, 8), f"Expected shape (4, 8), got {rewards.shape}"
    assert torch.all(rewards >= 0.0), "RND intrinsic rewards must be non-negative"

    # Test predictor update step
    loss = rnd.update(obs)
    assert isinstance(loss, float), "Update should return float loss"
    assert loss >= 0.0, "Loss must be non-negative"


def test_count_exploration_module():
    counter = CountExplorationModule(beta=0.1)

    pos1 = np.array([2, 5])
    r1 = counter.compute_reward(pos1)
    assert r1 == 0.1 / np.sqrt(1)

    r2 = counter.compute_reward(pos1)
    assert r2 == 0.1 / np.sqrt(2)
    assert r2 < r1, "Repeated visits should decrease count-based intrinsic reward"
