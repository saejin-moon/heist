"""Deterministic mechanics coverage for every curriculum stage."""

import numpy as np

from constants import DOOR, WAIT, WALL
from curriculum import CURRICULUM
from env import AGENTS, HeistEnv


def test_curriculum_stages_reset_and_step():
    """Every configured stage exposes valid observations and a fixed state size."""
    for stage_index, config in enumerate(CURRICULUM):
        env = HeistEnv(config)
        obs, _ = env.reset(seed=10_000 + stage_index)
        state_dim = env.state().shape
        for _ in range(3):
            obs, _, _, _, _ = env.step({agent: WAIT for agent in AGENTS})
            assert set(obs) == set(AGENTS)
            assert env.state().shape == state_dim
            for agent in AGENTS:
                assert obs[agent]["observation"].shape == (5, 5)
                assert obs[agent]["action_mask"].any()


def test_stage_four_converge_moves_guards_along_walkable_paths():
    """Exercise the compiled multi-source converge field on the largest map."""
    env = HeistEnv(CURRICULUM[4])
    env.reset(seed=20_000)
    before = list(env.guard_positions)
    env.alarm = env.config["converge_alarm"]
    env._move_guards()

    assert len(env.guard_positions) == len(before)
    for old, new in zip(before, env.guard_positions, strict=True):
        assert 0 <= new[0] < env.map_h and 0 <= new[1] < env.map_w
        assert env.grid[new] not in (WALL, DOOR)
        assert np.abs(np.subtract(new, old)).sum() <= 1
