"""
Smoke test for HEIST side-task extensions (enable_side_tasks=True).
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from constants import INTERACT  # noqa: E402
from env import HeistEnv  # noqa: E402


def test_side_tasks_disabled_by_default():
    env = HeistEnv()
    assert env.config["enable_side_tasks"] is False
    obs, _ = env.reset(seed=42)
    assert env.beacon_calibrated is False


def test_side_task_scout_decoy_ping():
    cfg = {"enable_side_tasks": True, "guard_count": 2}
    env = HeistEnv(cfg)
    obs, _ = env.reset(seed=100)

    # Place Scout away from any POIs and place a guard nearby
    scout_pos = (5, 5)
    env.agent_positions["scout"] = scout_pos
    env.guard_positions[0] = (5, 7)
    env.neutralized[0] = 0

    obs = env._get_all_obs()
    assert obs["scout"]["action_mask"][INTERACT] == 1

    actions = {a: 4 for a in env.possible_agents}  # WAIT
    actions["scout"] = INTERACT
    obs, rewards, term, trunc, info = env.step(actions)

    assert env._guard_search_target[0] == scout_pos
    assert env.guard_states[0] == "search"
    assert rewards["scout"] > 0


def test_side_task_hacker_door_override():
    cfg = {"enable_side_tasks": True, "door_count": 2}
    env = HeistEnv(cfg)
    obs, _ = env.reset(seed=200)

    # Place hacker adjacent to a door
    if env.door_positions:
        door_r, door_c = env.door_positions[0]
        env.agent_positions["hacker"] = (
            (door_r, door_c - 1) if door_c > 0 else (door_r, door_c + 1)
        )
        obs = env._get_all_obs()
        assert obs["hacker"]["action_mask"][INTERACT] == 1

        actions = {a: 4 for a in env.possible_agents}
        actions["hacker"] = INTERACT
        obs, rewards, term, trunc, info = env.step(actions)
        # Door tile should now be EMPTY
        assert env.grid[door_r, door_c] != 2  # DOOR = 2


def test_side_task_extractor_beacon_calibration():
    cfg = {"enable_side_tasks": True}
    env = HeistEnv(cfg)
    obs, _ = env.reset(seed=300)

    # Move Extractor adjacent to extract tile before loot is acquired
    ex_r, ex_c = env.extract_pos
    env.agent_positions["extractor"] = (ex_r, ex_c)
    obs = env._get_all_obs()

    assert obs["extractor"]["action_mask"][INTERACT] == 1

    actions = {a: 4 for a in env.possible_agents}
    actions["extractor"] = INTERACT
    obs, rewards, term, trunc, info = env.step(actions)

    assert env.beacon_calibrated is True
    assert rewards["extractor"] > 0
