"""
Smoke test for CAR (Counterfactual Affordance Reward) in HeistEnv & VectorEnv.

Validates that:
1. Environment detects when an INTERACT action unlocks an affordance for a teammate.
2. car_unlocked flag is correctly set in infos dictionary returned by step().
3. VectorEnv passes through infos correctly to the trainer.

Run with: uv run pytest src/test_car_smoke.py
"""

from constants import INTERACT
from env import HeistEnv
from vec_env import VectorEnv


def test_car_affordance_unlock():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "spawn_mode": "role",
        }
    )
    obs, _ = env.reset(seed=42)

    # Scout starts adjacent to terminal in role spawn mode.
    # Scout executes INTERACT to tag terminal, unlocking Hacker's INTERACT capability.
    actions = {a: 4 for a in env.possible_agents}  # WAIT
    actions["scout"] = INTERACT

    _, _, _, _, infos = env.step(actions)

    assert "car_unlocked" in infos["scout"]
    assert infos["scout"]["car_unlocked"] is True
    assert infos["hacker"]["car_unlocked"] is False


def test_vec_env_car_info_pass_through():
    vec = VectorEnv(
        2,
        config={
            "map_size": (11, 11),
            "guard_count": 0,
            "spawn_mode": "role",
        },
        base_seed=10,
    )
    obs, _ = vec.reset()

    actions = {a: [4, 4] for a in vec.envs[0].possible_agents}
    actions["scout"] = [INTERACT, 4]  # Env 0 scout interacts

    _, _, _, _, infos = vec.step(actions)

    assert "car_unlocked" in infos[0]["scout"]
    assert infos[0]["scout"]["car_unlocked"] is True
    assert infos[1]["scout"]["car_unlocked"] is False

    vec.close()
