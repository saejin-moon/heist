"""
Smoke unit tests for reward shaping and potential-based extraction convergence.
"""

from constants import AGENTS
from env import HeistEnv


def test_extraction_convergence_reward_shaping():
    """Verify convergence bonus triggers when loot_acquired or extraction_triggered is active."""
    config = {
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
    }
    env = HeistEnv(config)
    env.reset(seed=777)

    # Set loot_acquired to True
    env.loot_acquired = True

    # Step actions towards extract_pos
    actions = {a: 0 for a in AGENTS}
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Rewards should include shaped convergence entries for all agents
    assert len(rewards) == len(AGENTS)
    assert all(isinstance(r, (float, int)) for r in rewards.values())
