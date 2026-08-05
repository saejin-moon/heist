"""
Smoke unit tests for multi-process VectorEnv lifecycle.
"""

from constants import AGENTS
from vec_env import VectorEnv


def test_multiprocessing_vector_env_lifecycle():
    """Verify VectorEnv multiprocessing workers start, reset, step, and close cleanly."""
    config = {
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
    }
    vec = VectorEnv(num_envs=2, config=config, base_seed=42)
    assert vec.num_envs == 2

    obs, state = vec.reset(seed=100)
    assert state.shape[0] == 2
    assert "scout" in obs
    assert obs["scout"]["observation"].shape[0] == 2

    actions = {a: [0, 0] for a in AGENTS}
    next_obs, rewards, terms, truncs, infos = vec.step(actions)

    assert len(infos) == 2
    assert "scout" in rewards
    assert len(rewards["scout"]) == 2
    assert len(terms["scout"]) == 2
    assert len(truncs["scout"]) == 2

    vec.close()
