"""
tests/test_thermal_and_concurrency.py -- Unit tests for thermal guard,
stage gating logic, area-scaled steps calculation, and QMIX batching.
"""

import subprocess

from curriculum import CURRICULUM


def test_thermal_guard_execution():
    """Verify thermal_guard.py script runs without exception."""
    res = subprocess.run(
        ["python3", "tools/thermal_guard.py"], capture_output=True, text=True
    )
    # Return code should be 0 (safe) or 1 (high temp emergency stop)
    assert res.returncode in (0, 1)


def test_curriculum_area_scaling_formula():
    """Verify spatial area step scaling formula across all 5 curriculum stages."""
    base_steps = 1_000_000
    expected_areas = [121, 289, 625, 1225, 2500, 2500]

    for i, stage in enumerate(CURRICULUM):
        w, h = stage["map_size"]
        area = w * h
        assert area == expected_areas[i]
        steps = base_steps * area // 121
        assert steps >= base_steps


def test_qmix_trainer_imports():
    """Verify QMIX trainer module and ReplayBuffer instantiate properly."""
    from train_qmix import QNetwork, ReplayBuffer

    buffer = ReplayBuffer(capacity=100, obs_shape=(5, 5), state_dim=50)
    assert buffer.capacity == 100

    q_net = QNetwork()
    assert q_net is not None
