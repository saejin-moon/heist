"""
Smoke unit tests for eval_stage.py CLI runner, 6-character max subdirectories,
and run name formatting.
"""

import os
import tempfile

import torch

from env import HeistEnv
from eval_stage import eval_one, get_next_run_id


def test_6char_run_id_generation():
    """Verify get_next_run_id produces auto-incrementing 6-character max names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        id1 = get_next_run_id(base_dir=tmpdir, prefix="run")
        assert id1 == "run001"
        assert len(id1) <= 6

        # Create dummy run001 directory
        os.makedirs(os.path.join(tmpdir, id1))
        id2 = get_next_run_id(base_dir=tmpdir, prefix="run")
        assert id2 == "run002"
        assert len(id2) <= 6

        os.makedirs(os.path.join(tmpdir, id2))
        id3 = get_next_run_id(base_dir=tmpdir, prefix="exp")
        assert id3 == "exp001"
        assert len(id3) <= 6


def test_single_suffix_run_name_logic():
    """Verify run_name formatting strips duplicate seed suffixes (e.g. ippo_s0 not ippo_s0_s0)."""
    exp_name = "ippo_s0"
    seed = 0
    run_name = (
        exp_name
        if (f"_s{seed}" in exp_name or exp_name.endswith(f"_s{seed}"))
        else f"{exp_name}_s{seed}"
    )
    assert run_name == "ippo_s0"

    exp_name_base = "ippo"
    seed = 1
    run_name_base = (
        exp_name_base
        if (f"_s{seed}" in exp_name_base or exp_name_base.endswith(f"_s{seed}"))
        else f"{exp_name_base}_s{seed}"
    )
    assert run_name_base == "ippo_s1"


def test_eval_one_checkpoint_skipping():
    """Verify eval_one successfully handles missing checkpoint directories without error."""
    config = {
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
    }
    env = HeistEnv(config)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp_root:
        res = eval_one(
            "ippo", "ippo", 0, 0, env, env.state().shape[0], device, tmp_root, 5
        )
        assert res is None
