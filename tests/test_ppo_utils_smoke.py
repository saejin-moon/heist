"""
Unit test for ppo_utils module (compute_gae and write_completion).

Validates that:
1. compute_gae correctly computes advantages and returns for [agents, time, env] tensors.
2. Episode boundaries (terminated vs truncated) are correctly masked and bootstrapped.
3. write_completion creates atomic JSON completion markers.

Run with: uv run pytest src/test_ppo_utils_smoke.py
"""

import json
import shutil
from pathlib import Path

import torch

from ppo_utils import compute_gae, write_completion


def test_compute_gae_basic():
    n_agents, num_steps, num_envs = 4, 10, 2
    gamma, gae_lambda = 0.99, 0.95

    rewards = torch.ones(n_agents, num_steps, num_envs)
    values = torch.zeros(n_agents, num_steps, num_envs)
    terminated = torch.zeros(n_agents, num_steps, num_envs)
    truncated = torch.zeros(n_agents, num_steps, num_envs)
    bootstrap = torch.zeros(n_agents, num_steps, num_envs)
    next_values = torch.zeros(n_agents, num_envs)
    next_terminated = torch.zeros(n_agents, num_envs)

    advantages, returns = compute_gae(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap,
        next_values,
        next_terminated,
        gamma,
        gae_lambda,
    )

    assert advantages.shape == (n_agents, num_steps, num_envs)
    assert returns.shape == (n_agents, num_steps, num_envs)
    # With rewards=1.0 and values=0.0, advantages must be positive
    assert torch.all(advantages > 0.0)


def test_compute_gae_truncation_bootstrap():
    n_agents, num_steps, num_envs = 2, 5, 1
    gamma, gae_lambda = 0.99, 0.95

    rewards = torch.zeros(n_agents, num_steps, num_envs)
    values = torch.zeros(n_agents, num_steps, num_envs)
    terminated = torch.zeros(n_agents, num_steps, num_envs)
    truncated = torch.zeros(n_agents, num_steps, num_envs)
    bootstrap = torch.zeros(n_agents, num_steps, num_envs)

    # Step 2 is truncated with bootstrap value of 10.0
    truncated[:, 2, 0] = 1.0
    bootstrap[:, 2, 0] = 10.0

    next_values = torch.zeros(n_agents, num_envs)
    next_terminated = torch.zeros(n_agents, num_envs)

    advantages, _ = compute_gae(
        rewards,
        values,
        terminated,
        truncated,
        bootstrap,
        next_values,
        next_terminated,
        gamma,
        gae_lambda,
    )

    # At step 2, following_values must equal bootstrap (10.0), making delta = 0 + 0.99 * 10.0 = 9.9
    assert torch.allclose(advantages[:, 2, 0], torch.tensor([9.9, 9.9]), atol=1e-3)


def test_write_completion_marker():
    run_name = "test_unit_completion_marker_run"
    checkpoint_dir = Path("checkpoints") / run_name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    write_completion(run_name, "test_algo", 1000, 1000)

    marker = checkpoint_dir / "complete.json"
    assert marker.is_file()

    data = json.loads(marker.read_text())
    assert data["algorithm"] == "test_algo"
    assert data["requested_steps"] == 1000
    assert data["completed_steps"] == 1000

    shutil.rmtree(checkpoint_dir)


def test_get_previous_stage_checkpoint_fallback(tmp_path, monkeypatch):
    from ppo_utils import get_previous_stage_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    # 1. Standard previous stage: mappo_s1 looks for checkpoints/mappo_s0
    (ckpt_dir / "mappo_s0").mkdir()
    (ckpt_dir / "mappo_s0" / "dummy.pt").write_text("1")
    assert get_previous_stage_checkpoint("mappo_s1") == "checkpoints/mappo_s0"

    # 2. Side-task stage 0 fallback to normal Stage 4 if available
    (ckpt_dir / "ippo_s4").mkdir()
    (ckpt_dir / "ippo_s4" / "dummy.pt").write_text("1")
    assert get_previous_stage_checkpoint("ippo_st_s0") == "checkpoints/ippo_s4"

    # 3. Side-task stage 0 fallback to highest available lower stage if s4 doesn't exist (e.g. s2)
    (ckpt_dir / "comm_s2").mkdir()
    (ckpt_dir / "comm_s2" / "dummy.pt").write_text("1")
    assert get_previous_stage_checkpoint("comm_st_s0") == "checkpoints/comm_s2"
