import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add tools directory to path
REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import status


def test_get_next_run_id_helper(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Initial state -> run001
    assert status._get_next_run_id(results_dir, prefix="run") == "run001"

    # Add run001 -> run002
    (results_dir / "run001").mkdir()
    assert status._get_next_run_id(results_dir, prefix="run") == "run002"

    # Add run019 -> run020
    (results_dir / "run019").mkdir()
    assert status._get_next_run_id(results_dir, prefix="run") == "run020"


def test_stale_launch_out_fallback_to_results(tmp_path):
    log_dir = tmp_path / "log"
    results_dir = tmp_path / "results"
    log_dir.mkdir()
    results_dir.mkdir()

    # Create stale launch.out for run019
    launch_out = log_dir / "launch.out"
    launch_out.write_text("Campaign Evaluation Run ID: run019\nStarting training for stage 0")

    # Set launch.out mtime to past
    old_time = time.time() - 3600
    os.utime(launch_out, (old_time, old_time))

    # Create newer results directory run020
    r020 = results_dir / "run020"
    r020.mkdir()
    new_time = time.time() - 300
    os.utime(r020, (new_time, new_time))

    with patch.object(status, "LOG_DIR", log_dir), \
         patch.object(status, "RESULTS_DIR", results_dir), \
         patch.object(status, "get_running_pids", return_value=set()):
        info = status.get_active_campaign_info()
        assert info["run_id"] == "run020"


def test_active_training_predicts_next_run_id(tmp_path):
    log_dir = tmp_path / "log"
    results_dir = tmp_path / "results"
    log_dir.mkdir()
    results_dir.mkdir()

    # Old launch.out
    launch_out = log_dir / "launch.out"
    launch_out.write_text("Campaign Evaluation Run ID: run019\nStarting training for stage 0")
    old_time = time.time() - 3600
    os.utime(launch_out, (old_time, old_time))

    # Existing results directory run019
    r019 = results_dir / "run019"
    r019.mkdir()
    os.utime(r019, (old_time, old_time))

    # Active log file updated right now
    active_log = log_dir / "ippo_s0.log"
    active_log.write_text("step=100 sps=50 mean_reward=-0.1")

    with patch.object(status, "LOG_DIR", log_dir), \
         patch.object(status, "RESULTS_DIR", results_dir), \
         patch.object(status, "get_running_pids", return_value={1234}):
        info = status.get_active_campaign_info()
        assert info["run_id"] == "run020"
