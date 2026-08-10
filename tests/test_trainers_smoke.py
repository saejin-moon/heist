import glob
import subprocess

import pytest


def test_all_trainers_execution():
    """
    Run all trainers for a minimal number of steps to ensure there are no
    glaring runtime bugs (e.g. AttributeError, typo in method names) during
    the main loop.
    """
    train_scripts = sorted(glob.glob("src/train_*.py"))
    assert len(train_scripts) > 0, "No train scripts found."

    env_config = '{"map_size": [9, 9], "num_rooms_range": [1, 2], "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 20, "spawn_mode": "role"}'

    failures = []

    for script in train_scripts:
        cmd = [
            "python",
            script,
            "--total-timesteps",
            "128",
            "--num-envs",
            "8",
            "--num-steps",
            "16",
            "--env-config",
            env_config,
        ]

        # Add no-save-model for trainers that support it
        if "train_comm" not in script and "train_qmix" not in script:
            cmd.append("--no-save-model")

        # QMIX needs different args
        if "train_qmix" in script:
            cmd = [
                "python",
                script,
                "--total-steps",
                "8",
                "--train-freq",
                "2",
                "--env-config",
                env_config,
            ]
        elif "train_comm" in script:
            cmd = [
                "python",
                script,
                "--total-steps",
                "128",
                "--num-envs",
                "8",
                "--num-steps",
                "16",
                "--env-config",
                env_config,
            ]

        print(f"Running smoke test for {script}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            failures.append(
                f"---- {script} FAILED ----\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}\n"
            )

    if failures:
        pytest.fail("\n\n".join(failures))
