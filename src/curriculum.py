"""
Curriculum learning support for HEIST.

The PLAN.md "Sparsity Wall" section anticipates that standard MARL
flatlines on complex procedural maps.  This module provides staged env
configs so training can start on tiny, guard-free, camera-free maps and
ramp up to the full 50x50 benchmark.

Each stage is a dict override for HeistEnv's DEFAULT_CONFIG.

Example:
    uv run python src/train_ippo.py --env-config '{"map_size": [11, 11], "guard_count": 0, "camera_count": 0}'
"""

# ---------------------------------------------------------------------------
# Manual curriculum stages (linear ramp)
# ---------------------------------------------------------------------------
CURRICULUM = [
    {
        # Stage 0: tiny room, no security system.  Agents learn the causal
        # chain (scout -> hacker -> extractor -> converge) with zero pressure.
        "map_size": (11, 11),
        "num_rooms_range": (1, 2),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
        "spawn_mode": "role",
    },
    {
        # Stage 1: small map, guards but no cameras.  Learn to avoid patrols.
        "map_size": (15, 15),
        "num_rooms_range": (2, 4),
        "guard_count": 2,
        "camera_count": 0,
        "door_count": 1,
        "max_steps": 120,
        "spawn_mode": "role",
    },
    {
        # Stage 2: medium map, cameras added.  Learn to disable the terminal.
        "map_size": (21, 21),
        "num_rooms_range": (3, 5),
        "guard_count": 3,
        "camera_count": 2,
        "door_count": 2,
        "max_steps": 180,
        "spawn_mode": "role",
    },
    {
        # Stage 3: near-full difficulty.
        "map_size": (35, 35),
        "num_rooms_range": (5, 8),
        "guard_count": 5,
        "camera_count": 3,
        "door_count": 3,
        "max_steps": 240,
        "spawn_mode": "role",
    },
    {
        # Stage 4: full benchmark.
        "map_size": (50, 50),
        "num_rooms_range": (8, 12),
        "guard_count": 6,
        "camera_count": 3,
        "door_count": 4,
        "max_steps": 300,
        "spawn_mode": "role",
    },
]

# ---------------------------------------------------------------------------
# Automatic curriculum driver
# ---------------------------------------------------------------------------
class Curriculum:
    """Advances the env config after a given number of env interactions.

    Usage:
        curriculum = Curriculum(stages=CURRICULUM, steps_per_stage=1_000_000)
        config = curriculum.config_for_step(global_step)
    """

    def __init__(self, stages=None, steps_per_stage=1_000_000):
        self.stages = stages if stages is not None else CURRICULUM
        self.steps_per_stage = steps_per_stage

    def config_for_step(self, global_step):
        stage_idx = min(global_step // self.steps_per_stage, len(self.stages) - 1)
        return self.stages[stage_idx]

    def stage_for_step(self, global_step):
        return min(global_step // self.steps_per_stage, len(self.stages) - 1)


def env_config_str(config: dict) -> str:
    """Serialize a stage dict into the --env-config JSON string format.

    Example:
        env_config_str(CURRICULUM[0])
        # -> '{"map_size": [11, 11], "num_rooms_range": [1, 2], ...}'
    """
    import json
    return json.dumps(config)


if __name__ == "__main__":
    print("Curriculum stages (as --env-config strings):")
    for i, stage in enumerate(CURRICULUM):
        print(f"  stage {i}: {env_config_str(stage)}")
