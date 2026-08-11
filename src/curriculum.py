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
        # Stage 0: tiny room (11x11), no security. Learn basic causal chain.
        "map_size": (11, 11),
        "num_rooms_range": (1, 2),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 60,
        "spawn_mode": "role",
    },
    {
        # Stage 1: small map (17x17), 1 guard, no cameras. Learn patrol avoidance.
        "map_size": (17, 17),
        "num_rooms_range": (2, 4),
        "guard_count": 1,
        "camera_count": 0,
        "door_count": 1,
        "max_steps": 120,
        "spawn_mode": "role",
    },
    {
        # Stage 2: medium map (25x25), 2 guards, 1 camera. Learn camera evasion & terminal hack.
        "map_size": (25, 25),
        "num_rooms_range": (3, 5),
        "guard_count": 2,
        "camera_count": 1,
        "door_count": 2,
        "max_steps": 180,
        "spawn_mode": "role",
    },
    {
        # Stage 3: large facility (35x35), 3 guards, 2 cameras. Multi-security coordination.
        "map_size": (35, 35),
        "num_rooms_range": (5, 8),
        "guard_count": 3,
        "camera_count": 2,
        "door_count": 3,
        "max_steps": 240,
        "spawn_mode": "role",
    },
    {
        # Stage 4: full benchmark facility (50x50), 4 guards, 3 cameras. Full facility heist.
        "map_size": (50, 50),
        "num_rooms_range": (8, 12),
        "guard_count": 4,
        "camera_count": 3,
        "door_count": 4,
        "max_steps": 300,
        "spawn_mode": "role",
    },
]


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
