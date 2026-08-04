"""
Unit test for vision, line-of-sight, and Numba JIT search primitives.

Validates that:
1. get_valid_moves filters out walls and doors correctly.
2. pick_search_tile picks valid non-wall tiles within radius.
3. line_is_clear and calculate_fov mark line of sight and wall occlusion properly.
4. bfs_next_step and distance_to_nearest_target calculate shortest grid paths.

Run with: uv run pytest src/test_vision_smoke.py
"""

import numpy as np

from constants import DOOR, WALL
from vision import (
    bfs_next_step,
    calculate_fov,
    distance_to_nearest_target,
    get_valid_moves,
    line_is_clear,
    pick_search_tile,
)


def test_get_valid_moves():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[0, 1] = WALL  # top blocked by wall
    grid[1, 0] = DOOR  # left blocked by door

    moves = get_valid_moves(grid, r=1, c=1, wall_val=WALL, door_val=DOOR)
    # Valid moves should be down (2,1) and right (1,2)
    assert len(moves) == 2
    move_tuples = set(tuple(m) for m in moves)
    assert (2, 1) in move_tuples
    assert (1, 2) in move_tuples
    assert (0, 1) not in move_tuples
    assert (1, 0) not in move_tuples


def test_pick_search_tile():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[2, 2] = WALL

    # pick_search_tile with radius 1 around (2,2)
    tr, tc = pick_search_tile(
        grid,
        center_r=2,
        center_c=2,
        radius=1,
        wall_val=WALL,
        door_val=DOOR,
        rand_val=0.5,
    )
    assert (tr, tc) != (2, 2)
    assert grid[tr, tc] != WALL


def test_line_of_sight_and_fov():
    grid = np.zeros((7, 7), dtype=np.int32)
    grid[3, 3] = WALL  # wall in the center

    # Clear line of sight
    assert line_is_clear(grid, 1, 1, 1, 5, WALL, DOOR) is True

    # Blocked line of sight across wall
    assert line_is_clear(grid, 3, 1, 3, 5, WALL, DOOR) is False

    explored = np.zeros((7, 7), dtype=bool)
    calculate_fov(
        grid, explored, start_r=1, start_c=1, radius=2, map_h=7, map_w=7, wall_val=WALL
    )
    assert bool(explored[1, 1]) is True
    assert bool(explored[2, 2]) is True


def test_bfs_pathfinding():
    grid = np.zeros((5, 5), dtype=np.int32)
    grid[1, 1:4] = WALL  # horizontal wall barrier

    targets = np.array([[2, 2]], dtype=np.int32)
    dist_map = distance_to_nearest_target(grid, targets, wall_val=WALL, door_val=DOOR)
    assert dist_map[2, 2] == 0
    assert dist_map[0, 2] > 2  # Must route around wall

    next_r, next_c = bfs_next_step(
        grid, start_r=0, start_c=2, target_r=2, target_c=2, wall_val=WALL, door_val=DOOR
    )
    assert (next_r, next_c) in [(0, 1), (0, 3)]  # Step around wall
