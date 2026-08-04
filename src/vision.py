"""
Vision and line-of-sight primitives for HEIST.

All heavy loops are JIT-compiled with Numba so that high-FPS headless
training rollouts are not throttled by the Python interpreter.  This
addresses the "Performance Horizon (Raycasting Bottleneck)" section of
PLAN.md: raycasting is now fully numba-vectorized and callable from plain
NumPy code.

Functions
---------
raycast : fill a Bresenham line into explored_map until a wall is hit
calculate_fov : reveal a radius around a position using raycast
line_is_clear : True when the Bresenham line between two tiles is unobstructed
camera_exposure : which cameras see which agents (vectorized over agents)
"""

import numpy as np
from numba import njit

from constants import DOOR, WALL


@njit(cache=True)
def raycast(grid, explored_map, start_r, start_c, target_r, target_c, wall_val):
    """Walk a Bresenham line from (start_r, start_c) to (target_r, target_c),
    marking every tile traversed as explored until a wall blocks the line."""
    r0, c0 = start_r, start_c
    r1, c1 = target_r, target_c

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        explored_map[r0, c0] = True

        if grid[r0, c0] == wall_val:
            break

        if r0 == r1 and c0 == c1:
            break

        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


@njit(cache=True)
def calculate_fov(grid, explored_map, start_r, start_c, radius, map_h, map_w, wall_val):
    """Reveal every tile within `radius` of (start_r, start_c), clipped to the
    map bounds.  Walls terminate each ray, which is what creates the fog."""
    min_row = max(start_r - radius, 0)
    max_row = min(start_r + radius, map_h - 1)
    min_col = max(start_c - radius, 0)
    max_col = min(start_c + radius, map_w - 1)

    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            raycast(grid, explored_map, start_r, start_c, r, c, wall_val)


@njit(cache=True)
def line_is_clear(grid, r0, c0, r1, c1, wall_val, door_val):
    """Return True when the tile at (r0, c0) can 'see' the tile at (r1, c1):
    the Bresenham line between them is not blocked by a wall or a locked door.

    The starting tile itself never blocks (a camera on the same tile as an
    agent would be a trivial edge case, and we treat the start as transparent).
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        if r == r1 and c == c1:
            return True
        if (r != r0 or c != c0) and (grid[r, c] == wall_val or grid[r, c] == door_val):
            return False
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False


def camera_exposure(
    grid, camera_positions, agent_positions, wall_val=WALL, door_val=DOOR, max_range=12
):
    """Return a boolean matrix [n_cameras, n_agents] indicating which cameras
    currently have a clear line of sight to which agents, restricted to
    `max_range` Manhattan distance.  Thin Python wrapper around the
    numba-compiled `line_is_clear`; camera counts are small (typically <= 4)
    so the overhead is negligible."""
    n_cam, n_agent = len(camera_positions), len(agent_positions)
    if n_cam == 0 or n_agent == 0:
        return np.zeros((n_cam, n_agent), dtype=bool)
    exposure = np.zeros((n_cam, n_agent), dtype=bool)
    for i, (cr, cc) in enumerate(camera_positions):
        for j, (ar, ac) in enumerate(agent_positions):
            if abs(cr - ar) + abs(cc - ac) > max_range:
                continue
            exposure[i, j] = line_is_clear(grid, cr, cc, ar, ac, wall_val, door_val)
    return exposure
