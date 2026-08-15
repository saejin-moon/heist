"""Smoke tests for M1 mechanics (REV-5/6/8/9).

Run:  PYTHONPATH=src .venv/bin/python src/test_mechanics_smoke.py
"""

from constants import BREACH, EMPTY, INTERACT, UP, WAIT, WALL
from env import HeistEnv, manhattan


def reset_open(seed=0):
    """An env with no guards/cameras so we control the exact layout."""
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 0,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 200,
            "spawn_mode": "random",
        }
    )
    env.reset(seed=seed)
    return env


def test_wall_breach():
    env = reset_open(1)
    # teleport the muscle next to a wall
    _muscle = env.agent_positions["muscle"]
    walls = [
        (r, c)
        for r in range(1, 10)
        for c in range(1, 10)
        if env.grid[r, c] == WALL and env.grid[r - 1, c] == EMPTY
    ]
    wr, wc = walls[0]
    env.agent_positions["muscle"] = (wr - 1, wc)
    # muscle is at (wr-1, wc); check walls adjacent to muscle
    mr, mc = wr - 1, wc
    adj_before = {
        (mr + dr, mc + dc): env.grid[mr + dr, mc + dc]
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
        if 0 <= mr + dr < 11 and 0 <= mc + dc < 11
    }
    n_walls_before = sum(1 for v in adj_before.values() if v == WALL)
    mask = env._action_mask("muscle")
    assert mask[BREACH] == 1, f"breach should be allowed adjacent to wall: {mask}"
    env.alarm = 0.0
    env.step({"scout": WAIT, "hacker": WAIT, "muscle": BREACH, "extractor": WAIT})
    n_walls_after = sum(
        1 for (nr, nc), v in adj_before.items() if env.grid[nr, nc] == WALL
    )
    assert n_walls_before >= 1 and n_walls_after == n_walls_before - 1, (
        "breach should destroy exactly one adjacent wall"
    )
    assert env.alarm >= 29.9, f"breach alarm should jump ~30, got {env.alarm}"
    print(f"  wall breach OK (alarm -> {env.alarm:.0f})")


def test_delayed_neutralize_alarm():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 1,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 100,
            "spawn_mode": "random",
        }
    )
    obs, _ = env.reset(seed=3)
    gpos = env.guard_positions[0]
    # teleport muscle next to the guard
    env.agent_positions["muscle"] = (gpos[0] + 1, gpos[1])
    env.alarm = 0.0
    assert env._action_mask("muscle")[INTERACT] == 1
    env.step({"scout": WAIT, "hacker": WAIT, "muscle": INTERACT, "extractor": WAIT})
    assert env.alarm == 0.0, "neutralize alarm must be DELAYED, not instant"
    assert env.neutralized[0] > 0, "guard should be neutralized"
    pending = env._pending_events
    assert (
        pending and pending[0][0] == env.current_step + env.config["alarm_neut_delay"]
    ), f"expected delayed event, got {pending}"
    fired = False
    for _ in range(env.config["alarm_neut_delay"] + 1):
        env.step({"scout": WAIT, "hacker": WAIT, "muscle": WAIT, "extractor": WAIT})
        if env.alarm > 0:
            fired = True
            break
    assert fired, "delayed alarm should fire ALARM_NEUTRALIZE_DELAY turns later"
    print(
        f"  delayed neutralize OK (fired at step {env.current_step}, alarm={env.alarm:.0f})"
    )


def test_extractor_burden():
    env = reset_open(2)
    env.loot_acquired = True
    env.current_step = 3
    env._burden_start_step = 2
    env.agent_positions["extractor"] = (5, 5)
    env.grid[4, 5] = EMPTY
    mask = env._action_mask("extractor")
    assert mask[UP] == 1, f"extractor should not be blocked: {mask}"
    env.step({"scout": WAIT, "hacker": WAIT, "muscle": WAIT, "extractor": UP})
    assert env.agent_positions["extractor"] == (4, 5), (
        "extractor should move freely every turn"
    )
    print(f"  extractor burden OK (moves freely to {env.agent_positions['extractor']})")


def test_guard_fsm():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 1,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 300,
            "spawn_mode": "random",
        }
    )
    obs, _ = env.reset(seed=7)
    gpos = env.guard_positions[0]
    # put an agent in clear line of sight: same row, no wall between
    env.agent_positions["scout"] = (gpos[0], gpos[1] + 3)
    for c in range(gpos[1] + 1, gpos[1] + 4):
        env.grid[gpos[0], c] = EMPTY
    env.step({"scout": WAIT, "hacker": WAIT, "muscle": WAIT, "extractor": WAIT})
    assert env.guard_states[0] == "search", (
        f"guard should switch to Search after spotting, got {env.guard_states[0]}"
    )
    print(
        f"  guard FSM OK (spotted -> {env.guard_states[0]}, target {env._guard_search_target[0]})"
    )


def test_breach_triggers_guard_search():
    env = HeistEnv(
        {
            "map_size": (11, 11),
            "guard_count": 2,
            "camera_count": 0,
            "door_count": 0,
            "max_steps": 300,
            "spawn_mode": "random",
        }
    )
    obs, _ = env.reset(seed=9)
    # find a wall for breach; muscle needs no guard within 2 tiles
    _muscle = env.agent_positions["muscle"]
    walls = [
        (r, c)
        for r in range(1, 10)
        for c in range(1, 10)
        if env.grid[r, c] == WALL and env.grid[r - 1, c] == EMPTY
    ]
    wr, wc = walls[0]
    env.agent_positions["muscle"] = (wr - 1, wc)
    # place guard 0 within breach_radius but far from muscle (> 2 tiles)
    env.guard_positions[0] = (min(wr + 4, 10), min(wc + 4, 10))
    # guard 1 in the farthest in-bounds corner (outside breach_radius)
    corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
    env.guard_positions[1] = max(corners, key=lambda p: manhattan(p, (wr - 1, wc)))
    env.guard_states = ["patrol", "patrol"]
    env._guard_search_target = [None, None]
    env._guard_search_turns = [0, 0]
    env.step({"scout": WAIT, "hacker": WAIT, "muscle": BREACH, "extractor": WAIT})
    assert env.guard_states[0] == "search", (
        f"nearby guard should Search after breach, got {env.guard_states[0]}"
    )
    assert env.guard_states[1] == "patrol", (
        f"far guard stays Patrol, got {env.guard_states[1]}"
    )
    print(
        f"  breach->search OK (near={env.guard_states[0]}, far={env.guard_states[1]})"
    )


if __name__ == "__main__":
    print("M1 mechanics smoke:")
    test_wall_breach()
    test_delayed_neutralize_alarm()
    test_extractor_burden()
    test_guard_fsm()
    test_breach_triggers_guard_search()
    print("ALL M1 MECHANICS PASSED")
