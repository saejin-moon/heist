"""
Smoke test for HEIST reward refinements.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from constants import INTERACT  # noqa: E402
from env import HeistEnv  # noqa: E402


def test_muscle_reward_deduplication():
    env = HeistEnv({"guard_count": 1})
    env.reset(seed=42)

    # Position Muscle next to guard 0
    gpos = env.guard_positions[0]
    env.agent_positions["muscle"] = (gpos[0], max(0, gpos[1] - 1))
    env.neutralized[0] = 0

    actions = {a: 4 for a in env.possible_agents}
    actions["muscle"] = INTERACT

    # First neutralization -> rewarded +2.0
    _, rewards1, _, _, _ = env.step(actions)
    assert rewards1["muscle"] >= 1.9

    # Fast forward neutralization timer
    env.neutralized[0] = 0
    # Second neutralization of SAME guard -> 0 additional task reward
    _, rewards2, _, _, _ = env.step(actions)
    assert rewards2["muscle"] < 1.0


def test_hacker_dense_hack_rewards():
    env = HeistEnv()
    env.reset(seed=42)

    # Place Hacker adjacent to terminal and tag terminal
    term_r, term_c = env.terminal_pos
    env.tagged_pois.add(env.terminal_pos)
    env.agent_positions["hacker"] = (term_r, max(0, term_c - 1))

    actions = {a: 4 for a in env.possible_agents}
    actions["hacker"] = INTERACT

    # Turn 1 hack -> +0.5 progress reward
    _, rewards1, _, _, _ = env.step(actions)
    assert env.hack_progress == 1
    assert rewards1["hacker"] > 0.4

    # Turn 2 hack -> +0.5 progress reward
    _, rewards2, _, _, _ = env.step(actions)
    assert env.hack_progress == 2
    assert rewards2["hacker"] > 0.4

    # Turn 3 hack -> completes hack (+1.0)
    _, rewards3, _, _, _ = env.step(actions)
    assert env.terminal_disabled is True
    assert rewards3["hacker"] > 0.9


def test_incremental_alarm_step_penalty():
    env = HeistEnv()
    env.reset(seed=42)

    prev_alarm = env.alarm
    r = {a: 0.0 for a in env.possible_agents}
    env._add_alarm(10.0, r)
    assert env.alarm == prev_alarm + 10.0
    assert r["scout"] == -0.1
