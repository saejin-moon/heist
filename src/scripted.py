"""Scripted near-optimal controller for HEIST.

Used for two purposes:
1. Validate the diagnostic metrics: a near-optimal team should show high,
   clean CAI/counterfactual signal, proving the metrics are not noise.
2. Debugging / sanity-checking the env: the controller wins ~29/30 at
   stage-0, demonstrating the environment is solvable.

The controller is stateful: it reads env internals (positions, grid,
phase flags) to BFS path to objectives, then executes the RG-Dec-POMDP
chain: scout tags terminal -> hacker disables it -> extractor secures
loot -> everyone converges on the extraction tile.
"""

from collections import deque

import numpy as np

from env import AGENTS, HeistEnv
from constants import (
    ACTION_DELTAS, INTERACT, UP, DOWN, LEFT, RIGHT, WAIT,
    WALL, DOOR, WIN_CONVERGE_RADIUS,
)


def bfs_next_step(grid, start, goal, blocked=None):
    """First move toward `goal` (BFS), avoiding walls/doors plus any
    extra blocked cells.  Returns an action id (UP/DOWN/LEFT/RIGHT) or
    None if already there / unreachable."""
    if start == goal:
        return None
    h, w = grid.shape
    blocked = blocked or set()
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for action in (UP, DOWN, LEFT, RIGHT):
            dr, dc = ACTION_DELTAS[action]
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] in (WALL, DOOR) or nxt in blocked:
                continue
            if nxt in prev:
                continue
            prev[nxt] = (action, cur)
            q.append(nxt)
    if goal not in prev:
        return None
    # walk back to find the first step from start
    node, first = goal, None
    while node != start:
        action, parent = prev[node]
        first = action
        node = parent
    return first


def _danger_zone(env):
    """Tiles a safe policy avoids: every non-neutralized guard and all
    tiles within catch distance of one."""
    blocked = set()
    for gi, gpos in enumerate(env.guard_positions):
        if env.neutralized[gi] > 0:
            continue
        gr, gc = gpos
        d = env.config["catch_distance"]
        for r in range(max(0, gr - d), min(env.map_h, gr + d + 1)):
            for c in range(max(0, gc - d), min(env.map_w, gc + d + 1)):
                blocked.add((r, c))
    return blocked


class ScriptedTeam:
    """Deterministic BFS controller.  `act()` returns {agent: action}."""

    def __init__(self, env: HeistEnv):
        self.env = env

    def _move_toward(self, agent, goal, avoid_guards=True):
        blocked = _danger_zone(self.env) if avoid_guards else set()
        step = bfs_next_step(self.env.grid, self.env.agent_positions[agent],
                             goal, blocked=blocked)
        if step is None:
            return WAIT
        return step

    def _near_goal(self, agent, goal):
        return (bfs_next_step(self.env.grid, self.env.agent_positions[agent], goal)
                is None)

    def _adjacent_door(self, pos):
        grid = self.env.grid
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.env.map_h and 0 <= nc < self.env.map_w \
                    and grid[nr, nc] == DOOR:
                return True
        return False

    def _hacker_clear_door(self):
        """Hacker moves to and bypasses a door blocking the terminal path."""
        env = self.env
        pos = env.agent_positions["hacker"]
        if self._adjacent_door(pos):
            return INTERACT
        # nearest door tile
        best, best_d = None, 1e9
        for dr, dc in env.door_positions:
            d = abs(pos[0] - dr) + abs(pos[1] - dc)
            if d < best_d:
                best_d, best = d, (dr, dc)
        if best is None:
            return WAIT
        # walk to a tile adjacent to the door, then bypass next step
        grid = env.grid
        best_adj, best_adj_d = None, 1e9
        for ar in range(max(0, best[0] - 1), min(env.map_h, best[0] + 2)):
            for ac in range(max(0, best[1] - 1), min(env.map_w, best[1] + 2)):
                if grid[ar, ac] in (WALL, DOOR):
                    continue
                d = abs(ar - pos[0]) + abs(ac - pos[1])
                if d < best_adj_d:
                    best_adj_d, best_adj = d, (ar, ac)
        if best_adj is None:
            return WAIT
        step = bfs_next_step(grid, pos, best_adj,
                             blocked=_danger_zone(env))
        return step if step is not None else WAIT

    def act(self):
        env = self.env
        blocked = _danger_zone(env)
        actions = {}
        for agent in AGENTS:
            pos = env.agent_positions[agent]
            if agent == "scout":
                # tag the terminal first, then converge on extract
                if env.terminal_pos not in env.tagged_pois:
                    if self._near_goal(agent, env.terminal_pos):
                        actions[agent] = INTERACT
                    else:
                        actions[agent] = self._move_toward(agent, env.terminal_pos)
                elif not self._within_win_radius(agent):
                    actions[agent] = self._move_toward(agent, env.extract_pos)
                else:
                    actions[agent] = WAIT
            elif agent == "hacker":
                # disable the terminal once the scout has tagged it
                if (env.terminal_pos in env.tagged_pois
                        and not env.terminal_disabled):
                    if self._near_goal(agent, env.terminal_pos):
                        actions[agent] = INTERACT
                    else:
                        step = self._move_toward(agent, env.terminal_pos)
                        if step == WAIT and self._adjacent_door(pos):
                            actions[agent] = INTERACT
                        else:
                            actions[agent] = step if step != WAIT \
                                else self._hacker_clear_door()
                elif not self._within_win_radius(agent):
                    actions[agent] = self._move_toward(agent, env.extract_pos)
                else:
                    actions[agent] = WAIT
            elif agent == "muscle":
                # neutralize a nearby guard, else converge on extract
                best_g = min(env.guard_positions,
                             key=lambda g: abs(pos[0] - g[0]) + abs(pos[1] - g[1]),
                             default=None)
                if best_g is not None:
                    gi = env.guard_positions.index(best_g)
                    if (env.neutralized[gi] == 0
                            and abs(pos[0] - best_g[0]) + abs(pos[1] - best_g[1]) <= 2):
                        actions[agent] = INTERACT
                        continue
                if not self._within_win_radius(agent):
                    actions[agent] = self._move_toward(agent, env.extract_pos)
                else:
                    actions[agent] = WAIT
            else:  # extractor
                if not env.loot_acquired:
                    if self._near_goal(agent, env.loot_pos):
                        actions[agent] = INTERACT
                    else:
                        actions[agent] = self._move_toward(agent, env.loot_pos)
                elif not env.extraction_triggered:
                    actions[agent] = INTERACT  # call extraction
                elif pos != env.extract_pos:
                    actions[agent] = self._move_toward(agent, env.extract_pos)
                else:
                    actions[agent] = WAIT
        return actions

    def _within_win_radius(self, agent):
        r, c = self.env.agent_positions[agent]
        er, ec = self.env.extract_pos
        return abs(r - er) + abs(c - ec) <= WIN_CONVERGE_RADIUS


def run_scripted_episode(env, seed=None, noop_agent=None):
    """One episode with the scripted team, mirroring `evaluate.run_episode`
    so CAI/counterfactual numbers are directly comparable."""
    env.reset(seed=seed)
    team = ScriptedTeam(env)
    total_return = 0.0
    length = 0
    credit = {a: 0.0 for a in AGENTS}
    terminal_reward = 0.0
    win = False
    for _ in range(env.config["max_steps"]):
        planned = team.act()
        if noop_agent is None:
            actions = planned
        else:
            actions = {a: (WAIT if a == noop_agent else planned[a]) for a in AGENTS}
        obs, rewards, terms, truncs, infos = env.step(actions)
        length += 1
        terminated = bool(any(terms.values()))
        truncated = bool(any(truncs.values()))
        done = terminated or truncated
        if terminated:
            terminal_reward = rewards[AGENTS[0]]
            win = bool(infos[AGENTS[0]].get("win", False))
        elif not done:
            for a in AGENTS:
                credit[a] += rewards[a]
        total_return += sum(rewards.values()) / len(AGENTS)
        if done:
            break
    return {
        "return": total_return,
        "length": length,
        "terminal_reward": terminal_reward,
        "win": win,
        "credit": credit,
        "alarm": env.alarm,
        "terminal_disabled": env.terminal_disabled,
        "loot_acquired": env.loot_acquired,
        "extraction_triggered": env.extraction_triggered,
        "hack_progress": env.hack_progress,
    }


def evaluate_scripted(env, episodes=30, seed=0, noop_agent=None):
    results = [run_scripted_episode(env, seed=seed + i, noop_agent=noop_agent)
               for i in range(episodes)]
    wins = sum(r["win"] for r in results)
    return {
        "win_rate": wins / episodes,
        "mean_return": float(np.mean([r["return"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "mean_alarm": float(np.mean([r["alarm"] for r in results])),
        "terminal_rate": float(np.mean([r["terminal_disabled"] for r in results])),
        "loot_rate": float(np.mean([r["loot_acquired"] for r in results])),
        "extraction_rate": float(np.mean([r["extraction_triggered"] for r in results])),
    }


def scripted_cai(env, episodes=50, seed=0):
    results = [run_scripted_episode(env, seed=seed + i) for i in range(episodes)]
    outcomes = np.array([r["terminal_reward"] for r in results])
    cai = {}
    for a in AGENTS:
        credits = np.array([r["credit"][a] for r in results])
        if outcomes.std() == 0 or credits.std() == 0:
            cai[a] = 0.0
        else:
            cai[a] = float(np.corrcoef(credits, outcomes)[0, 1])
    return cai


def scripted_counterfactual(env, episodes=30, seed=0):
    base = evaluate_scripted(env, episodes=episodes, seed=seed)
    importance = {}
    for a in AGENTS:
        r = evaluate_scripted(env, episodes=episodes, seed=seed, noop_agent=a)
        importance[a] = base["win_rate"] - r["win_rate"]
    return {"baseline_win_rate": base["win_rate"], "importance": importance}


if __name__ == "__main__":
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--env-config", type=str,
                    default='{"map_size": [11, 11], "num_rooms_range": [1, 2], '
                            '"guard_count": 0, "camera_count": 0, "door_count": 0, '
                            '"max_steps": 90}')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = HeistEnv(json.loads(args.env_config))
    print("=" * 64)
    print("HEIST scripted controller evaluation")
    print("=" * 64)
    m = evaluate_scripted(env, episodes=args.episodes, seed=args.seed)
    for k, v in m.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")
    print("-" * 64)
    cai = scripted_cai(env, episodes=args.episodes, seed=args.seed)
    print("Credit Attribution Index (scripted):")
    for a in AGENTS:
        print(f"  {a:>9}: {cai[a]:+.3f}")
    print("-" * 64)
    imp = scripted_counterfactual(env, episodes=max(args.episodes // 2, 10),
                                  seed=args.seed + 10_000)
    print("Counterfactual importance (scripted):")
    print(f"  baseline win rate: {imp['baseline_win_rate']:.3f}")
    for a in AGENTS:
        print(f"  {a:>9}: {imp['importance'][a]:+.3f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"metrics": m, "cai": cai, "counterfactual": imp,
                       "controller": "scripted_bfs",
                       "episodes": args.episodes, "seed": args.seed},
                      f, indent=2)
        print(f"\nsaved results to {args.out}")
