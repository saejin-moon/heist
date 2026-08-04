"""
HEIST: Hierarchical Environment for Interdependent Sequential Tasks.

A PettingZoo ParallelEnv implementation of the RG-Dec-POMDP described in
PLAN.md.  Four specialized agents must cooperate through a strict sequential
causal dependency chain (Scout reveals -> Hacker disables terminal -> Extractor
secures loot -> all converge at extraction) while a rule-based security system
(guards + cameras + incremental alarm meter) applies opposing pressure.

Environment contract (see PLAN.md):
  * observation  : 5x5 Fog-Masked local Box (gated by Scout reveals)
  * action_mask  : 6-element binary vector enforcing causal action gates
  * global_state : 4-element vector (step, alarm, terminal, loot)

The env is fully configurable (map size, guard/camera counts, reward scaling,
spawn placement, hack difficulty) so curriculum.py can stage difficulty.
"""

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Dict, Discrete, Box

from constants import *
from vision import calculate_fov, camera_exposure
from map_gen import generate_procedural_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def manhattan(a, b):
    """Manhattan distance between two (r, c) tuples/arrays."""
    return int(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def parse_env_config(env_config_str: str) -> dict | None:
    """Parse an --env-config JSON string into a config dict.

    Example:
        '{"map_size": [11, 11], "guard_count": 0, "spawn_mode": "role"}'

    JSON avoids the tuple/string ambiguity of comma-separated key=value
    strings.  curriculum.env_config_str() serializes stages into this form.
    """
    if not env_config_str:
        return None
    import json
    cfg = json.loads(env_config_str)
    # accept JSON arrays for tuple-typed config keys
    for k, v in cfg.items():
        if isinstance(v, list):
            cfg[k] = tuple(v)
    return cfg


DEFAULT_CONFIG = {
    "map_size": MAP_SIZE,
    "num_rooms_range": (8, 12),
    "guard_count": GUARD_COUNT,
    "camera_count": CAMERA_COUNT,
    "door_count": DOOR_COUNT,
    "max_steps": 300,
    "scout_vision": SCOUT_VISION_RADIUS,
    "agent_vision": AGENT_VISION_RADIUS,
    "hack_turns": HACK_TURNS,
    "extraction_countdown": EXTRACTION_COUNTDOWN,
    "spawn_mode": "role",            # "role" | "random"
    "catch_distance": CATCH_DISTANCE,
    "converge_alarm": CONVERGE_ALARM,
    "neutral_turns": NEUTRALIZE_TURNS,
    "alarm_camera": ALARM_CAMERA,
    "alarm_hack_turn": ALARM_HACK_TURN,
    "alarm_bypass": ALARM_BYPASS,
    "alarm_neutralize": ALARM_NEUTRALIZE,
    "alarm_guard_spot": ALARM_GUARD_SPOT,
    "alarm_extraction_timeout": ALARM_EXTRACTION_TIMEOUT,
    "alarm_max": ALARM_MAX,
    "camera_range": CAMERA_RANGE,
    "reward_win": REWARD_WIN,
    "reward_lose": REWARD_LOSE,
    "reward_task": REWARD_TASK,
    "reward_tag": REWARD_TAG,
    "reward_time_bleed": REWARD_TIME_BLEED,
    "converge_bonus": CONVERGE_BONUS,
    "converge_radius": CONVERGE_RADIUS,
    "win_converge_radius": WIN_CONVERGE_RADIUS,
}


class HeistEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "name": "heist_v0"}

    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.possible_agents = AGENTS[:]
        self.agents = self.possible_agents[:]

        self.map_h, self.map_w = self.config["map_size"]
        self.grid = np.zeros((self.map_h, self.map_w), dtype=np.int32)

        self.action_spaces = {a: Discrete(ACTION_SPACE_SIZE) for a in self.possible_agents}
        self.observation_spaces = {
            a: Dict({
                # fov is 5x5; FOG (-1) masks unrevealed tiles
                "observation": Box(low=FOG, high=255, shape=OBSERVATION_SIZE, dtype=np.int32),
                # 6-element binary causal action gate
                "action_mask": Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8),
                # step, alarm level (0-100), terminal disabled, loot acquired
                "global_state": Box(low=0, high=255, shape=(4,), dtype=np.int32),
            })
            for a in self.possible_agents
        }

        self.rng = np.random.default_rng(0)
        self.mode = "human"
        self.explored_map = np.zeros((self.map_h, self.map_w), dtype=bool)

        # state used by step()
        self.terminal_pos = (0, 0)
        self.loot_pos = (0, 0)
        self.extract_pos = (0, 0)
        self.camera_positions = []
        self.door_positions = []
        self.guard_positions = []
        self.neutralized = np.zeros(0, dtype=np.int32)
        self.agent_positions = {a: (0, 0) for a in self.possible_agents}

        self.current_step = 0
        self.alarm = 0.0
        self.terminal_disabled = False
        self.loot_acquired = False
        self.extraction_triggered = False
        self.extraction_countdown = self.config["extraction_countdown"]
        self.hack_progress = 0
        self.tagged_pois = set()
        self._prev_extract_dist = {}

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.alarm = 0.0
        self.terminal_disabled = False
        self.loot_acquired = False
        self.extraction_triggered = False
        self.extraction_countdown = self.config["extraction_countdown"]
        self.hack_progress = 0
        self.tagged_pois = set()
        self._prev_extract_dist = {}
        self.agents = self.possible_agents[:]
        self.explored_map = np.zeros((self.map_h, self.map_w), dtype=bool)

        # --- regenerate the procedural map ---
        map_data = generate_procedural_map(
            self.rng,
            map_size=(self.map_h, self.map_w),
            num_rooms_range=self.config["num_rooms_range"],
            camera_count=self.config["camera_count"],
            door_count=self.config["door_count"],
        )
        self.grid = map_data["grid"]
        self.terminal_pos = map_data["terminal_pos"]
        self.loot_pos = map_data["loot_pos"]
        self.extract_pos = map_data["extract_pos"]
        self.camera_positions = map_data["camera_positions"]
        self.door_positions = map_data["door_positions"]

        # --- spawn guards on empty tiles ---
        empty = [tuple(int(x) for x in c) for c in np.argwhere(self.grid == EMPTY)]
        guard_count = self.config["guard_count"]
        self.guard_positions = []
        if guard_count > 0 and len(empty) >= guard_count:
            self.rng.shuffle(empty)
            self.guard_positions = [empty.pop() for _ in range(guard_count)]
        self.neutralized = np.zeros(len(self.guard_positions), dtype=np.int32)

        # --- spawn agents ---
        used = set(self.guard_positions)
        self.agent_positions = self._spawn_agents(used)

        # --- initial reveals (Scout widest, everyone sees locally) ---
        self._refresh_scout_fov()
        for agent in self.possible_agents:
            self._reveal_around(agent, self.config["agent_vision"])

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        rewards = {a: self.config["reward_time_bleed"] for a in self.agents}

        # ------------------ agent actions ------------------
        for agent in list(self.agents):
            action = actions[agent]
            if action == WAIT:
                continue
            if action == INTERACT:
                self._special_action(agent, rewards)
                continue
            self._move_agent(agent, action)

        # interruption of a multi-turn hack resets progress and raises alarm
        if self.hack_progress > 0 and not self.terminal_disabled:
            if manhattan(self.agent_positions["hacker"], self.terminal_pos) > 1:
                self.hack_progress = 0
                self._add_alarm(self.config["alarm_hack_turn"])

        # ------------------ guards ------------------
        self._move_guards()

        # ------------------ cameras ------------------
        if not self.terminal_disabled and self.camera_positions:
            exposure = camera_exposure(
                self.grid, self.camera_positions,
                list(self.agent_positions.values()), WALL, DOOR,
                self.config["camera_range"],
            )
            n_visible = int(exposure.sum())
            self._add_alarm(self.config["alarm_camera"] * n_visible)

        # ------------------ catch check ------------------
        caught = self._check_caught()
        if caught:
            self._add_alarm(self.config["alarm_guard_spot"])

        # ------------------ extraction countdown ------------------
        if self.extraction_triggered:
            self.extraction_countdown -= 1
            # timeout fires exactly once: it is a penalty for a slow gather,
            # not an automatic loss
            if self.extraction_countdown == 0:
                self._add_alarm(self.config["alarm_extraction_timeout"])

        # ------------------ extraction-phase shaping (PBRS) ------------------
        # Once the loot is secured, steer every agent toward the extract tile.
        # Potential-based shaping with phi = -dist(agent, extract) is
        # policy-invariant (guaranteed not to change the optimal policy) while
        # giving the final convergence phase a dense gradient that the sparse
        # shared terminal reward cannot provide.  It is gated on loot_acquired
        # so it cannot be farmed before the heist's final phase.
        if self.loot_acquired:
            for a in self.agents:
                d_cur = manhattan(self.agent_positions[a], self.extract_pos)
                d_prev = self._prev_extract_dist.get(a, d_cur)
                rewards[a] += self.config["converge_bonus"] * (d_prev - d_cur)
                self._prev_extract_dist[a] = d_cur

        # ------------------ episode outcome ------------------
        win = self._win_condition()
        lose = self._lose_condition()
        if win:
            rewards = {a: self.config["reward_win"] for a in self.agents}
        elif lose:
            rewards = {a: self.config["reward_lose"] for a in self.agents}

        terminations = {a: bool(win or lose) for a in self.agents}
        truncations = {a: bool(self.current_step >= self.config["max_steps"]) for a in self.agents}
        infos = {a: {"alarm": self.alarm, "win": bool(win), "lose": bool(lose)} for a in self.agents}

        observations = {a: self._get_obs(a) for a in self.agents}

        # PettingZoo: once the episode is over, drop the agent handles
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.mode == "ansi":
            return self._render_ansi()
        print(self._render_ansi())

    def render_pygame(self, screen, active_agent=None, font=None):
        """Pygame renderer with fog, cameras, doors, and an alarm meter."""
        import pygame as pg
        screen.fill((0, 0, 0))
        for row in range(self.map_h):
            for col in range(self.map_w):
                tile = self.grid[row][col]
                x, y = col * TILE_SIZE, row * TILE_SIZE
                color = COLORS.get(tile)
                if color is not None:
                    pg_rect = (x, y, TILE_SIZE, TILE_SIZE)
                    pg.draw.rect(screen, color, pg_rect)
                    if tile == DOOR:
                        pg.draw.line(screen, (60, 40, 10),
                                     (x + 2, y + 2), (x + TILE_SIZE - 2, y + TILE_SIZE - 2), 3)
                    elif tile == CAMERA:
                        pg.draw.circle(screen, (0, 0, 0), (x + TILE_SIZE // 2, y + TILE_SIZE // 2), 3)

        # fog of war
        fog_surface = pg.Surface((self.map_w * TILE_SIZE, self.map_h * TILE_SIZE), pg.SRCALPHA)
        for row in range(self.map_h):
            for col in range(self.map_w):
                if not self.explored_map[row][col]:
                    pg.draw.rect(fog_surface, COLORS["EXPLORED"],
                                 (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        screen.blit(fog_surface, (0, 0))

        # agents
        for agent in self.agents:
            pos = self.agent_positions[agent]
            circle = (pos[1] * TILE_SIZE + TILE_SIZE // 2, pos[0] * TILE_SIZE + TILE_SIZE // 2)
            color = COLORS["AGENT"].get(agent)
            if agent == active_agent:
                pg.draw.circle(screen, (255, 255, 255), circle, TILE_SIZE // 2 + 4)
            pg.draw.circle(screen, color, circle, TILE_SIZE // 2)
            if agent == "extractor" and self.loot_acquired:
                pg.draw.circle(screen, (255, 215, 0), circle, TILE_SIZE // 4)

        # guards (skip neutralized ones)
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            circle = (gpos[1] * TILE_SIZE + TILE_SIZE // 2, gpos[0] * TILE_SIZE + TILE_SIZE // 2)
            pg.draw.circle(screen, COLORS[GUARD], circle, TILE_SIZE // 2)

        # alarm meter
        bar_w = self.map_w * TILE_SIZE
        pg.draw.rect(screen, (40, 40, 40), (0, self.map_h * TILE_SIZE, bar_w, 10))
        pg.draw.rect(screen, (255, 60, 60),
                     (0, self.map_h * TILE_SIZE, int(bar_w * min(self.alarm, ALARM_MAX) / ALARM_MAX), 10))
        if font is not None:
            status = f"step {self.current_step} | alarm {self.alarm:.0f}/100"
            if self.terminal_disabled:
                status += " | terminal: DISABLED"
            if self.loot_acquired:
                status += " | loot: SECURED"
            if self.extraction_triggered:
                status += f" | extract: {self.extraction_countdown}"
            screen.blit(font.render(status, True, (255, 255, 255)), (4, self.map_h * TILE_SIZE + 14))

    # ------------------------------------------------------------------
    # Internals: spawning
    # ------------------------------------------------------------------
    def _spawn_agents(self, used):
        empty = [tuple(int(x) for x in c) for c in np.argwhere(self.grid == EMPTY)]
        positions = {}
        if self.config["spawn_mode"] == "role":
            # role spawns put the causal chain in motion quickly
            order = [
                ("scout", self.terminal_pos),
                ("hacker", self.terminal_pos),
                ("muscle", self.loot_pos),
                ("extractor", self.loot_pos),
            ]
            for agent, target in order:
                pos = self._nearest_empty(target, used)
                used.add(pos)
                positions[agent] = pos
        else:
            self.rng.shuffle(empty)
            for agent in self.possible_agents:
                pos = empty.pop()
                while pos in used and empty:
                    pos = empty.pop()
                used.add(pos)
                positions[agent] = pos
        return positions

    def _nearest_empty(self, target, used):
        best, best_d = None, 1e9
        for r in range(self.map_h):
            for c in range(self.map_w):
                if self.grid[r, c] != EMPTY:
                    continue
                pos = (r, c)
                if pos in used:
                    continue
                d = manhattan(pos, target)
                if d < best_d:
                    best_d, best = d, pos
        return best if best is not None else (0, 0)

    # ------------------------------------------------------------------
    # Internals: movement and reveals
    # ------------------------------------------------------------------
    def _move_agent(self, agent, action):
        row, col = self.agent_positions[agent]
        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
            return
        tile = self.grid[nr, nc]
        if tile == WALL or tile == DOOR:
            return
        self.agent_positions[agent] = (nr, nc)
        self._reveal_around(agent, self.config["agent_vision"])
        if agent == "scout":
            self._refresh_scout_fov()

    def _refresh_scout_fov(self):
        sr, sc = self.agent_positions["scout"]
        calculate_fov(self.grid, self.explored_map, sr, sc,
                      self.config["scout_vision"], self.map_h, self.map_w, WALL)

    def _reveal_around(self, agent, radius):
        r, c = self.agent_positions[agent]
        calculate_fov(self.grid, self.explored_map, r, c,
                      radius, self.map_h, self.map_w, WALL)

    # ------------------------------------------------------------------
    # Internals: role special actions (action 5)
    # ------------------------------------------------------------------
    def _special_action(self, agent, rewards):
        pos = self.agent_positions[agent]
        if agent == "scout":
            self._scout_tag(pos, rewards)
        elif agent == "hacker":
            self._hacker_hack(pos, rewards)
        elif agent == "muscle":
            self._muscle_neutralize(pos, rewards)
        elif agent == "extractor":
            self._extractor_act(pos, rewards)

    def _scout_tag(self, pos, rewards):
        """Scout broadcasts intel on a nearby point of interest.

        Each point of interest can only be tagged once per episode.  Without
        this guard the scout can stand next to a single POI and farm +0.5 per
        step, which collapses the policy into reward-hacking instead of
        executing the causal chain.
        """
        pois = [self.terminal_pos, self.loot_pos, self.extract_pos] \
               + self.camera_positions + self.door_positions
        for p in pois:
            if p not in self.tagged_pois and manhattan(pos, p) <= 1:
                self.tagged_pois.add(p)
                rewards["scout"] += self.config["reward_tag"]

    def _hacker_hack(self, pos, rewards):
        """Multi-turn terminal hack; interruption resets progress.

        The terminal must first be tagged by the scout (causal gate), so
        the scout's action is a strict prerequisite for the whole chain.
        """
        if (not self.terminal_disabled and self.terminal_pos in self.tagged_pois
                and manhattan(pos, self.terminal_pos) <= 1):
            self.hack_progress += 1
            self._add_alarm(self.config["alarm_hack_turn"])
            if self.hack_progress >= self.config["hack_turns"]:
                self.terminal_disabled = True
                self.hack_progress = 0
                rewards["hacker"] += self.config["reward_task"]
            return
        # fallback: force-bypass an adjacent locked door
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < self.map_h and 0 <= nc < self.map_w and self.grid[nr, nc] == DOOR:
                self.grid[nr, nc] = EMPTY
                self._add_alarm(self.config["alarm_bypass"])
                return

    def _muscle_neutralize(self, pos, rewards):
        """Temporarily remove a nearby guard at a delayed alarm cost."""
        best, best_d = None, 1e9
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            d = manhattan(pos, gpos)
            if d < best_d:
                best_d, best = d, gi
        if best is not None and best_d <= 2:
            self.neutralized[best] = self.config["neutral_turns"]
            self._add_alarm(self.config["alarm_neutralize"])
            rewards["muscle"] += self.config["reward_task"]

    def _extractor_act(self, pos, rewards):
        """Secure loot (needs disabled terminal), then call extraction."""
        if not self.loot_acquired and self.terminal_disabled and manhattan(pos, self.loot_pos) <= 1:
            self.loot_acquired = True
            rewards["extractor"] += self.config["reward_task"]
            return
        if self.loot_acquired and not self.extraction_triggered:
            self.extraction_triggered = True
            self.extraction_countdown = self.config["extraction_countdown"]
            rewards["extractor"] += 0.5

    # ------------------------------------------------------------------
    # Internals: guards and alarm
    # ------------------------------------------------------------------
    def _valid_moves(self, r, c):
        moves = []
        for dr, dc in ACTION_DELTAS.values():
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
                continue
            tile = self.grid[nr, nc]
            if tile != WALL and tile != DOOR:
                moves.append((nr, nc))
        return moves

    def _move_guards(self):
        converge = self.alarm >= self.config["converge_alarm"]
        new_positions = []
        for gi, (gr, gc) in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                self.neutralized[gi] -= 1
                new_positions.append((gr, gc))
                continue
            valid = self._valid_moves(gr, gc)
            if not valid:
                new_positions.append((gr, gc))
                continue
            if converge:
                target = min(self.agent_positions.values(),
                             key=lambda p: manhattan(p, (gr, gc)))
                new_positions.append(min(valid, key=lambda p: manhattan(p, target)))
            else:
                new_positions.append(valid[int(self.rng.integers(len(valid)))])
        self.guard_positions = new_positions

    def _check_caught(self):
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            for apos in self.agent_positions.values():
                if manhattan(gpos, apos) <= self.config["catch_distance"]:
                    return True
        return False

    def _add_alarm(self, amount):
        self.alarm = min(self.alarm + amount, self.config["alarm_max"])

    def _win_condition(self):
        return (
            self.loot_acquired
            and self.extraction_triggered
            # the extractor must carry the loot out through the tile itself
            and self.agent_positions["extractor"] == self.extract_pos
            # everyone else gathers within the convergence zone
            and all(
                manhattan(self.agent_positions[a], self.extract_pos)
                <= self.config["win_converge_radius"]
                for a in self.possible_agents
            )
        )

    def _lose_condition(self):
        return self.alarm >= self.config["alarm_max"]

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _action_mask(self, agent):
        mask = np.ones(ACTION_SPACE_SIZE, dtype=np.int8)
        mask[INTERACT] = 0
        pos = self.agent_positions[agent]

        # block movement into walls / locked doors / out of bounds
        for a in range(4):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
                mask[a] = 0
            elif self.grid[nr, nc] == WALL or self.grid[nr, nc] == DOOR:
                mask[a] = 0

        # causal gate: only allow INTERACT when it is actually possible
        if agent == "scout":
            pois = [self.terminal_pos, self.loot_pos, self.extract_pos] \
                   + self.camera_positions + self.door_positions
            if any(manhattan(pos, p) <= 1 for p in pois):
                mask[INTERACT] = 1
        elif agent == "hacker":
            # causal gate: the terminal must have been tagged by the scout
            # before the hacker can act on it (RG-Dec-POMDP chain step 1)
            near_terminal = (not self.terminal_disabled
                             and self.terminal_pos in self.tagged_pois
                             and manhattan(pos, self.terminal_pos) <= 1)
            near_door = any(
                0 <= pos[0] + dr < self.map_h and 0 <= pos[1] + dc < self.map_w
                and self.grid[pos[0] + dr, pos[1] + dc] == DOOR
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))
            )
            if near_terminal or near_door:
                mask[INTERACT] = 1
        elif agent == "muscle":
            if any(self.neutralized[gi] == 0 and manhattan(pos, gpos) <= 2
                   for gi, gpos in enumerate(self.guard_positions)):
                mask[INTERACT] = 1
        elif agent == "extractor":
            near_loot = (not self.loot_acquired and self.terminal_disabled
                         and manhattan(pos, self.loot_pos) <= 1)
            can_call = self.loot_acquired and not self.extraction_triggered
            if near_loot or can_call:
                mask[INTERACT] = 1
        return mask

    def _get_obs(self, agent):
        composite = self.grid.copy()
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] == 0:
                composite[gpos[0], gpos[1]] = GUARD
        for other, opos in self.agent_positions.items():
            if other != agent:
                composite[opos[0], opos[1]] = ALLY

        pos = self.agent_positions[agent]
        pad = OBSERVATION_SIZE[0] // 2  # 2 -> 5x5 window
        pad_grid = np.pad(composite, pad_width=pad, mode="constant", constant_values=WALL)
        vpos = (pos[0] + pad, pos[1] + pad)
        obs = pad_grid[vpos[0] - pad:vpos[0] + pad + 1, vpos[1] - pad:vpos[1] + pad + 1]

        pad_explored = np.pad(self.explored_map, pad_width=pad, mode="constant",
                              constant_values=False)
        obs_explored = pad_explored[vpos[0] - pad:vpos[0] + pad + 1,
                                    vpos[1] - pad:vpos[1] + pad + 1]
        masked = np.where(obs_explored, obs, FOG)

        mask = self._action_mask(agent)
        pr, pc = pos
        global_state = np.array([
            self.current_step,
            int(min(self.alarm, 100)),
            int(self.terminal_disabled),
            int(self.loot_acquired),
            # relative bearings to the three objectives (navigation signal)
            self.terminal_pos[0] - pr, self.terminal_pos[1] - pc,
            self.loot_pos[0] - pr, self.loot_pos[1] - pc,
            self.extract_pos[0] - pr, self.extract_pos[1] - pc,
        ], dtype=np.int32)

        return {"observation": masked, "action_mask": mask, "global_state": global_state}

    def state(self):
        """Rich global state for centralized critics (MAPPO) / QMIX mixing.

        Concatenates scalar phase indicators, the full grid, all agent
        positions, all guard positions, and neutralization timers.
        """
        parts = [np.array([
            self.current_step, self.alarm, int(self.terminal_disabled),
            int(self.loot_acquired), int(self.extraction_triggered),
            self.extraction_countdown,
        ], dtype=np.float32)]
        parts.append(self.grid.astype(np.float32).ravel())
        for a in self.possible_agents:
            parts.append(np.array(self.agent_positions[a], dtype=np.float32))
        for gpos in self.guard_positions:
            parts.append(np.array(gpos, dtype=np.float32))
        parts.append(self.neutralized.astype(np.float32))
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------
    def _render_ansi(self):
        map_dict = {
            EMPTY: " ", WALL: "#", TERMINAL: "T", LOOT: "$",
            EXTRACT: "@", GUARD: "G", CAMERA: "C", DOOR: "D",
        }
        v_map = np.vectorize(map_dict.get, otypes=[str])
        char_grid = v_map(self.grid).copy()
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] == 0:
                char_grid[gpos[0], gpos[1]] = "G"
        for agent in self.agents:
            pos = self.agent_positions[agent]
            char_grid[pos[0], pos[1]] = AGENT_CHAR[agent]
        lines = ["".join(row) for row in char_grid]
        status = (f"step={self.current_step} alarm={self.alarm:.1f}/100 "
                  f"terminal={'off' if self.terminal_disabled else 'on'} "
                  f"loot={'yes' if self.loot_acquired else 'no'} "
                  f"extract={'calling' if self.extraction_triggered else 'idle'}")
        return "\n".join(lines) + "\n" + status

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
