"""
HEIST: Hierarchical Environment for Interdependent Sequential Tasks.

A PettingZoo ParallelEnv implementation of the RG-Dec-POMDP described in
PLAN.md.  Four specialized agents must cooperate through a strict sequential
causal dependency chain (Scout reveals -> Hacker disables terminal -> Extractor
secures loot -> all converge at extraction) while a rule-based security system
(guards + cameras + incremental alarm meter) applies opposing pressure.

Observation contract (see REVISION_PLAN.md, PLAN.md):
  * observation   : 5x5 Fog-Masked local Box (gated by Scout reveals)
  * action_mask   : 6-element binary vector enforcing causal action gates
  * role_id       : 4-element one-hot identifying the agent's role (REV-2)

REV-7 (REVISION_PLAN.md §6): global_state was deleted from this per-agent
contract.  Centralized critics (MAPPO/QMIX) consume env.state() instead;
agents communicate phase status via learned TarMAC messages (train_comm.py).

Mechanics implemented from the approved roadmap:
  * REV-5 : Muscle wall breach (instant alarm, guard repath)
  * REV-6 : Extractor loot-carry burden (1 tile per 2 turns)
  * REV-8 : Guard directional LOS + Patrol/Search/Converge + BFS
  * REV-9 : Delayed alarm event queue (15-turn post-neutralization spike)

The env is fully configurable (map size, guard/camera counts, reward scaling,
spawn placement, hack difficulty) so curriculum.py can stage difficulty.
"""

import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from pettingzoo import ParallelEnv

from constants import *
from map_gen import generate_procedural_map
from vision import (
    bfs_next_step,
    bfs_shortest_path_distance,
    calculate_fov,
    camera_exposure,
    distance_to_nearest_target,
    get_valid_moves,
    line_is_clear,
    next_step_from_distance,
    pick_search_tile,
)


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
    "spawn_mode": "role",  # "role" | "random"
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
    # REV-5 wall breach
    "alarm_breach": ALARM_BREACH,
    "breach_radius": BREACH_RADIUS,
    "breach_search_trigger": BREACH_SEARCH_TRIGGER,
    # REV-6 extractor burden
    "extractor_burden": EXTRACTOR_BURDEN_TURNS,
    # REV-9 delayed alarm
    "alarm_neut_delay": ALARM_NEUTRALIZE_DELAY,
    # REV-8 guard AI
    "guard_los_range": GUARD_LOS_RANGE,
    "search_radius": SEARCH_RADIUS,
    "search_turns": SEARCH_TURNS,
    # Side-tasks option (disabled by default)
    "enable_side_tasks": False,
}


class HeistEnv(ParallelEnv):
    def __init__(self, config=None):
        self.config = dict(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.possible_agents = AGENTS[:]
        self.agents = self.possible_agents[:]

        self.map_h, self.map_w = self.config["map_size"]
        self.grid = np.zeros((self.map_h, self.map_w), dtype=np.int32)

        # REV-1: fixed Box shape and bounds (kept for backward compat during
        # M0/M1; REV-7 removes global_state entirely - see below).
        self.action_spaces = {
            a: Discrete(ACTION_SPACE_SIZE) for a in self.possible_agents
        }
        self.observation_spaces = {
            a: Dict(
                {
                    "observation": Box(
                        low=FOG, high=255, shape=OBSERVATION_SIZE, dtype=np.int32
                    ),
                    "action_mask": Box(
                        low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
                    ),
                    # REV-7 (REVISION_PLAN.md §6): global_state is deleted from the
                    # per-agent observation contract.  Centralized critics keep
                    # env.state().  Agents must communicate phase status instead.
                    "role_id": Box(low=0, high=1, shape=(N_AGENTS,), dtype=np.int8),
                }
            )
            for a in self.possible_agents
        }

        self.rng = np.random.default_rng(0)

        # Pre-allocated arrays for O(1) BFS in guards logic
        max_cells = self.map_h * self.map_w
        self._distance_map = np.full((self.map_h, self.map_w), -1, dtype=np.int32)
        self._bfs_queue = np.empty(max_cells, dtype=np.int32)
        self._bfs_previous = np.full(max_cells, -1, dtype=np.int32)
        self._bfs_reset = np.empty(max_cells, dtype=np.int32)
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
        self.beacon_calibrated = False

        self._rewarded_neutralized_guards = set()
        self._rewarded_breached_walls = set()

        # REV-5: breach coordinate (set when Muscle breaks a wall)

        # REV-6: extractor burden start step
        self._burden_start_step = -1

        # REV-9: delayed-alarm event queue [(trigger_step, alarm_amount), ...]
        self._pending_events = []

        # REV-8: per-guard FSM state ("patrol" | "search" | "converge")
        self.guard_states = []
        self._guard_search_target = []  # (r,c) or None per guard
        self._guard_search_turns = []  # remaining search steps per guard

        self._pad = OBSERVATION_SIZE[0] // 2
        self._pad_grid_cache = np.full(
            (self.map_h + 2 * self._pad, self.map_w + 2 * self._pad),
            WALL,
            dtype=np.int32,
        )
        self._pad_explored_cache = np.full(
            (self.map_h + 2 * self._pad, self.map_w + 2 * self._pad), False, dtype=bool
        )

        state_dim = (
            6
            + (self.map_h * self.map_w)
            + (len(self.possible_agents) * 2)
            + (self.config["guard_count"] * 2)
            + self.config["guard_count"]
        )
        self._state_buffer = np.zeros(state_dim, dtype=np.float32)

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------
    def reset(self, seed=None, _options=None):
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
        self.beacon_calibrated = False
        self._rewarded_neutralized_guards = set()
        self._rewarded_breached_walls = set()
        self.agents = self.possible_agents[:]
        self.explored_map = np.zeros((self.map_h, self.map_w), dtype=bool)
        # REV-5/6/9/8: clear milestone state on every reset
        self._burden_start_step = -1
        self._pending_events = []
        self._neutralized_pos = {}
        self.guard_states = []
        self._guard_search_target = []
        self._guard_search_turns = []

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
        self.guard_states = ["patrol"] * len(self.guard_positions)
        self._guard_search_target = [None] * len(self.guard_positions)
        self._guard_search_turns = [0] * len(self.guard_positions)
        self._neutralized_pos = {}

        # --- spawn agents ---
        used = set(self.guard_positions)
        self.agent_positions = self._spawn_agents(used)

        # --- initial reveals (Scout widest, everyone sees locally) ---
        self._refresh_scout_fov()
        for agent in self.possible_agents:
            self._reveal_around(agent, self.config["agent_vision"])

        observations = self._get_all_obs()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        rewards = {a: self.config["reward_time_bleed"] for a in self.agents}

        # --- CAR Tracking: Capture pre-step global state ---
        old_tagged_pois_len = len(self.tagged_pois)
        old_terminal_disabled = self.terminal_disabled
        old_loot_acquired = self.loot_acquired
        old_extraction_triggered = self.extraction_triggered
        old_hack_progress = self.hack_progress
        old_grid = self.grid.copy()

        interact_actors = [
            a for a in self.agents if actions.get(a) in (INTERACT, BREACH)
        ]

        # REV-9: fire any delayed alarms (neutralization notices) due this turn
        self._process_pending_events(rewards)

        # ------------------ agent actions ------------------
        for agent in list(self.agents):
            action = actions[agent]
            if action == WAIT:
                continue
            if action == INTERACT:
                self._special_action(agent, rewards)
                continue
            if action == BREACH:
                if agent == "muscle":
                    self._muscle_breach(self.agent_positions[agent], rewards)
                continue
            self._move_agent(agent, action)

        # interruption of a multi-turn hack resets progress and raises alarm
        if (
            self.hack_progress > 0
            and not self.terminal_disabled
            and manhattan(self.agent_positions["hacker"], self.terminal_pos) > 1
        ):
            self.hack_progress = 0
            self._add_alarm(self.config["alarm_hack_turn"], rewards)

        # ------------------ guards ------------------
        self._move_guards()

        # ------------------ cameras ------------------
        if not self.terminal_disabled and self.camera_positions:
            exposure = camera_exposure(
                self.grid,
                self.camera_positions,
                list(self.agent_positions.values()),
                WALL,
                DOOR,
                self.config["camera_range"],
            )
            n_visible = int(exposure.sum())
            self._add_alarm(self.config["alarm_camera"] * n_visible, rewards)

        # ------------------ catch check ------------------
        caught = self._check_caught()
        if caught:
            self._add_alarm(self.config["alarm_guard_spot"], rewards)

        # ------------------ extraction countdown ------------------
        if self.extraction_triggered:
            self.extraction_countdown -= 1
            # timeout fires exactly once: it is a penalty for a slow gather,
            # not an automatic loss
            if self.extraction_countdown == 0:
                self._add_alarm(self.config["alarm_extraction_timeout"], rewards)

        # ------------------ extraction-phase shaping (PBRS) ------------------
        # Once the loot is secured, steer every agent toward the extract tile.
        # Potential-based shaping with phi = -dist(agent, extract) is
        # policy-invariant (guaranteed not to change the optimal policy) while
        # giving the final convergence phase a dense gradient that the sparse
        # shared terminal reward cannot provide.  It is gated on loot_acquired
        # so it cannot be farmed before the heist's final phase.
        if self.loot_acquired or self.extraction_triggered:
            for a in self.agents:
                d_cur = manhattan(self.agent_positions[a], self.extract_pos)
                d_prev = self._prev_extract_dist.get(a, d_cur)
                rewards[a] += self.config["converge_bonus"] * (d_prev - d_cur)
                if d_cur <= self.config["converge_radius"]:
                    rewards[a] += 0.05
                self._prev_extract_dist[a] = d_cur

        # ------------------ episode outcome ------------------
        win = self._win_condition()
        lose = self._lose_condition()
        if win:
            rewards = {a: self.config["reward_win"] for a in self.agents}
        elif lose:
            rewards = {a: self.config["reward_lose"] for a in self.agents}

        terminations = {a: bool(win or lose) for a in self.agents}
        truncations = {
            a: bool(self.current_step >= self.config["max_steps"]) for a in self.agents
        }

        # --- CAR Tracking: Detect affordance unlocks globally ---
        new_masks = {a: self._action_mask(a) for a in self.agents}
        global_unlock = False
        if len(self.tagged_pois) > old_tagged_pois_len:
            global_unlock = True
        if self.terminal_disabled and not old_terminal_disabled:
            global_unlock = True
        if self.loot_acquired and not old_loot_acquired:
            global_unlock = True
        if self.extraction_triggered and not old_extraction_triggered:
            global_unlock = True
        if self.hack_progress > old_hack_progress:
            global_unlock = True

        car_unlocked_agents = set()
        for a in interact_actors:
            if (
                global_unlock
                or actions.get(a) == BREACH
                and self._evaluate_objective_affordance(
                    old_grid, self.agent_positions[a]
                )
            ):
                car_unlocked_agents.add(a)

        infos = {
            a: {
                "alarm": self.alarm,
                "win": bool(win),
                "lose": bool(lose),
                "car_unlocked": (a in car_unlocked_agents),
                "pos": self.agent_positions[a],
            }
            for a in self.agents
        }

        observations = self._get_all_obs(precomputed_masks=new_masks)

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
                        pg.draw.line(
                            screen,
                            (60, 40, 10),
                            (x + 2, y + 2),
                            (x + TILE_SIZE - 2, y + TILE_SIZE - 2),
                            3,
                        )
                    elif tile == CAMERA:
                        pg.draw.circle(
                            screen,
                            (0, 0, 0),
                            (x + TILE_SIZE // 2, y + TILE_SIZE // 2),
                            3,
                        )

        # fog of war
        fog_surface = pg.Surface(
            (self.map_w * TILE_SIZE, self.map_h * TILE_SIZE), pg.SRCALPHA
        )
        for row in range(self.map_h):
            for col in range(self.map_w):
                if not self.explored_map[row][col]:
                    pg.draw.rect(
                        fog_surface,
                        COLORS["EXPLORED"],
                        (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                    )
        screen.blit(fog_surface, (0, 0))

        # agents
        for agent in self.agents:
            pos = self.agent_positions[agent]
            circle = (
                pos[1] * TILE_SIZE + TILE_SIZE // 2,
                pos[0] * TILE_SIZE + TILE_SIZE // 2,
            )
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
            circle = (
                gpos[1] * TILE_SIZE + TILE_SIZE // 2,
                gpos[0] * TILE_SIZE + TILE_SIZE // 2,
            )
            pg.draw.circle(screen, COLORS[GUARD], circle, TILE_SIZE // 2)

        # alarm meter
        bar_w = self.map_w * TILE_SIZE
        pg.draw.rect(screen, (40, 40, 40), (0, self.map_h * TILE_SIZE, bar_w, 10))
        pg.draw.rect(
            screen,
            (255, 60, 60),
            (
                0,
                self.map_h * TILE_SIZE,
                int(bar_w * min(self.alarm, ALARM_MAX) / ALARM_MAX),
                10,
            ),
        )
        if font is not None:
            status = f"step {self.current_step} | alarm {self.alarm:.0f}/100"
            if self.terminal_disabled:
                status += " | terminal: DISABLED"
            if self.loot_acquired:
                status += " | loot: SECURED"
            if self.extraction_triggered:
                status += f" | extract: {self.extraction_countdown}"
            if self._pending_events:
                status += f" | delayed_events: {len(self._pending_events)}"
            screen.blit(
                font.render(status, True, (255, 255, 255)),
                (4, self.map_h * TILE_SIZE + 14),
            )

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
        # REV-6 (REVISION_PLAN.md §5): while loot_acquired the extractor may
        # only move 1 tile every 2 turns (escort dynamic); mirror the slow-turn
        # gate in _action_mask so the policy sees the constraint.
        if agent == "extractor" and self.loot_acquired and self._burden_start_step >= 0:
            turns_since = self.current_step - self._burden_start_step
            if (
                turns_since > 0
                and self.config["extractor_burden"] > 1
                and turns_since % self.config["extractor_burden"] == 0
            ):
                return
        row, col = self.agent_positions[agent]
        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc
        if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
            return
        tile = self.grid[nr, nc]
        if tile in (WALL, DOOR):
            return
        self.agent_positions[agent] = (nr, nc)
        self._reveal_around(agent, self.config["agent_vision"])
        if agent == "scout":
            self._refresh_scout_fov()

    def _refresh_scout_fov(self):
        sr, sc = self.agent_positions["scout"]
        calculate_fov(
            self.grid,
            self.explored_map,
            sr,
            sc,
            self.config["scout_vision"],
            self.map_h,
            self.map_w,
            WALL,
        )

    def _reveal_around(self, agent, radius):
        r, c = self.agent_positions[agent]
        calculate_fov(
            self.grid, self.explored_map, r, c, radius, self.map_h, self.map_w, WALL
        )

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
        """Scout broadcasts intel on a nearby point of interest or performs a Decoy Noise Ping."""
        pois = (
            [self.terminal_pos, self.loot_pos, self.extract_pos]
            + self.camera_positions
            + self.door_positions
        )
        for p in pois:
            if p not in self.tagged_pois and manhattan(pos, p) <= 1:
                self.tagged_pois.add(p)
                rewards["scout"] += self.config["reward_tag"]
                return

        # Side-task: Decoy Noise Ping
        if self.config.get("enable_side_tasks", False):
            ping_triggered = False
            for gi, gpos in enumerate(self.guard_positions):
                if self.neutralized[gi] == 0 and manhattan(pos, gpos) <= 6:
                    if self.guard_states[gi] != "search":
                        ping_triggered = True
                    self.guard_states[gi] = "search"
                    self._guard_search_target[gi] = pos
                    self._guard_search_turns[gi] = self.config["search_turns"]
            if ping_triggered:
                rewards["scout"] += 0.2

    def _hacker_hack(self, pos, rewards):
        """Multi-turn terminal hack; interruption resets progress."""
        if (
            not self.terminal_disabled
            and self.terminal_pos in self.tagged_pois
            and manhattan(pos, self.terminal_pos) <= 1
        ):
            self.hack_progress += 1
            self._add_alarm(self.config["alarm_hack_turn"], rewards)
            if self.hack_progress >= self.config["hack_turns"]:
                self.terminal_disabled = True
                self.hack_progress = 0
                rewards["hacker"] += 1.0
            else:
                rewards["hacker"] += 0.5
            return
        # fallback: force-bypass an adjacent locked door
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if (
                0 <= nr < self.map_h
                and 0 <= nc < self.map_w
                and self.grid[nr, nc] == DOOR
            ):
                self.grid[nr, nc] = EMPTY
                self._add_alarm(self.config["alarm_bypass"], rewards)
                rewards["hacker"] += 0.2
                return

    def _muscle_neutralize(self, pos, rewards):
        """REV-5/REV-9: neutralize a nearby guard or breach an adjacent wall."""
        # --- neutralize an adjacent guard (within 2 tiles) ---
        best, best_d = None, 1e9
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            d = manhattan(pos, gpos)
            if d < best_d:
                best_d, best = d, gi
        if best is not None and best_d <= 2:
            self.neutralized[best] = self.config["neutral_turns"]
            self._neutralized_pos[best] = self.guard_positions[best]
            self._pending_events.append(
                (
                    self.current_step + self.config["alarm_neut_delay"],
                    self.config["alarm_neutralize"],
                    ("neutralize", best),
                )
            )
            if best not in self._rewarded_neutralized_guards:
                self._rewarded_neutralized_guards.add(best)
                rewards["muscle"] += self.config["reward_task"]

    def _muscle_breach(self, pos, rewards):
        """REV-5: breach an adjacent wall."""
        grid_pre = self.grid.copy()
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.map_h and 0 <= nc < self.map_w):
                continue
            if self.grid[nr, nc] == WALL:
                self.grid[nr, nc] = EMPTY
                self._add_alarm(
                    self.config["alarm_breach"], rewards, acting_agent="muscle"
                )
                if (nr, nc) not in self._rewarded_breached_walls:
                    self._rewarded_breached_walls.add((nr, nc))
                    rewards["muscle"] += self.config["reward_task"]
                if self.config["breach_search_trigger"]:
                    self._trigger_breach_search(nr, nc)
                self._last_car_unlocked = self._evaluate_objective_affordance(
                    grid_pre, pos
                )
                return

    def _trigger_breach_search(self, br, bc):
        """REV-5: guards near a fresh breach switch to Search at that tile."""
        radius = self.config["breach_radius"]
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            if manhattan((br, bc), gpos) <= radius:
                self.guard_states[gi] = "search"
                self._guard_search_target[gi] = (br, bc)
                self._guard_search_turns[gi] = self.config["search_turns"]

    def _process_pending_events(self, rewards=None):
        """REV-9: fire delayed alarm events whose trigger step has arrived."""
        due = [e for e in self._pending_events if e[0] <= self.current_step]
        if not due:
            return
        self._pending_events = [
            e for e in self._pending_events if e[0] > self.current_step
        ]
        for _, amount, source in due:
            self._add_alarm(amount, rewards)
            if source is not None and source[0] == "neutralize":
                gi = source[1]
                last = self._neutralized_pos.get(gi)
                if last is not None:
                    for j, jpos in enumerate(self.guard_positions):
                        if (
                            j != gi
                            and self.neutralized[j] == 0
                            and manhattan(jpos, last) <= self.config["breach_radius"]
                        ):
                            self.guard_states[j] = "search"
                            self._guard_search_target[j] = last
                            self._guard_search_turns[j] = self.config["search_turns"]

    def _extractor_act(self, pos, rewards):
        """Secure loot (needs disabled terminal), call extraction, or pre-calibrate beacon."""
        if (
            not self.loot_acquired
            and self.terminal_disabled
            and manhattan(pos, self.loot_pos) <= 1
        ):
            self.loot_acquired = True
            self._burden_start_step = self.current_step
            rewards["extractor"] += self.config["reward_task"]
            return
        if self.loot_acquired and not self.extraction_triggered:
            self.extraction_triggered = True
            if self.beacon_calibrated:
                self.extraction_countdown = min(3, self.config["extraction_countdown"])
            else:
                self.extraction_countdown = self.config["extraction_countdown"]
            rewards["extractor"] += 0.5
            return
        # Side-task: Extraction Beacon Pre-Calibration
        if (
            self.config.get("enable_side_tasks", False)
            and not self.beacon_calibrated
            and manhattan(pos, self.extract_pos) <= 1
        ):
            self.beacon_calibrated = True
            rewards["extractor"] += 0.2

    # ------------------------------------------------------------------
    # Internals: guards and alarm
    # ------------------------------------------------------------------
    def _valid_moves(self, r, c):
        moves = get_valid_moves(self.grid, r, c, WALL, DOOR)
        return [(int(m[0]), int(m[1])) for m in moves]

    def _pick_search_tile(self, center):
        """REV-8: a random walkable tile within search_radius of `center`."""
        r, c = pick_search_tile(
            self.grid,
            center[0],
            center[1],
            self.config["search_radius"],
            WALL,
            DOOR,
            float(self.rng.random()),
        )
        return (int(r), int(c))

    def _move_guards(self):
        """REV-8 (REVISION_PLAN.md §7): per-guard Patrol/Search/Converge FSM.

        * Patrol     : random walk until an agent crosses the guard's
                       directional line of sight (vision.line_is_clear), a
                       breach occurs within range (REV-5), or command reports
                       a missing guard (REV-9) - both flip the guard to Search.
        * Search     : BFS towards the last-known position for up to
                       search_turns; sweeps a random nearby tile once the
                       target is reached, then returns to Patrol.
        * Converge   : global alarm >= converge_alarm puts every active guard
                       into a BFS hunt for the nearest agent.
        """
        converge = self.alarm >= self.config["converge_alarm"]
        # In converge mode all guards pursue the nearest agent.  One
        # multi-source BFS produces the exact shortest-path distance for every
        # guard cell to replace N identical full-grid searches per frame.
        converge_distance = None
        if converge:
            targets = np.asarray(list(self.agent_positions.values()), dtype=np.int32)
            converge_distance = distance_to_nearest_target(
                self.grid, targets, WALL, DOOR, self._distance_map, self._bfs_queue
            )
        new_positions = []
        for gi, (gr, gc) in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                self.neutralized[gi] -= 1
                new_positions.append((gr, gc))
                continue

            # ---- sense: directional line-of-sight ----------------
            spotted = self._guard_spot(gr, gc)
            if spotted is not None:
                self.guard_states[gi] = "search"
                self._guard_search_target[gi] = spotted
                self._guard_search_turns[gi] = self.config["search_turns"]

            # ---- decide ------------------------------------------
            if converge:
                self.guard_states[gi] = "converge"
            elif (
                self.guard_states[gi] == "search" and self._guard_search_turns[gi] <= 0
            ):
                self.guard_states[gi] = "patrol"
                self._guard_search_target[gi] = None

            valid = self._valid_moves(gr, gc)
            if not valid:
                new_positions.append((gr, gc))
                continue

            state = self.guard_states[gi]
            if state == "converge":
                nr, nc = next_step_from_distance(converge_distance, gr, gc)
                nxt = None if nr < 0 else (int(nr), int(nc))
                best_target = None
                best_dist = 1e9
                for p in self.agent_positions.values():
                    d = abs(p[0] - gr) + abs(p[1] - gc)
                    if d < best_dist:
                        best_dist = d
                        best_target = p
                target = best_target
                if nxt is not None:
                    new_positions.append(nxt)
                else:
                    best_v = None
                    best_d = 1e9
                    for v in valid:
                        d = abs(v[0] - target[0]) + abs(v[1] - target[1])
                        if d < best_d:
                            best_d = d
                            best_v = v
                    new_positions.append(best_v)
            elif state == "search" and self._guard_search_target[gi] is not None:
                target = self._guard_search_target[gi]
                if (gr, gc) == target:
                    # reached the last-known spot: sweep the neighbourhood
                    target = self._pick_search_tile(target)
                    self._guard_search_target[gi] = target
                nxt = self._bfs_next(gr, gc, target)
                if nxt is None:
                    best_v = None
                    best_d = 1e9
                    for v in valid:
                        d = abs(v[0] - target[0]) + abs(v[1] - target[1])
                        if d < best_d:
                            best_d = d
                            best_v = v
                    new_positions.append(best_v)
                else:
                    new_positions.append(nxt)
                self._guard_search_turns[gi] -= 1
            else:
                # patrol: random walk
                new_positions.append(valid[int(self.rng.integers(len(valid)))])
        self.guard_positions = new_positions

    def _guard_spot(self, gr, gc):
        """REV-8: return the position of the first agent visible to this guard,
        or None.  Uses vision.line_is_clear over a bounded manhattan range so
        guards cannot see through walls or locked doors."""
        los_range = self.config["guard_los_range"]
        for _agent, apos in self.agent_positions.items():
            if manhattan((gr, gc), apos) > los_range:
                continue
            if line_is_clear(self.grid, gr, gc, apos[0], apos[1], WALL, DOOR):
                return apos
        return None

    def _bfs_next(self, gr, gc, target):
        """REV-8: BFS from (gr, gc) toward `target`, returning the first step.

        BFS is used in place of greedy Manhattan because wall clutter makes the
        straight-line step a dead end; a full BFS on a 50x50 map is ~2500 cells
        and trivially cheap at 6 guards/frame.
        """
        nr, nc = bfs_next_step(
            self.grid,
            gr,
            gc,
            target[0],
            target[1],
            WALL,
            DOOR,
            self._bfs_queue,
            self._bfs_previous,
            self._bfs_reset,
        )
        return None if nr < 0 else (int(nr), int(nc))

    def _check_caught(self):
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] > 0:
                continue
            for apos in self.agent_positions.values():
                if manhattan(gpos, apos) <= self.config["catch_distance"]:
                    return True
        return False

    def _add_alarm(self, amount, rewards=None, acting_agent=None):
        prev_alarm = self.alarm
        self.alarm = min(self.alarm + amount, self.config["alarm_max"])
        delta = self.alarm - prev_alarm
        if delta > 0 and rewards is not None:
            penalty = 0.01 * delta
            for a in self.agents:
                rewards[a] -= penalty
            if acting_agent is not None and acting_agent in rewards:
                rewards[acting_agent] -= 0.10 * delta

    def _evaluate_objective_affordance(self, grid_pre, agent_pos, threshold=4):
        objectives = [
            p
            for p in [self.terminal_pos, self.loot_pos, self.extract_pos]
            if p is not None
        ]
        for obj_pos in objectives:
            d_pre = bfs_shortest_path_distance(
                grid_pre,
                agent_pos[0],
                agent_pos[1],
                obj_pos[0],
                obj_pos[1],
                WALL,
                DOOR,
                self._bfs_queue,
                self._distance_map,
                self._bfs_reset,
            )
            d_post = bfs_shortest_path_distance(
                self.grid,
                agent_pos[0],
                agent_pos[1],
                obj_pos[0],
                obj_pos[1],
                WALL,
                DOOR,
                self._bfs_queue,
                self._distance_map,
                self._bfs_reset,
            )
            if (d_pre >= 999999 and d_post < 999999) or (d_pre - d_post >= threshold):
                return True
        return False

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
        mask[BREACH] = 0
        pos = self.agent_positions[agent]

        # block movement into walls / locked doors / out of bounds
        for a in range(4):
            dr, dc = ACTION_DELTAS[a]
            nr, nc = pos[0] + dr, pos[1] + dc
            if (
                not (0 <= nr < self.map_h and 0 <= nc < self.map_w)
                or self.grid[nr, nc] == WALL
                or self.grid[nr, nc] == DOOR
            ):
                mask[a] = 0

        # causal gate: only allow INTERACT when it is actually possible
        enable_st = self.config.get("enable_side_tasks", False)
        if agent == "scout":
            pois = (
                [self.terminal_pos, self.loot_pos, self.extract_pos]
                + self.camera_positions
                + self.door_positions
            )
            near_poi = enable_st
            if not near_poi:
                for p in pois:
                    if manhattan(pos, p) <= 1:
                        near_poi = True
                        break
            if near_poi:
                mask[INTERACT] = 1
        elif agent == "hacker":
            # causal gate: the terminal must have been tagged by the scout
            # before the hacker can act on it (RG-Dec-POMDP chain step 1)
            near_terminal = (
                not self.terminal_disabled
                and self.terminal_pos in self.tagged_pois
                and manhattan(pos, self.terminal_pos) <= 1
            )
            near_door = False
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = pos[0] + dr, pos[1] + dc
                if (
                    0 <= nr < self.map_h
                    and 0 <= nc < self.map_w
                    and self.grid[nr, nc] == DOOR
                ):
                    near_door = True
                    break
            if near_terminal or near_door:
                mask[INTERACT] = 1
        elif agent == "muscle":
            near_guard = False
            for gi, gpos in enumerate(self.guard_positions):
                if self.neutralized[gi] == 0 and manhattan(pos, gpos) <= 2:
                    near_guard = True
                    break
            # REV-5: wall breach is valid when adjacent to a wall AND guards exist or side tasks enabled
            near_wall = False
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = pos[0] + dr, pos[1] + dc
                if (
                    0 <= nr < self.map_h
                    and 0 <= nc < self.map_w
                    and self.grid[nr, nc] == WALL
                ):
                    near_wall = True
                    break
            if near_guard:
                mask[INTERACT] = 1
            if near_wall:
                mask[BREACH] = 1
        elif agent == "extractor":
            near_loot = (
                not self.loot_acquired
                and self.terminal_disabled
                and manhattan(pos, self.loot_pos) <= 1
            )
            can_call = self.loot_acquired and not self.extraction_triggered
            near_extract_pre = (
                enable_st
                and not self.beacon_calibrated
                and manhattan(pos, self.extract_pos) <= 1
            )
            if near_loot or can_call or near_extract_pre:
                mask[INTERACT] = 1
            # REV-6: on a slow turn the extractor cannot move at all.  The mask
            # is evaluated before step() increments current_step, so look one
            # turn ahead to match the movement gate in _move_agent.
            if self.loot_acquired and self._burden_start_step >= 0:
                turns_since = (self.current_step + 1) - self._burden_start_step
                if (
                    turns_since > 0
                    and self.config["extractor_burden"] > 1
                    and turns_since % self.config["extractor_burden"] == 0
                ):
                    mask[:4] = 0
        return mask

    def _get_all_obs(self, precomputed_masks=None):
        """Build all local observations from one dynamic-grid composition.

        The static map and explored mask are shared by every agent.  Drawing
        dynamic entities and padding those arrays once avoids four full-grid
        copies and eight ``np.pad`` calls per environment step.
        """
        composite = self.grid.copy()
        for gi, gpos in enumerate(self.guard_positions):
            if self.neutralized[gi] == 0:
                composite[gpos[0], gpos[1]] = GUARD
        for opos in self.agent_positions.values():
            composite[opos[0], opos[1]] = ALLY
        pad = self._pad
        self._pad_grid_cache[pad:-pad, pad:-pad] = composite
        self._pad_explored_cache[pad:-pad, pad:-pad] = self.explored_map
        pad_grid = self._pad_grid_cache
        pad_explored = self._pad_explored_cache
        observations = {}
        for agent in self.agents:
            pos = self.agent_positions[agent]
            vpos = (pos[0] + pad, pos[1] + pad)
            obs = pad_grid[
                vpos[0] - pad : vpos[0] + pad + 1,
                vpos[1] - pad : vpos[1] + pad + 1,
            ].copy()
            # The focal agent is intentionally transparent, matching the old
            # per-agent composite without rebuilding an entire map per agent.
            obs[pad, pad] = self.grid[pos[0], pos[1]]
            obs_explored = pad_explored[
                vpos[0] - pad : vpos[0] + pad + 1,
                vpos[1] - pad : vpos[1] + pad + 1,
            ]
            mask = (
                precomputed_masks[agent]
                if precomputed_masks is not None and agent in precomputed_masks
                else self._action_mask(agent)
            )
            # Project directional beacon for active tagged POIs if outside 7x7 vision window
            for poi in self.tagged_pois:
                if poi is None:
                    continue
                is_active = (
                    (poi == self.terminal_pos and not self.terminal_disabled)
                    or (
                        poi == self.loot_pos
                        and self.terminal_disabled
                        and not self.loot_acquired
                    )
                    or (
                        poi == self.extract_pos
                        and (self.loot_acquired or self.extraction_triggered)
                    )
                )
                if is_active:
                    dr = poi[0] - pos[0]
                    dc = poi[1] - pos[1]
                    if abs(dr) > pad or abs(dc) > pad:
                        edge_r = pad + int(np.clip(dr, -pad, pad))
                        edge_c = pad + int(np.clip(dc, -pad, pad))
                        if (edge_r, edge_c) != (pad, pad):
                            obs[edge_r, edge_c] = WAYPOINT

            observations[agent] = {
                "observation": np.where(obs_explored, obs, FOG),
                "action_mask": mask,
                "role_id": ROLE_ONEHOT_ARRAYS[agent],
            }
        return observations

    def state(self):
        """Rich global state for centralized critics (MAPPO) / QMIX mixing.

        Concatenates scalar phase indicators, the full grid, all agent
        positions, all guard positions, and neutralization timers.
        """
        self._state_buffer[0] = self.current_step
        self._state_buffer[1] = self.alarm
        self._state_buffer[2] = int(self.terminal_disabled)
        self._state_buffer[3] = int(self.loot_acquired)
        self._state_buffer[4] = int(self.extraction_triggered)
        self._state_buffer[5] = self.extraction_countdown

        idx = 6
        grid_len = self.map_h * self.map_w
        self._state_buffer[idx : idx + grid_len] = self.grid.ravel()
        idx += grid_len

        for a in self.possible_agents:
            pos = self.agent_positions[a]
            self._state_buffer[idx] = pos[0]
            self._state_buffer[idx + 1] = pos[1]
            idx += 2

        for gpos in self.guard_positions:
            self._state_buffer[idx] = gpos[0]
            self._state_buffer[idx + 1] = gpos[1]
            idx += 2

        num_guards = len(self.neutralized)
        if num_guards > 0:
            self._state_buffer[idx : idx + num_guards] = self.neutralized

        return self._state_buffer

    # ------------------------------------------------------------------
    # Renderer helpers
    # ------------------------------------------------------------------
    def _render_ansi(self):
        map_dict = {
            EMPTY: " ",
            WALL: "#",
            TERMINAL: "T",
            LOOT: "$",
            EXTRACT: "@",
            GUARD: "G",
            CAMERA: "C",
            DOOR: "D",
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
        status = (
            f"step={self.current_step} alarm={self.alarm:.1f}/100 "
            f"terminal={'off' if self.terminal_disabled else 'on'} "
            f"loot={'yes' if self.loot_acquired else 'no'} "
            f"extract={'calling' if self.extraction_triggered else 'idle'}"
        )
        # REV-9: pending alarm event count
        if self._pending_events:
            status += f" pending_events={len(self._pending_events)}"
        return "\n".join(lines) + "\n" + status
