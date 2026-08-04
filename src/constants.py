"""
Global constants for the HEIST environment.

HEIST: Hierarchical Environment for Interdependent Sequential Tasks.
Four specialized agents (Scout, Hacker, Muscle, Extractor) must coordinate
through a sequential causal dependency chain against a rule-based security
system (guards + security cameras + incremental alarm meter).

Tile semantics are central to the observation encoding: an agent's 5x5 local
view is FOG-masked (value -1) until the Scout (or the agent itself) has
revealed the tile, which is what makes the environment partially observable
in a way that is causally gated by upstream behavior.
"""

# ---------------------------------------------------------------------------
# Tile types (values observed by agents / used by the renderer)
# ---------------------------------------------------------------------------
FOG = -1        # tile hidden behind fog of war
EMPTY = 0       # walkable floor
WALL = 1        # solid wall (blocks movement and line of sight)
TERMINAL = 2    # security terminal - hacker disables cameras/vault here
LOOT = 3        # the heist loot - extractor must secure it
EXTRACT = 4     # extraction point - all agents must end here with loot
GUARD = 5       # rule-based adversary (dynamic entity)
ALLY = 6        # other agent (dynamic entity)
CAMERA = 7      # security camera (line-of-sight alarm source)
DOOR = 8        # locked door (blocks movement; hacker can bypass)

# ---------------------------------------------------------------------------
# Actions
#
# All agents share the same action space for architectural simplicity
# (this keeps the 6-element action_mask contract from the design doc):
#   0..3 : move up / down / left / right
#   4    : wait
#   5    : role-specific special action (see ROLE_ACTIONS below)
# ---------------------------------------------------------------------------
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4
INTERACT = 5

ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    WAIT: (0, 0),
}

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
AGENTS = ["scout", "hacker", "muscle", "extractor"]
AGENT_CHAR = {"scout": "S", "hacker": "H", "muscle": "M", "extractor": "E"}

# Role identity: a one-hot vector emitted as its own observation key so that
# shared policies (IPPO --shared, MAPPO, and the learned-communication agents)
# can distinguish which role the network is currently controlling.  This
# survives the REV-7 removal of global_state.
N_AGENTS = len(AGENTS)
ROLE_IDS = {a: i for i, a in enumerate(AGENTS)}
ROLE_ONEHOT = {
    a: [1 if i == j else 0 for j in range(N_AGENTS)]
    for i, a in enumerate(AGENTS)
}

# What each agent's INTERACT (action 5) does.  This is the per-role
# specialization that replaces the monolithic "interact" of the prototype.
ROLE_ACTIONS = {
    "scout": "tag_tile",          # reveal + broadcast a point of interest
    "hacker": "hack_terminal",    # multi-turn terminal hack / bypass doors
    "muscle": "neutralize",       # temporarily remove a nearby guard
    "extractor": "call_extract",  # trigger the final extraction countdown
}

# ---------------------------------------------------------------------------
# Observation / layout dimensions
# ---------------------------------------------------------------------------
MAP_SIZE = (50, 50)              # default; overridable via env config
OBSERVATION_SIZE = (5, 5)        # agent local view window
ACTION_SPACE_SIZE = 6            # |A| for every agent
TILE_SIZE = 20                   # renderer pixels per tile
SCOUT_VISION_RADIUS = 8          # scout reveals this far in every direction
AGENT_VISION_RADIUS = 2          # every agent sees/reveals this local radius
GUARD_COUNT = 6
CAMERA_COUNT = 3
DOOR_COUNT = 4

# ---------------------------------------------------------------------------
# Causal-chain mechanics
# ---------------------------------------------------------------------------
HACK_TURNS = 3                   # multi-turn hack before terminal is disabled
EXTRACTION_COUNTDOWN = 45        # turns to gather at extract after loot secured

# ---------------------------------------------------------------------------
# Reward structure (design doc section 5)
# ---------------------------------------------------------------------------
REWARD_WIN = 10.0                # shared terminal reward: successful extraction
REWARD_LOSE = -10.0              # shared terminal reward: caught / alarm / timeout
REWARD_TASK = 2.0                # key task completions (hack, loot, extract call)
REWARD_TAG = 0.5                 # scout revealing a point of interest
REWARD_TIME_BLEED = -0.01        # baseline penalty per step
CONVERGE_BONUS = 0.2             # per-step proximity bonus to the extract tile
                                 # once extraction is called (shapes the final phase)
CONVERGE_RADIUS = 4              # manhattan distance that counts as "converging"
WIN_CONVERGE_RADIUS = 2          # all agents within this radius of extract to win
                                 # (0 = strict stacking on the tile)

# ---------------------------------------------------------------------------
# Alarm system
#
# A shared global alarm meter in [0, 100].  Events raise it incrementally;
# at 100 the guards converge and the episode terminates with REWARD_LOSE.
# ---------------------------------------------------------------------------
ALARM_MAX = 100.0
ALARM_CAMERA = 0.35              # per visible agent per camera per step
ALARM_HACK_TURN = 2.0            # per turn of hacking a terminal
ALARM_BYPASS = 6.0               # door bypassed by force
ALARM_NEUTRALIZE = 15.0          # guard knocked out (delay until noticed)
ALARM_GUARD_SPOT = 25.0          # guard gets within catch distance
ALARM_EXTRACTION_TIMEOUT = 25.0  # extraction countdown expired (fires once)
CAMERA_RANGE = 12                # max line-of-sight distance for cameras

# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------
# global_state layout: [step, alarm, terminal_disabled, loot_acquired] then
# per-objective relative displacement (dx, dy) for the terminal, loot, and
# extract tiles.  Agent-relative coordinates keep the state Markovian and
# give every agent a navigation signal toward the objectives.
GLOBAL_STATE_DIM = 4 + 3 * 2

# ---------------------------------------------------------------------------
# Guard behavior
# ---------------------------------------------------------------------------
CATCH_DISTANCE = 1               # Manhattan distance that triggers a catch
CONVERGE_ALARM = 50.0            # alarm level above which guards hunt agents
NEUTRALIZE_TURNS = 8             # turns a guard stays neutralized

# ---------------------------------------------------------------------------
# M1 mechanics (REV-5/6/8/9): wall breach, extractor burden, guard AI,
# delayed alarm
# ---------------------------------------------------------------------------
ALARM_BREACH = 30.0              # instant global-alarm cost of a wall breach
BREACH_RADIUS = 10               # guards within this range repath to the breach
EXTRACTOR_BURDEN_TURNS = 2       # loot-carrying extractor moves 1 tile per N turns
ALARM_NEUTRALIZE_DELAY = 15      # turns until command notices a missing guard
GUARD_LOS_RANGE = 8              # directional line-of-sight reach for guards
SEARCH_RADIUS = 5                # tiles around a last-known position to sweep
SEARCH_TURNS = 6                 # turns a guard stays in Search before Patrol
BREACH_SEARCH_TRIGGER = True     # a fresh breach puts nearby guards into Search

# ---------------------------------------------------------------------------
# M2 communication (REV-7): learned message channel
# ---------------------------------------------------------------------------
COMM_MESSAGE_DIM = 16            # TarMAC-style message embedding size
COMM_HIDDEN_DIM = 64             # communication network hidden width

# ---------------------------------------------------------------------------
# Renderer palette
# ---------------------------------------------------------------------------
COLORS = {
    WALL: (20, 20, 30),
    EMPTY: (245, 245, 245),
    TERMINAL: (0, 100, 255),
    LOOT: (255, 215, 0),
    EXTRACT: (0, 200, 90),
    GUARD: (255, 60, 60),
    CAMERA: (120, 40, 180),
    DOOR: (160, 110, 40),
    "AGENT": {
        "scout": (0, 255, 255),
        "hacker": (150, 60, 220),
        "muscle": (180, 60, 60),
        "extractor": (255, 150, 40),
    },
    "EXPLORED": (0, 0, 0, 90),   # fog overlay alpha
}
