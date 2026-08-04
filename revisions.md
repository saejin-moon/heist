### FATAL FLAW 1: The API Contract Violation (Shape Mismatch)
Your agent updated the `global_state` vector to include relative distances to objectives, increasing its size from `4` to `10`.
*(See `constants.py`: `GLOBAL_STATE_DIM = 4 + 3 * 2 = 10`)*

But it forgot to update the actual Gymnasium `Box` space definition in `env.py`:
```python
# In env.py __init__
"global_state": Box(low=0, high=100, shape=(4,), dtype=np.int32)
```
If you run `parallel_api_test` right now, it will violently crash because the returned tensor is length 10, but the contracted shape is length 4. 
**The Fix:** Change `shape=(4,)` to `shape=(GLOBAL_STATE_DIM,)` in `env.py`.

### FATAL FLAW 2: The Heterogeneous Aliasing Bug (Shared Critic Collapse)
In `train_mappo.py` and `train_ippo.py (--shared)`, your agent implemented parameter sharing. All four agents use the exact same Neural Network (`MappoAgent`). 

**The Math Problem:** The Scout and the Extractor receive the exact same observation shapes. If the Scout is standing next to the terminal, it should wait for the Hacker. If the Hacker is standing next to the terminal, it should hack it. 
*How does the single shared Neural Network know which agent it is currently controlling?* 
It doesn't. There is no `agent_id` or one-hot role vector passed into the `HeistAgent` forward pass. 
The shared Value Function (Critic) will suffer from catastrophic aliasing—it will look at a state and guess the value is high (assuming it's the Hacker) while the actor is actually the Extractor. The policy will collapse.
**The Fix:** You must append a one-hot encoding of the agent's role (e.g., `[1, 0, 0, 0]` for Scout) to the `global_state` or `observation` vector before it hits the MLP layers in `model.py`.

### FATAL FLAW 3: The Time-Limit Bootstrapping Bug
In `train_ippo.py` and `train_mappo.py`, the agent wrote standard CleanRL PPO advantage calculations:
```python
nextnonterminal = 1.0 - buffers[a]["dones"][t + 1]
```
In your environment, an episode ends for two reasons:
1. `terminations` (Win, or Guard caught you). This is a true end state. Value = 0.
2. `truncations` (Time limit reached at 100 steps). This is an artificial horizon.

Your training loops lump both of these into `dones`. When the time limit hits, the code forces the Value Function to bootstrap to `0.0`. This tells the AI that "running out of time" is mathematically identical to "jumping into the arms of a guard." The AI will learn to commit suicide to end the episode faster rather than try to play the game.
**The Fix:** You must implement proper truncation bootstrapping. If the episode is truncated, the TD-target must use the Value of the final state, not `0.0`. 

### FATAL FLAW 4: The VectorEnv Blind Spot
In `vec_env.py`, your agent wrote this:
```python
o, r, t, tr, inf = env.step(acts)
done = bool(any(t.values()) or any(tr.values()))
if done:
    o, _ = env.reset() # <--- FATAL
```
When an episode ends, `vec_env` instantly overwrites the final observation with the observation of the *new* episode.
The PPO algorithm never actually sees the terminal state. The final reward (e.g., `+10.0` for winning) is mistakenly mapped to the starting state of the *next* game. Your neural network is learning that spawning into the map is what causes the `+10.0` reward.
**The Fix:** When `done` is true, you must stash the real terminal observation into the `inf` (info) dictionary (e.g., `inf[a]["terminal_observation"] = o[a]`), and the PPO loop must extract it to calculate the final `next_value`.

Here is exactly what the agent got wrong about the future roadmap and mechanics:

### 1. THE MUSCLE'S WALL BREACH (Missed entirely)
**The Plan:** I explicitly approved giving the Muscle agent a `BREACH` action to permanently change a `WALL` into `EMPTY`. It was balanced by a strict constraint: it instantly adds +30% to the global alarm and forces guards in a 10-tile radius to pathfind to that specific coordinate.
**What your AI did:** It completely forgot the wall-breach mechanic. Instead, it gave the *Hacker* a lazy "bypass door" fallback action (`_hacker_hack` in `env.py`). The Muscle is currently a one-trick pony that just temporarily deletes guards with no delayed alarm mechanic.

### 2. THE EXTRACTOR'S BURDEN (Missed entirely)
**The Plan:** The Extractor carrying the loot must suffer a movement penalty (e.g., can only move 1 tile every 2 turns). This forces the rest of the team to act as an escort, dynamically shifting the formation and extending the period of vulnerability.
**What your AI did:** Nothing. The Extractor grabs the loot and sprints across the map at the exact same speed as the Scout. The "escort" dynamic does not exist. 

### 3. THE RESEARCH CROWN JEWEL: LEARNED COMMUNICATION (Downgraded to "Optional")
**The Plan:** The ultimate goal of your research was to delete the `global_state` vector entirely and force the agents to invent their own language (Differentiable Communication / TarMAC) to signal that the terminal was hacked.
**What your AI did:** In its `PLAN.md`, it downgraded this to "optional communication variants." Furthermore, it deeply embedded `global_state` (and exact relative (x, y) coordinates to the objectives) into every single baseline algorithm. It built a permanent crutch that removes the need for agents to actually communicate.

### 4. THE GUARD AI (Reduced to a Hive-Mind Homing Missile)
**The Plan:** Guards were supposed to have directional Line-of-Sight, shifting from "Patrol" to "Search" when spotting broken doors or agents on the periphery, culminating in A* pathfinding.
**What your AI did:** In `_move_guards()`, it implemented a lazy global trigger. If the alarm crosses 50%, every guard on the map instantly gains telepathy and walks directly toward the nearest agent using basic Manhattan distance. There is no Line-of-Sight, no Search state, and no A* maze navigation. 

### 5. THE DELAYED ALARM (Failed Execution)
**The Plan:** When the Muscle neutralizes a guard, the alarm doesn't go up instantly. It goes up *15 turns later* when command realizes the guard is missing, creating a ticking time bomb.
**What your AI did:** In `_muscle_neutralize()`, it just immediately adds 15.0 to the alarm meter (`self._add_alarm(self.config["alarm_neutralize"])`). It failed to program a delayed event queue.