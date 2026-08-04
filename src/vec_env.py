"""
Vectorized environment wrapper for HEIST.

Wraps `num_envs` independent HeistEnv instances so the PPO training loops
can collect rollouts in a simple numpy array layout (one buffer per agent),
mirroring the CleanRL vector-env convention.

Each env is stepped with a dict of per-agent actions and, when an episode
terminates or truncates, that env is immediately reset so the rollout buffer
can be kept dense (the `done` flags tell the learner where episode boundaries
are).
"""

import numpy as np

from env import HeistEnv, AGENTS


class VectorEnv:
    def __init__(self, num_envs, config=None, base_seed=0):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.envs = [HeistEnv(config) for _ in range(num_envs)]
        self.state_dim = self.envs[0].state().shape[0]
        self.obs = None
        self.state = None

    # ------------------------------------------------------------------
    def _pack(self, obs_list):
        packed = {}
        for a in AGENTS:
            packed[a] = {
                "observation": np.stack([o[a]["observation"] for o in obs_list]),
                "action_mask": np.stack([o[a]["action_mask"] for o in obs_list]),
                "global_state": np.stack([o[a]["global_state"] for o in obs_list]),
            }
        return packed

    def reset(self, seed=None):
        obs_list, states = [], []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=(seed if seed is not None else self.base_seed) + i)
            obs_list.append(o)
            states.append(env.state())
        self.obs = self._pack(obs_list)
        self.state = np.stack(states).astype(np.float32)
        return self.obs, self.state

    def step(self, actions):
        """actions: dict[agent] -> np.ndarray[num_envs] of action ids."""
        rewards = {a: np.zeros(self.num_envs, dtype=np.float32) for a in AGENTS}
        dones = {a: np.zeros(self.num_envs, dtype=bool) for a in AGENTS}
        next_obs_list = []
        next_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)

        for i, env in enumerate(self.envs):
            # If the env was left in a terminated state (e.g. an evaluation
            # routine reset it and ran it to completion, or a previous caller
            # ended the episode), reset it before stepping. PettingZoo envs
            # clear `agents` once the episode is over.
            if not env.agents:
                env.reset()
            acts = {a: int(actions[a][i]) for a in AGENTS}
            o, r, t, tr, inf = env.step(acts)
            # REV-4 (REVISION_PLAN.md §3a): stash the true terminal
            # observation/state in inf[a]["terminal_observation"] before
            # resetting; return terminations and truncations separately so
            # the PPO loop (REV-3) can bootstrap correctly on truncation.
            done = bool(any(t.values()) or any(tr.values()))
            if done:
                o, _ = env.reset()
            for a in AGENTS:
                rewards[a][i] = r[a]
                dones[a][i] = done
            next_obs_list.append(o)
            next_states[i] = env.state()

        self.obs = self._pack(next_obs_list)
        self.state = next_states
        return self.obs, rewards, dones

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()
