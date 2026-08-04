"""
Vectorized environment wrapper for HEIST.

Wraps `num_envs` independent HeistEnv instances so the PPO training loops
can collect rollouts in a simple numpy array layout (one buffer per agent),
mirroring the CleanRL vector-env convention.

Each env is stepped with a dict of per-agent actions.  When an episode
terminates or truncates, that env is immediately reset so the rollout buffer
can be kept dense.  REV-4 (REVISION_PLAN.md §3a) fixes the contract so the
TRUE terminal observation is not lost: ``step`` stashes it (packed per agent)
in ``infos[env]["terminal_observation"]`` (and the terminal env-state in
``infos[env]["terminal_state"]``) BEFORE the internal reset, and returns
terminations and truncations separately so the trainer can bootstrap value
estimates correctly on truncation (REV-3, REVISION_PLAN.md §3b).
"""

import numpy as np

from env import AGENTS, HeistEnv


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
                # REV-7 (REVISION_PLAN.md §6): global_state deleted from the
                # per-agent obs contract; central critics still get env.state().
                "role_id": np.stack([o[a]["role_id"] for o in obs_list]),
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
        # state_dim may change after reset (guards are spawned during reset),
        # so recompute from the actual state vector.
        self.state_dim = self.state.shape[1]
        return self.obs, self.state

    def step(self, actions):
        """actions: dict[agent] -> np.ndarray[num_envs] of action ids.

        Returns (obs, rewards, terminations, truncations, infos):
          * obs          : packed post-step observation (for done envs this is
                           the POST-RESET observation so the next rollout step
                           always selects actions on a valid state).
          * terminations : dict[agent] -> np.ndarray[num_envs] of bools.
          * truncations  : dict[agent] -> np.ndarray[num_envs] of bools.
          * infos        : per-env dicts; done envs carry
                           "terminal_observation" (packed) and "terminal_state"
                           so REV-3 bootstrapping sees the true terminal state.
        """
        rewards = {a: np.zeros(self.num_envs, dtype=np.float32) for a in AGENTS}
        terminations = {a: np.zeros(self.num_envs, dtype=bool) for a in AGENTS}
        truncations = {a: np.zeros(self.num_envs, dtype=bool) for a in AGENTS}
        next_obs_list = []
        next_states = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]

        for i, env in enumerate(self.envs):
            # If the env was left in a terminated state (e.g. an evaluation
            # routine reset it and ran it to completion, or a previous caller
            # ended the episode), reset it before stepping. PettingZoo envs
            # clear `agents` once the episode is over.
            if not env.agents:
                env.reset()
            acts = {a: int(actions[a][i]) for a in AGENTS}
            o, r, t, tr, inf = env.step(acts)
            done = bool(any(t.values()) or any(tr.values()))
            for a in AGENTS:
                rewards[a][i] = r[a]
                terminations[a][i] = bool(t[a])
                truncations[a][i] = bool(tr[a])
            if done:
                # REV-4: stash the true terminal observation/state BEFORE reset.
                infos[i]["terminal_observation"] = self._pack([o])
                infos[i]["terminal_state"] = env.state()
                o, _ = env.reset()
            next_obs_list.append(o)
            next_states[i] = env.state()

        self.obs = self._pack(next_obs_list)
        self.state = next_states
        return self.obs, rewards, terminations, truncations, infos

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()
