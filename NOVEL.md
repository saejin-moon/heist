# SYSTEM INSTRUCTION: MARL Causal Credit Assignment Implementation
**Objective:** Implement two novel, theoretically rigorous credit-assignment algorithms into the `HEIST` codebase. 

## PART 1: Implement CIR (Causal Influence Routing)
**Concept:** Ablate (zero-out) messages in the TarMAC channel to measure their counterfactual impact on the receiver's Value Function. Use this exact causal influence to route the GAE advantage vector.

### 1.1: Add the Influence Matrix Calculation
**File:** `src/model.py`
**Target:** Inside the `CommAgent` class.
**Action:** Add a new method `get_influence_matrix` that calculates the counterfactual impact of muting each sender.
```python
    def get_influence_matrix(self, obs_list, role_list):
        """Calculates the Causal Influence Routing (CIR) matrix via Counterfactual Message Ablation."""
        features = self._joint_features(obs_list, role_list)
        encoded = self.encoder(features)
        aggregated, messages_gated, attention = self.comm(encoded)
        
        B, n, _ = features.shape
        x_base = torch.cat([features, aggregated], dim=-1)
        # Baseline value function with all messages
        base_values = self.critic(x_base.view(B * n, -1)).view(B, n)
        
        # [B, receiver, sender]
        influence_matrix = torch.zeros((B, n, n), device=features.device)
        
        for sender in range(n):
            # Counterfactual: What if we mute this sender?
            alt_aggregated = aggregated.clone()
            msg_s = messages_gated[:, sender:sender+1, :] # [B, 1, dim]
            att_s = attention[:, :, sender:sender+1]      # [B, n, 1]
            
            # Ablate the sender's message from the aggregation
            alt_aggregated -= att_s * msg_s 
            
            x_alt = torch.cat([features, alt_aggregated], dim=-1)
            alt_values = self.critic(x_alt.view(B * n, -1)).view(B, n)
            
            # Influence is the absolute change in the Value Function
            influence_matrix[:, :, sender] = torch.abs(base_values - alt_values)
            
        # Zero out self-influence (diagonal)
        mask = torch.eye(n, device=features.device).bool()
        influence_matrix.masked_fill_(mask, 0.0)
        
        # Normalize across senders to get routing weights (sum to 1 per receiver)
        row_sums = influence_matrix.sum(dim=-1, keepdim=True) + 1e-8
        routing_weights = influence_matrix / row_sums
        
        return routing_weights
```

### 1.2: Capture and Route Advantage
**File:** `src/train_comm.py`
**Target 1:** Add `--cir-coef` (default `0.0`) to `parse_args()`.
**Target 2:** Initialize `buf_influence = torch.zeros((args.num_steps, args.num_envs, n, n), device=device)` near line 125.
**Target 3:** Inside the rollout loop, call `buf_influence[step] = policy.get_influence_matrix(obs_list, role_list).detach()`.
**Target 4:** Immediately after `compute_gae` (around line 260), route the advantages:
```python
        # --- CIR: Causal Influence Routing ---
        if args.cir_coef > 0.0:
            adv_t = stacked_advantages.permute(1, 2, 0) # [steps, envs, n_agents]
            # Route Advantage from Receiver to Sender based on Causal Influence
            routed_adv = torch.einsum('sten,ste->stn', buf_influence, adv_t)
            # Convex combination to preserve total advantage variance
            adv_t_new = (1.0 - args.cir_coef) * adv_t + (args.cir_coef * routed_adv)
            stacked_advantages = adv_t_new.permute(2, 0, 1)
```

---

## PART 2: Implement CAR (Counterfactual Affordance Reward)
**Concept:** If an agent's action causes a teammate's `action_mask` to flip from `0` to `1` (unlocking an affordance), issue an intrinsic reward equal to the Centralized Critic's evaluation of the newly unlocked state.

### 2.1: Detect Action Space Expansion
**File:** `src/env.py`
**Target:** Modify `step()` to capture old masks, compare them to new masks, and flag the responsible agent.
```python
    def step(self, actions):
        self.current_step += 1
        rewards = {a: self.config["reward_time_bleed"] for a in self.agents}
        
        # --- CAR Tracking: Capture pre-step masks & actors ---
        old_masks = {a: self._action_mask(a) for a in self.agents}
        interact_actors = [a for a in self.agents if actions[a] == INTERACT]
        
        # ... [Keep existing step logic here] ...
        
        # --- CAR Tracking: Detect affordance unlocks ---
        new_masks = {a: self._action_mask(a) for a in self.agents}
        expansion_occurred = False
        for a in self.agents:
            if old_masks[a][INTERACT] == 0 and new_masks[a][INTERACT] == 1:
                expansion_occurred = True
                break
                
        infos = {a: {"alarm": self.alarm, "win": bool(win), "lose": bool(lose)} for a in self.agents}
        for a in self.agents:
            # Credit goes to the agent(s) who took the INTERACT action this turn
            infos[a]["car_unlocked"] = (expansion_occurred and a in interact_actors)
            
        observations = self._get_all_obs()
```

### 2.2: Fix vec_env Info Dropping (CRITICAL BUGFIX)
**File:** `src/vec_env.py`
**Target:** Inside `step()`
```python
            # CHANGE THIS:
            o, r, t, tr, inf = env.step(acts)
            done = bool(any(t.values()) or any(tr.values()))
            # ...
            if done:
                infos[i]["terminal_observation"] = self._pack([o])
                
            # TO THIS:
            o, r, t, tr, inf = env.step(acts)
            infos[i] = inf  # Ensure the info dict passes through the vector wrapper!
            done = bool(any(t.values()) or any(tr.values()))
            # ...
            if done:
                infos[i]["terminal_observation"] = self._pack([o])
```

### 2.3: Apply the Centralized Critic Reward
**File:** `src/train_mappo.py`
**Target:** Add `--car-coef` (default `0.0`) to `Args` and `parse_args()`. Immediately after `vec_env.step(actions_dict)` in the rollout loop, apply the reward:
```python
            # --- CAR: Counterfactual Affordance Reward ---
            if args.car_coef > 0.0:
                with torch.no_grad():
                    next_state_t = torch.as_tensor(vec_env.state, device=device)
                    v_s_next = policy.get_value(next_state_t).squeeze(-1) # [num_envs]
                
                for env_idx in range(args.num_envs):
                    for a in AGENTS:
                        if infos[env_idx][a].get("car_unlocked", False):
                            # The agent unlocked a new action for the team. 
                            # Reward = beta * max(0, V(new_state))
                            intrinsic_bonus = args.car_coef * max(0.0, v_s_next[env_idx].item())
                            rewards[a][env_idx] += intrinsic_bonus
```