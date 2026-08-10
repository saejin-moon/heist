# CO-OP: Confidence-Oriented Option Pool

Hierarchical reinforcement learning in multi-agent systems usually relies on a top-down manager network to assign tasks. The manager looks at the state and picks a worker. The problem with this structure is manager mode-collapse. The manager learns to pick the exact same worker for every state because it is easier than learning how to route properly. The chosen worker is forced to learn the entire environment, defeating the purpose of the hierarchy.

CO-OP solves this by deleting the manager entirely.

Instead of a top-down network dictating assignments, CO-OP uses a decentralized router where experts bid for control. The model maintains a pool of expert neural networks. At any given step, the state $s_t$ is passed to all active experts. Each expert outputs a value prediction $V_k(s_t)$. We treat this value prediction as a confidence score. The router simply takes the argmax of these values. The expert with the highest confidence takes control of the agent for that step:
$$$
k^* = \arg\max_{k \in \{1 \dots K\}} V_k(s_t)
$$$

Because routing is an emergent property of value accuracy rather than a separate policy, mode-collapse mathematically cannot happen.

## Spawning and Optimistic Nativism

Standard MoE networks force you to hardcode the number of workers before training. CO-OP starts with two experts and grows its capacity dynamically.

If the agent encounters a completely novel state distribution, the maximum confidence across all experts will drop. If this global confidence falls below a critical negative threshold $\tau_{spawn}$, the system spawns a new PyTorch expert:
$$$
\max(V_1(s_t), \dots, V_K(s_t)) < \tau_{spawn}
$$$

When a new expert is initialized, its weights output values near $0.0$. Since the spawn condition explicitly requires all existing experts to have a confidence below the negative threshold (e.g., $-0.1$), the new expert naturally wins the argmax vote on its very first step. It inherently thinks it can do better than the failing experts. We call this optimistic nativism. The system does not need a UCB exploration bonus to force new experts to train.

To prevent the model from uncontrollably spawning experts early in training, CO-OP enforces a strict burn-in period. Existing experts get plenty of time to learn the initial state distribution before the router is allowed to judge their confidence.

## Shielding and Structural Affordances

CO-OP fixes causal credit dilution. If an upstream agent successfully completes a critical subtask, but a downstream teammate fails ten steps later, the upstream agent still receives a negative terminal reward in standard MARL architectures.

CO-OP shields upstream agents from downstream failures, but it detects these successes structurally. We do not use hardcoded environment flags. We measure the volume of the multi-agent action mask. If the team's total action space expands between step $t$ and step $t+1$, an affordance was unlocked. The expert that took a causal state-altering action during that step receives dense credit. 

The base rewards are then multiplied by a macro weighting factor $\Omega_t$:
$$$
\Omega_t = \begin{cases} 
\exp(-\alpha C_t) & \text{if Expert } k^* \text{ triggered an affordance (Shielded)} \\
O \cdot \exp(-\alpha C_t) & \text{otherwise (Unshielded)}
\end{cases}
$$$
Here $O$ is the terminal outcome ($1.0$ or $-0.5$) and $C_t$ is a continuous global environment cost or penalty.

By multiplying the immediate rewards by this factor before calculating the generalized advantage estimate (GAE), the $\gamma \cdot \lambda$ trace naturally propagates the failure shielding backward through time. Competent experts are insulated from incompetent teammates.
