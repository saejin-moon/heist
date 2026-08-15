# E-CO-OP: Evolutionary Confidence-Oriented Option Pool

The Evolutionary Confidence-Oriented Option Pool (E-CO-OP) integrates multi-agent dynamic routing with an internal, Fisher-guided genetic recombination loop. The architecture resolves causal credit dilution in Dec-POMDPs by strictly isolating sub-policies into specialized experts. But it does not rely on static gradient descent to optimize these experts. The system operates a concurrent Evolutionary Strategy (ES) to systematically crossover and mutate the internal parameter space.

## Decentralized Confidence Routing

Traditional hierarchical reinforcement learning enforces a top-down manager. But this design introduces structural lock-in. If the manager learns a suboptimal routing trajectory, the workers optimize for that flawed distribution. The entire system collapses into a local minimum.

E-CO-OP removes the top-down manager. The agent routes behavior bottom-up using the localized value functions of the experts themselves.

For a set of $ K $ active experts, each expert $ k $ maintains an actor network $ \pi_k(a|s) $ and a critic network $ V_k(s) $. At each timestep $ t $, the active expert $ e_t $ is selected deterministically by taking the maximum expected return across all critics.

$$
e_t = \arg\max_{k \in K} V_k(s_t)
$$

The selected expert controls the agent for that timestep. The critics are trained via standard Generalized Advantage Estimation (GAE). So the value functions converge to the true expected return of the specific phase of the environment that the expert has mastered.

## Causal Affordance Shielding

Dec-POMDPs suffer from causal credit dilution. Upstream agents execute necessary affordances, but they receive zero immediate reward. Downstream agents complete the terminal objective and absorb the shared return. E-CO-OP hard-masks the terminal rewards retroactively based on localized causal affordances.

If an upstream agent unlocks a necessary door at $ t_{afford} $, the environment broadcasts a binary success mask $ M \in \{0, 1\} $. The baseline advantage calculation is multiplied by this structural mask. Upstream failures are zeroed out if the affordance is not met. Downstream terminal rewards are scaled proportionally.

$$
A_i(s, a) = M \cdot \left( Q_i(s, a) - V_i(s) \right)
$$

## The Evolutionary Generational Cadence

The static CO-OP architecture spawns new experts reactively when the value function drops below a hard threshold. E-CO-OP discards reactive spawning. The framework executes a rigid generational cadence every $ N $ episodes.

The algorithm evaluates the mean empirical advantage $ \bar{A}_k $ for all active experts over the trailing $ N $ episodes. The system ranks the pool and strictly deletes the bottom 50% of the active experts. This creates a persistent structural vacuum. The remaining top-performing experts are selected as parents for the recombination phase.

## Fisher-Guided Genetic Crossover

Deep neural networks resist genetic crossover. Permutation invariance causes severe destructive interference when arbitrary matrices are mixed. E-CO-OP bypasses this alignment problem because all experts are spawned as topological clones. They share the same phylogenetic ancestry.

To ensure the crossover does not destroy established task heuristics, the operation is guided by the empirical Fisher Information Matrix (FIM). The FIM calculates the mathematical sensitivity of the policy to every single parameter $ \theta_i $.

$$
F_i = \mathbb{E} \left[ \left( \nabla_{\theta_i} \log \pi_\theta(a|s) \right)^2 \right]
$$

During the crossover phase, the system runs a single forward-backward pass over a held-out sample batch to compute $ F_A $ and $ F_B $ for the two parent experts. The child expert $ C $ inherits each individual parameter $ \theta_i $ from the parent possessing the higher Fisher scalar for that specific weight.

$$
\theta_{C, i} = \begin{cases} \theta_{A, i} & \text{if } F_{A, i} > F_{B, i} \\ \theta_{B, i} & \text{otherwise} \end{cases}
$$

The child becomes an optimal mosaic. It retains the strongest feature detectors from both parents.

## Inverse-Proportional Mutational Noise

Standard mutational exploration applies uniform Gaussian noise across all parameters. But uniform noise destroys highly confident weights. E-CO-OP targets the mutation explicitly into the uncertain parameter space.

The system normalizes the inherited Fisher confidence $ \hat{F}_{C, i} $ to a range of $ [0, 1] $. Gaussian noise is scaled by the inverse of this confidence matrix.

$$
\theta_{C, i} = \theta_{C, i} + \mathcal{N}(0, \sigma) \cdot (1 - \hat{F}_{C, i})
$$

Massive weights with high structural confidence receive zero mutational interference. They are locked. Uncertain, near-zero parameters absorb the full magnitude of the Gaussian noise. This systematically forces the neural network to explore new state boundaries without degrading its established baseline capabilities.
