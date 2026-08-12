with open("src/model.py", "r") as f:
    content = f.read()

replacement = """        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def reset(self):
        # Actor
        layer_init(self.actor[0])
        layer_init(self.actor[2])
        layer_init(self.actor[4], std=0.01)
        # Critic
        layer_init(self.critic[0])
        layer_init(self.critic[2])
        layer_init(self.critic[4], std=1.0)"""

content = content.replace("""        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim + 4, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )""", replacement)

with open("src/model.py", "w") as f:
    f.write(content)

print("Added reset() to CoopExpert")
