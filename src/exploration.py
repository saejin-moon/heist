"""
Exploration & Intrinsic Motivation Modules for HEIST.

Provides Random Network Distillation (RND) and Count-Based Intrinsic Motivation
to address the 'Sparsity Wall' in sequential MARL, enabling agents to navigate to
downstream objectives (e.g., cross-map extraction) even before extrinsic rewards fire.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from constants import OBSERVATION_SIZE


class RNDTargetNetwork(nn.Module):
    """Fixed random target network for RND."""

    def __init__(self, obs_dim: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, feature_dim),
        )
        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Trainable predictor network for RND."""

    def __init__(self, obs_dim: int, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule:
    """Random Network Distillation intrinsic curiosity module.

    Computes intrinsic reward as MSE loss between predictor and target networks:
    r_int = || f_pred(obs) - f_target(obs) ||^2
    """

    def __init__(
        self,
        obs_dim: int = OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1],
        feature_dim: int = 64,
        lr: float = 1e-4,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.target = RNDTargetNetwork(obs_dim, feature_dim).to(self.device)
        self.predictor = RNDPredictorNetwork(obs_dim, feature_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

        self.r_mean = 0.0
        self.r_std = 1.0
        self.count = 1e-4

    def compute_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute un-normalized or normalized intrinsic reward for a batch of observations."""
        with torch.no_grad():
            x = obs.float().flatten(start_dim=-2).to(self.device)
            target_feat = self.target(x)
            pred_feat = self.predictor(x)
            r_int = torch.sum((pred_feat - target_feat) ** 2, dim=-1)

            # Update running stats
            r_np = r_int.cpu().numpy()
            batch_mean = np.mean(r_np)
            batch_std = np.std(r_np)
            batch_count = r_np.size

            self.count += batch_count
            delta = batch_mean - self.r_mean
            self.r_mean += delta * batch_count / self.count
            self.r_std = max(1e-4, 0.99 * self.r_std + 0.01 * batch_std)

            normalized_r = r_int / (self.r_std + 1e-8)
            return normalized_r

    def update(self, obs: torch.Tensor) -> float:
        """Update predictor network to minimize error against target network."""
        x = obs.float().flatten(start_dim=-2).to(self.device)
        with torch.no_grad():
            target_feat = self.target(x)

        pred_feat = self.predictor(x)
        loss = torch.mean((pred_feat - target_feat) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()
