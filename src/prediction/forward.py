"""Forward translation head — Section 3.5."""

import torch
import torch.nn as nn


class ForwardHead(nn.Module):
    def __init__(self, latent_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
