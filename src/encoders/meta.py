"""MetaEncoder — Section 3.1. MLP for baseline metadata."""

import torch
import torch.nn as nn


class MetaEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.mlp(meta)
