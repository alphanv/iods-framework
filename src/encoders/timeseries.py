"""TSEncoder — Section 3.1. Temporal Convolutional Network."""

import torch
import torch.nn as nn


class TSEncoder(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 256, n_channels: int = 64):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, n_channels, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(n_channels, output_dim)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        return self.proj(self.tcn(ts))
