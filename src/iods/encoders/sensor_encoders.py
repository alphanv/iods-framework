"""Sensor-modality encoders (image, audio, time-series, MLPs). Sect. 3.1."""

from __future__ import annotations

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512, in_channels: int = 3) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class AudioEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_dim: int = 256, hidden: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2, dilation=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MetaEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PhenotypeEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EnvironmentEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64, hidden: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
