"""AudioEncoder — Section 3.1. Mel-spectrogram + CNN."""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int = 128, output_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, output_dim)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.proj(self.cnn(mel_spec))
