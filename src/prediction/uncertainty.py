"""Heteroskedastic uncertainty estimation — Section 3.6. Kendall & Gal (2017)."""

import torch
import torch.nn as nn


class HeteroskedasticHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mu_head = nn.Linear(input_dim, output_dim)
        self.logvar_head = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor):
        return self.mu_head(x), self.logvar_head(x)

    @staticmethod
    def nll_loss(mu, logvar, target):
        var = torch.exp(logvar)
        return (logvar + (target - mu) ** 2 / var).mean()
