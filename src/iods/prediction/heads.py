"""Prediction heads: forward (continuous) and inverse (DNA features). Sect. 3.5."""

from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousHead(nn.Module):
    """Heteroskedastic forward head (Kendall & Gal 2017, Sect. 3.6)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden, out_dim)
        self.log_sigma2_head = nn.Linear(hidden, out_dim)

    def forward(self, z):
        h = self.shared(z)
        mu = self.mu_head(h)
        log_sigma2 = self.log_sigma2_head(h).clamp(min=-10.0, max=10.0)
        return mu, log_sigma2


class DNAFeatureHead(nn.Module):
    """Inverse-translation head: predicts k-mer / marker / Pfam features."""

    def __init__(
        self,
        in_dim: int,
        kmer_dim: int = 256,
        marker_dim: int = 512,
        pfam_dim: int = 1024,
        hidden: int = 512,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.kmer_head = nn.Linear(hidden, kmer_dim)
        self.marker_head = nn.Linear(hidden, marker_dim)
        self.pfam_head = nn.Linear(hidden, pfam_dim)
        self.kmer_dim = kmer_dim
        self.marker_dim = marker_dim
        self.pfam_dim = pfam_dim

    def forward(self, z):
        h = self.shared(z)
        return {
            "kmer_logits": self.kmer_head(h),
            "marker": self.marker_head(h),
            "pfam_logits": self.pfam_head(h),
        }
