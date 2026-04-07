"""Inverse translation head — Section 3.5.

Predicts three DNA feature types:
  (i)   k-mer frequency spectra (256-dim for k=4)
  (ii)  marker gene embeddings (512-dim)
  (iii) Pfam domain profiles (~1000-dim binary)
"""

import torch
import torch.nn as nn


class InverseHead(nn.Module):
    def __init__(self, latent_dim: int = 512, kmer_dim: int = 256,
                 marker_dim: int = 512, pfam_dim: int = 1000):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
        )
        self.kmer_head = nn.Linear(latent_dim, kmer_dim)
        self.marker_head = nn.Linear(latent_dim, marker_dim)
        self.pfam_head = nn.Sequential(
            nn.Linear(latent_dim, pfam_dim), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> dict:
        h = self.shared(z)
        return {
            "kmer_spectrum": self.kmer_head(h),
            "marker_embedding": self.marker_head(h),
            "pfam_profile": self.pfam_head(h),
        }
