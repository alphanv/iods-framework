"""Cross-attention fusion with modality presence indicators — Sections 2.8, 3.2."""

import torch
import torch.nn as nn
from typing import Dict


class CrossAttentionFusion(nn.Module):
    def __init__(self, latent_dim: int = 512, n_modalities: int = 5, n_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.presence_proj = nn.Linear(n_modalities, latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, embeddings: Dict[str, torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        if not embeddings:
            raise ValueError("At least one modality must be present")
        tokens = list(embeddings.values())
        stacked = torch.stack(tokens, dim=1)  # [batch, n_available, dim]
        h_proj = self.presence_proj(h).unsqueeze(1)  # [batch, 1, dim]
        seq = torch.cat([stacked, h_proj], dim=1)
        attended, _ = self.attn(seq, seq, seq)
        out = self.norm(attended.mean(dim=1))
        return out + self.ffn(out)
