"""
Magnitude function M_m(t, e, s).

Implements formula [F1] from Vardarli (2026a):

    M_m(t, e, s) = sigma(w_m^T . [t_embed; e_embed; s_embed] + b_m)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ContextEmbedding(nn.Module):
    def __init__(
        self,
        n_env_categories: int = 4,
        n_phase_categories: int = 4,
        d_time: int = 32,
        d_env: int = 16,
        d_phase: int = 16,
    ) -> None:
        super().__init__()
        if d_time % 2 != 0:
            raise ValueError(f"d_time must be even, got {d_time}")
        self.d_time = d_time
        self.d_env = d_env
        self.d_phase = d_phase
        self.env_embedding = nn.Embedding(n_env_categories, d_env)
        self.phase_embedding = nn.Embedding(n_phase_categories, d_phase)

    @property
    def context_dim(self) -> int:
        return self.d_time + self.d_env + self.d_phase

    def _time_encoding(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.d_time // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, t: torch.Tensor, e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        t_emb = self._time_encoding(t.float())
        e_emb = self.env_embedding(e.long())
        s_emb = self.phase_embedding(s.long())
        return torch.cat([t_emb, e_emb, s_emb], dim=-1)


class MagnitudeFunction(nn.Module):
    """Per-modality magnitude function M_m(t, e, s) — formula F1.

    Sigmoid-gated, parameter-efficient, interpretable per Sect. 2.6.
    Each modality's magnitude is independent in (0, 1).
    """

    def __init__(self, n_modalities: int, context_embedding: ContextEmbedding) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.context_embedding = context_embedding
        d_c = context_embedding.context_dim
        self.w = nn.Parameter(torch.randn(n_modalities, d_c) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_modalities))

    def forward(self, t, e, s):
        c = self.context_embedding(t, e, s)
        logits = c @ self.w.t() + self.b
        return torch.sigmoid(logits)

    def renormalize_for_presence(self, magnitudes, presence):
        """Sect. 2.8 step 4: zero absent modalities and renormalize."""
        if magnitudes.shape != presence.shape:
            raise ValueError(
                f"shape mismatch: magnitudes {magnitudes.shape} vs "
                f"presence {presence.shape}"
            )
        masked = magnitudes * presence
        sum_before = magnitudes.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sum_after = masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return masked * (sum_before / sum_after)


class MagnitudeFunctionSoftmax(nn.Module):
    """Softmax-normalized variant (ablation per Sect. 2.6)."""

    def __init__(self, n_modalities, context_embedding):
        super().__init__()
        self.n_modalities = n_modalities
        self.context_embedding = context_embedding
        d_c = context_embedding.context_dim
        self.w = nn.Parameter(torch.randn(n_modalities, d_c) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_modalities))

    def forward(self, t, e, s):
        c = self.context_embedding(t, e, s)
        logits = c @ self.w.t() + self.b
        return torch.softmax(logits, dim=-1)

    def renormalize_for_presence(self, magnitudes, presence):
        masked = magnitudes * presence
        return masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def apply_magnitude(embeddings: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
    """Formula F2: phi_tilde_m = M_m . phi_m."""
    if embeddings.dim() != 3:
        raise ValueError(
            f"embeddings must be (B, M, D), got shape {tuple(embeddings.shape)}"
        )
    if magnitudes.shape != embeddings.shape[:2]:
        raise ValueError(
            f"magnitudes shape {tuple(magnitudes.shape)} must match "
            f"embeddings batch/modality dims {tuple(embeddings.shape[:2])}"
        )
    return embeddings * magnitudes.unsqueeze(-1)
