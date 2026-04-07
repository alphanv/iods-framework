"""Magnitude function M_m(t,e,s) — Section 2.6.

Context-dependent modality weighting with sigmoid gating.
Encodes the IODS premise that all modalities always contribute (M_m > 0)
but their relative importance varies across temporal, environmental,
and developmental contexts.

Proposition 1 separability: magnitude parameters {w_m, b_m} are
separable from cross-attention weights under conditions A1-A2.
"""

import torch
import torch.nn as nn


class MagnitudeFunction(nn.Module):
    """Sigmoid-gated magnitude: M_m(c) = sigma(w_m^T c + b_m)."""

    def __init__(self, n_modalities: int, context_dim: int):
        super().__init__()
        self.n_modalities = n_modalities
        # One linear layer per modality: w_m and b_m
        self.weights = nn.Linear(context_dim, n_modalities)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, context_dim] concatenation of [t_embed; e_embed; s_embed]
        Returns:
            magnitudes: [batch, n_modalities] in (0, 1)
        """
        return torch.sigmoid(self.weights(context))

    def renormalize(self, magnitudes: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """Zero absent modalities and renormalize remaining.

        Args:
            magnitudes: [batch, n_modalities]
            presence: [batch, n_modalities] binary indicator
        """
        masked = magnitudes * presence
        total = masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return masked / total
