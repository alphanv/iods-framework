"""Cross-attention fusion of magnitude-weighted modality embeddings.

Implements [F2] phi_tilde_m = M_m . phi_m and [F3] Z_i = f_enc(...).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from iods.magnitude.magnitude import apply_magnitude


class ModalityProjector(nn.Module):
    def __init__(self, modality_dims: dict, d_z: int) -> None:
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.projectors = nn.ModuleDict(
            {name: nn.Linear(d, d_z) for name, d in modality_dims.items()}
        )
        self.d_z = d_z

    def forward(self, embeddings: dict) -> torch.Tensor:
        projected = []
        for name in self.modality_names:
            if name not in embeddings:
                raise KeyError(f"missing embedding for modality '{name}'")
            projected.append(self.projectors[name](embeddings[name]))
        return torch.stack(projected, dim=1)


class CrossAttentionFusion(nn.Module):
    """Multi-head attention across modalities + presence indicator (Sect. 2.8)."""

    def __init__(
        self,
        d_z: int,
        n_modalities: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.d_z = d_z
        self.n_modalities = n_modalities
        self.use_cls_token = use_cls_token

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_z) * 0.02)

        self.presence_bias = nn.Parameter(torch.zeros(n_modalities, d_z))
        nn.init.normal_(self.presence_bias, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_z,
            nhead=n_heads,
            dim_feedforward=4 * d_z,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, embeddings, magnitudes, presence):
        B, M, D = embeddings.shape
        if M != self.n_modalities:
            raise ValueError(f"expected {self.n_modalities} modalities, got {M}")

        weighted = apply_magnitude(embeddings, magnitudes)
        bias = self.presence_bias.unsqueeze(0) * presence.unsqueeze(-1)
        weighted = weighted + bias

        key_padding_mask = (presence == 0)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, weighted], dim=1)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=presence.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        else:
            x = weighted

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if self.use_cls_token:
            return x[:, 0]
        present_f = presence.float().unsqueeze(-1)
        denom = present_f.sum(dim=1).clamp(min=1.0)
        return (x * present_f).sum(dim=1) / denom
