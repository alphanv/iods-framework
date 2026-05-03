"""Top-level IODS model — the full pipeline of Sect. 2.8."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from iods.encoders.dna_encoder import DNAEncoder
from iods.encoders.sensor_encoders import (
    AudioEncoder, EnvironmentEncoder, ImageEncoder,
    MetaEncoder, PhenotypeEncoder, TimeSeriesEncoder,
)
from iods.fusion.cross_attention import CrossAttentionFusion, ModalityProjector
from iods.magnitude.magnitude import ContextEmbedding, MagnitudeFunction
from iods.prediction.heads import ContinuousHead, DNAFeatureHead


MODALITY_NAMES = (
    "dna", "image", "audio", "timeseries",
    "phenotype", "metadata", "environment",
)


class IODSModel(nn.Module):
    """Full IODS pipeline: encoders -> magnitude -> fusion -> heads."""

    def __init__(
        self,
        d_z: int = 256,
        n_env_categories: int = 4,
        n_phase_categories: int = 4,
        meta_dim: int = 8,
        phenotype_dim: int = 16,
        env_dim: int = 8,
        kmer_dim: int = 256,
        marker_dim: int = 512,
        pfam_dim: int = 1024,
        phenotype_out_dim: int = 16,
        dna_kmer_k: int = 4,
        dna_d_model: int = 128,
        dna_n_layers: int = 2,
        dna_max_len: int = 512,
        n_fusion_layers: int = 2,
        n_fusion_heads: int = 4,
    ) -> None:
        super().__init__()

        self.dna_encoder = DNAEncoder(
            k=dna_kmer_k, d_model=dna_d_model, n_heads=4, n_layers=dna_n_layers,
            out_dim=d_z, max_len=dna_max_len,
        )
        self.image_encoder = ImageEncoder(out_dim=d_z)
        self.audio_encoder = AudioEncoder(out_dim=d_z)
        self.timeseries_encoder = TimeSeriesEncoder(out_dim=d_z)
        self.phenotype_encoder = PhenotypeEncoder(in_dim=phenotype_dim, out_dim=d_z)
        self.meta_encoder = MetaEncoder(in_dim=meta_dim, out_dim=d_z)
        self.environment_encoder = EnvironmentEncoder(in_dim=env_dim, out_dim=d_z)

        modality_dims = {name: d_z for name in MODALITY_NAMES}
        self.projector = ModalityProjector(modality_dims, d_z=d_z)
        self.modality_names = list(MODALITY_NAMES)
        self.n_modalities = len(self.modality_names)

        self.context_embedding = ContextEmbedding(
            n_env_categories=n_env_categories,
            n_phase_categories=n_phase_categories,
        )
        self.magnitude = MagnitudeFunction(
            n_modalities=self.n_modalities,
            context_embedding=self.context_embedding,
        )

        self.fusion = CrossAttentionFusion(
            d_z=d_z, n_modalities=self.n_modalities,
            n_heads=n_fusion_heads, n_layers=n_fusion_layers,
        )

        self.forward_head = ContinuousHead(in_dim=d_z, out_dim=phenotype_out_dim)
        self.inverse_head = DNAFeatureHead(
            in_dim=d_z, kmer_dim=kmer_dim, marker_dim=marker_dim, pfam_dim=pfam_dim,
        )
        self.d_z = d_z

    def encode_modalities(self, batch: dict):
        """Run each encoder; produce per-modality embeddings + presence vector."""
        B = None
        device = None
        for name in self.modality_names:
            tensor_for_b = None
            if name == "dna":
                if "dna_tokens" in batch and batch["dna_tokens"] is not None:
                    tensor_for_b = batch["dna_tokens"]
                elif "dna_strings" in batch and batch["dna_strings"] is not None:
                    B = len(batch["dna_strings"])
                    device = next(self.parameters()).device
                    break
            else:
                if name in batch and batch[name] is not None:
                    tensor_for_b = batch[name]
            if tensor_for_b is not None:
                B = tensor_for_b.shape[0]
                device = tensor_for_b.device
                break
        if B is None:
            raise ValueError("at least one modality must be present in batch")

        embeddings: dict = {}
        presence_list = []

        if "dna_tokens" in batch and batch["dna_tokens"] is not None:
            tokens = batch["dna_tokens"]
            mask = batch.get("dna_mask")
            embeddings["dna"] = self.dna_encoder(tokens, mask=mask)
            presence_list.append(torch.ones(B, device=device))
        elif "dna_strings" in batch and batch["dna_strings"] is not None:
            tokens, mask = self.dna_encoder.collate(batch["dna_strings"])
            tokens = tokens.to(device)
            mask = mask.to(device)
            embeddings["dna"] = self.dna_encoder(tokens, mask=mask)
            presence_list.append(torch.ones(B, device=device))
        else:
            embeddings["dna"] = torch.zeros(B, self.d_z, device=device)
            presence_list.append(torch.zeros(B, device=device))

        modality_to_encoder = {
            "image": self.image_encoder,
            "audio": self.audio_encoder,
            "timeseries": self.timeseries_encoder,
            "phenotype": self.phenotype_encoder,
            "metadata": self.meta_encoder,
            "environment": self.environment_encoder,
        }
        for name, encoder in modality_to_encoder.items():
            if name in batch and batch[name] is not None:
                embeddings[name] = encoder(batch[name])
                presence_list.append(torch.ones(B, device=device))
            else:
                embeddings[name] = torch.zeros(B, self.d_z, device=device)
                presence_list.append(torch.zeros(B, device=device))

        presence = torch.stack(presence_list, dim=1)
        return embeddings, presence

    def forward(self, batch: dict, context: dict) -> dict:
        embeddings, presence = self.encode_modalities(batch)
        stacked = self.projector(embeddings)
        m_raw = self.magnitude(context["t"], context["e"], context["s"])
        m = self.magnitude.renormalize_for_presence(m_raw, presence)
        z = self.fusion(stacked, m, presence)
        mu, log_sigma2 = self.forward_head(z)
        inverse = self.inverse_head(z)
        return {
            "z": z,
            "magnitudes": m,
            "magnitudes_raw": m_raw,
            "presence": presence,
            "forward_pred": {"mu": mu, "log_sigma2": log_sigma2},
            "inverse_pred": inverse,
        }
