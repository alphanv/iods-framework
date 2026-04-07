"""IODSModel: Full IODS computational pipeline (Section 2.8).

Guess-compare-update loop using freely available biological databases.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from .encoders.dna import DNAEncoder
from .encoders.image import ImageEncoder
from .encoders.audio import AudioEncoder
from .encoders.timeseries import TSEncoder
from .encoders.meta import MetaEncoder
from .magnitude.magnitude import MagnitudeFunction
from .fusion.cross_attention import CrossAttentionFusion
from .prediction.forward import ForwardHead
from .prediction.inverse import InverseHead

MODALITY_NAMES = ["dna", "image", "audio", "timeseries", "meta"]

class IODSModel(nn.Module):
    def __init__(self, dna_dim=512, img_dim=512, audio_dim=256, ts_dim=256,
                 meta_dim=64, latent_dim=512, n_modalities=5, context_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleDict({
            "dna": DNAEncoder(output_dim=dna_dim),
            "image": ImageEncoder(output_dim=img_dim),
            "audio": AudioEncoder(output_dim=audio_dim),
            "timeseries": TSEncoder(output_dim=ts_dim),
            "meta": MetaEncoder(output_dim=meta_dim),
        })
        dims = {"dna": dna_dim, "image": img_dim, "audio": audio_dim,
                "timeseries": ts_dim, "meta": meta_dim}
        self.projections = nn.ModuleDict({
            k: nn.Linear(v, latent_dim) for k, v in dims.items()
        })
        self.magnitude = MagnitudeFunction(n_modalities, context_dim * 3)
        self.fusion = CrossAttentionFusion(latent_dim, n_modalities)
        self.forward_head = ForwardHead(latent_dim)
        self.inverse_head = InverseHead(latent_dim)

    def forward(self, inputs: Dict[str, torch.Tensor],
                context: Optional[torch.Tensor] = None):
        embeddings = {}
        for name in MODALITY_NAMES:
            if name in inputs and inputs[name] is not None:
                embeddings[name] = self.projections[name](self.encoders[name](inputs[name]))
        batch_size = next(iter(embeddings.values())).shape[0]
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.n_modalities, device=device)
        for i, name in enumerate(MODALITY_NAMES):
            if name in embeddings:
                h[:, i] = 1.0
        if context is not None:
            mags = self.magnitude(context)
            mags = self.magnitude.renormalize(mags, h)
            weighted = {k: mags[:, i:i+1] * v for i, (k, v) in
                        enumerate((n, embeddings[n]) for n in MODALITY_NAMES if n in embeddings)}
        else:
            weighted = embeddings
        z = self.fusion(weighted, h)
        return {"z": z, "forward": self.forward_head(z), "inverse": self.inverse_head(z)}
