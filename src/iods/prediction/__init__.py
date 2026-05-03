"""Prediction heads and loss functions."""

from iods.prediction.heads import ContinuousHead, DNAFeatureHead
from iods.prediction.losses import (
    cosine_distance,
    heteroskedastic_nll,
    info_nce_estimate,
    info_nce_loss,
    kmer_kl_divergence,
    pfam_bce,
    total_loss,
)

__all__ = [
    "ContinuousHead", "DNAFeatureHead",
    "cosine_distance", "heteroskedastic_nll",
    "info_nce_estimate", "info_nce_loss",
    "kmer_kl_divergence", "pfam_bce", "total_loss",
]
