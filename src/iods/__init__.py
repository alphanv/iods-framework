"""IODS Framework — Intra-Organismal Data Symbiosis."""

from iods.model import IODSModel, MODALITY_NAMES
from iods.magnitude import (
    ContextEmbedding, MagnitudeFunction, MagnitudeFunctionSoftmax, apply_magnitude,
)
from iods.encoders import (
    DNAEncoder, ImageEncoder, AudioEncoder, TimeSeriesEncoder,
    MetaEncoder, PhenotypeEncoder, EnvironmentEncoder,
    kmer_frequency_spectrum,
)
from iods.fusion import CrossAttentionFusion, ModalityProjector
from iods.prediction import (
    ContinuousHead, DNAFeatureHead,
    cosine_distance, heteroskedastic_nll,
    info_nce_estimate, info_nce_loss,
    kmer_kl_divergence, pfam_bce, total_loss,
)
from iods.validation import (
    Taxonomy, TaxonomicMeanNull, NearestNeighbourNull, BrownianMotionNull,
    top_k_retrieval,
    evaluate_c1, evaluate_c2, evaluate_c3, evaluate_info_nce_symmetry,
)

__version__ = "0.1.0"

__all__ = [
    "IODSModel", "MODALITY_NAMES",
    "ContextEmbedding", "MagnitudeFunction", "MagnitudeFunctionSoftmax", "apply_magnitude",
    "DNAEncoder", "ImageEncoder", "AudioEncoder", "TimeSeriesEncoder",
    "MetaEncoder", "PhenotypeEncoder", "EnvironmentEncoder",
    "kmer_frequency_spectrum",
    "CrossAttentionFusion", "ModalityProjector",
    "ContinuousHead", "DNAFeatureHead",
    "cosine_distance", "heteroskedastic_nll",
    "info_nce_estimate", "info_nce_loss",
    "kmer_kl_divergence", "pfam_bce", "total_loss",
    "Taxonomy", "TaxonomicMeanNull", "NearestNeighbourNull", "BrownianMotionNull",
    "top_k_retrieval",
    "evaluate_c1", "evaluate_c2", "evaluate_c3", "evaluate_info_nce_symmetry",
]
