"""Phylogenetic null models and PGLS — Section 4.4.

Implements Brownian motion, Ornstein-Uhlenbeck, and early-burst null models.
Uses Pagel's lambda estimated from data.
"""

import numpy as np
from typing import Optional


class PhylogeneticNull:
    """Phylogenetic null model for C2 evaluation."""

    def __init__(self, tree_path: str, models: list = None):
        self.tree_path = tree_path
        self.models = models or ["BM", "OU", "EB"]
        # In practice, use dendropy/ete3 to load and process phylogeny

    def predict(self, taxonomy: np.ndarray) -> np.ndarray:
        """Predict modality values from taxonomy alone."""
        # Placeholder: return taxonomic group means
        raise NotImplementedError("Implement with dendropy for real phylogenies")

    def pgls_residual_test(self, residuals: np.ndarray, tree) -> dict:
        """Test whether residuals show phylogenetic signal."""
        raise NotImplementedError("Implement with R/rpy2 or scipy")
