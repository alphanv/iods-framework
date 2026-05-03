"""Phylogenetic null models (Sect. 4.4) for the C2 surplus test."""

from __future__ import annotations

from collections import defaultdict
from typing import Hashable

import numpy as np


class Taxonomy:
    """species -> genus -> family -> order -> class hierarchy."""

    RANKS = ("species", "genus", "family", "order", "class_")

    def __init__(self) -> None:
        self._ranks: dict[str, dict[str, str]] = {}

    def add(self, species, genus="", family="", order="", class_=""):
        self._ranks[species] = {
            "species": species, "genus": genus, "family": family,
            "order": order, "class_": class_,
        }

    def rank(self, species, level):
        if species not in self._ranks:
            raise KeyError(f"unknown species '{species}'")
        return self._ranks[species][level]

    def species_in_rank(self, level, value):
        return [sp for sp, r in self._ranks.items() if r[level] == value]


class TaxonomicMeanNull:
    """Predict trait as the mean within a taxonomic rank's training set."""

    def __init__(self, taxonomy: Taxonomy, rank: str = "genus") -> None:
        if rank not in Taxonomy.RANKS:
            raise ValueError(f"rank must be one of {Taxonomy.RANKS}")
        self.taxonomy = taxonomy
        self.rank = rank
        self.rank_means: dict[str, np.ndarray] = {}
        self.global_mean: np.ndarray | None = None

    def fit(self, train_species, train_traits):
        if len(train_species) != train_traits.shape[0]:
            raise ValueError("train_species and train_traits length mismatch")
        groups: dict[str, list[int]] = defaultdict(list)
        for i, sp in enumerate(train_species):
            key = self.taxonomy.rank(sp, self.rank)
            if key:
                groups[key].append(i)
        for key, idxs in groups.items():
            self.rank_means[key] = train_traits[idxs].mean(axis=0)
        self.global_mean = train_traits.mean(axis=0)

    def predict(self, species):
        if self.global_mean is None:
            raise RuntimeError("call fit() first")
        out = np.zeros((len(species), self.global_mean.shape[0]),
                       dtype=self.global_mean.dtype)
        for i, sp in enumerate(species):
            try:
                key = self.taxonomy.rank(sp, self.rank)
            except KeyError:
                key = ""
            if key in self.rank_means:
                out[i] = self.rank_means[key]
            else:
                out[i] = self.global_mean
        return out


class NearestNeighbourNull:
    """Stronger null: nearest training species by phylogenetic distance.

    Addresses Reviewer 1's concern that BM-on-k-mer-spectra is too weak.
    """

    def __init__(self, distance_fn) -> None:
        self.distance_fn = distance_fn
        self.train_species: list = []
        self.train_traits: np.ndarray | None = None

    def fit(self, train_species, train_traits):
        self.train_species = list(train_species)
        self.train_traits = np.asarray(train_traits)

    def predict(self, species):
        if self.train_traits is None:
            raise RuntimeError("call fit() first")
        out = np.zeros((len(species), self.train_traits.shape[1]),
                       dtype=self.train_traits.dtype)
        for i, sp in enumerate(species):
            best, best_d = None, float("inf")
            for j, train_sp in enumerate(self.train_species):
                d = self.distance_fn(sp, train_sp)
                if d < best_d:
                    best_d, best = d, j
            if best is not None:
                out[i] = self.train_traits[best]
        return out


class BrownianMotionNull:
    """Distance-weighted ancestral-state-style prediction (BM-like).

    Simplified: full REML estimation should use dendropy/ape in production.
    """

    def __init__(self, distance_fn) -> None:
        self.distance_fn = distance_fn
        self.train_species: list = []
        self.train_traits: np.ndarray | None = None

    def fit(self, train_species, train_traits):
        self.train_species = list(train_species)
        self.train_traits = np.asarray(train_traits)

    def predict(self, species, bandwidth: float = 1.0):
        if self.train_traits is None:
            raise RuntimeError("call fit() first")
        out = np.zeros((len(species), self.train_traits.shape[1]),
                       dtype=self.train_traits.dtype)
        for i, sp in enumerate(species):
            distances = np.array(
                [self.distance_fn(sp, t) for t in self.train_species]
            )
            weights = np.exp(-distances / bandwidth)
            wsum = weights.sum()
            if wsum > 0:
                weights = weights / wsum
                out[i] = (weights[:, None] * self.train_traits).sum(axis=0)
            else:
                out[i] = self.train_traits.mean(axis=0)
        return out


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_n * b_n).sum(axis=1)


def top_k_retrieval(pred: np.ndarray, targets: np.ndarray, k: int = 5) -> float:
    """Top-K cosine retrieval accuracy."""
    pred = np.asarray(pred)
    targets = np.asarray(targets)
    N = pred.shape[0]
    if N == 0:
        return 0.0
    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    tgt_n = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-12)
    sims = pred_n @ tgt_n.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1)
    return float((ranks < k).mean())


def phylogenetic_surplus(iods_pred, null_pred, targets, k: int = 5) -> dict:
    iods_acc = top_k_retrieval(iods_pred, targets, k=k)
    null_acc = top_k_retrieval(null_pred, targets, k=k)
    return {
        "iods_topk": iods_acc,
        "null_topk": null_acc,
        "surplus": iods_acc - null_acc,
        "k": k,
    }
