"""Tests for phylogenetic null models and retrieval metrics."""

import numpy as np
import pytest

from iods.validation.phylo_null import (
    BrownianMotionNull, NearestNeighbourNull, Taxonomy, TaxonomicMeanNull,
    cosine_similarity_np, phylogenetic_surplus, top_k_retrieval,
)


def test_taxonomy_basic():
    tax = Taxonomy()
    tax.add("sp_a", genus="g1", family="f1")
    tax.add("sp_b", genus="g1", family="f1")
    tax.add("sp_c", genus="g2", family="f1")
    tax.add("sp_d", genus="g3", family="f2")
    assert tax.rank("sp_a", "genus") == "g1"
    assert tax.rank("sp_d", "family") == "f2"
    assert set(tax.species_in_rank("genus", "g1")) == {"sp_a", "sp_b"}
    assert set(tax.species_in_rank("family", "f1")) == {"sp_a", "sp_b", "sp_c"}


def test_taxonomy_unknown_species_raises():
    tax = Taxonomy()
    with pytest.raises(KeyError):
        tax.rank("missing", "genus")


def test_taxonomic_mean_null_within_genus():
    tax = Taxonomy()
    for sp in ("a1", "a2", "a3"):
        tax.add(sp, genus="A", family="F")
    for sp in ("b1", "b2", "b3"):
        tax.add(sp, genus="B", family="F")
    train_species = ["a1", "a2", "b1", "b2"]
    train_traits = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0], [0.0, 7.0]])
    null = TaxonomicMeanNull(tax, rank="genus")
    null.fit(train_species, train_traits)
    pred = null.predict(["a3", "b3"])
    np.testing.assert_allclose(pred[0], [2.0, 0.0])
    np.testing.assert_allclose(pred[1], [0.0, 6.0])


def test_taxonomic_mean_null_falls_back_for_unseen():
    tax = Taxonomy()
    tax.add("a1", genus="A")
    tax.add("a2", genus="A")
    tax.add("c1", genus="C")
    null = TaxonomicMeanNull(tax, rank="genus")
    null.fit(["a1", "a2"], np.array([[1.0], [3.0]]))
    pred = null.predict(["c1"])
    np.testing.assert_allclose(pred[0], [2.0])


def test_top_k_retrieval_perfect():
    targets = np.eye(10)
    pred = targets.copy()
    assert top_k_retrieval(pred, targets, k=1) == 1.0
    assert top_k_retrieval(pred, targets, k=5) == 1.0


def test_top_k_retrieval_random():
    rng = np.random.default_rng(0)
    N, D = 200, 64
    pred = rng.standard_normal((N, D))
    targets = rng.standard_normal((N, D))
    acc = top_k_retrieval(pred, targets, k=5)
    assert acc < 0.15


def test_top_k_retrieval_orthogonal_correct():
    N = 5
    targets = np.eye(N)
    pred = np.eye(N)
    assert top_k_retrieval(pred, targets, k=1) == 1.0


def test_phylogenetic_surplus_signs():
    rng = np.random.default_rng(0)
    N, D = 50, 32
    targets = rng.standard_normal((N, D))
    iods_pred = targets.copy()
    null_pred = rng.standard_normal((N, D))
    out = phylogenetic_surplus(iods_pred, null_pred, targets, k=5)
    assert out["iods_topk"] >= 0.99
    assert out["null_topk"] < 0.5
    assert out["surplus"] > 0.5


def test_nearest_neighbour_null():
    distances = {
        ("a", "b"): 1.0, ("a", "c"): 5.0,
        ("d", "b"): 4.0, ("d", "c"): 0.5,
    }

    def dist(x, y):
        return distances.get((x, y), 100.0)

    null = NearestNeighbourNull(dist)
    null.fit(["b", "c"], np.array([[10.0], [20.0]]))
    pred = null.predict(["a", "d"])
    np.testing.assert_allclose(pred, [[10.0], [20.0]])


def test_brownian_motion_null_weighted_average():
    def dist(x, y):
        if y == "b":
            return 1.0
        return 2.0

    null = BrownianMotionNull(dist)
    null.fit(["b", "c"], np.array([[1.0], [10.0]]))
    pred = null.predict(["a"], bandwidth=1.0)
    assert 1.0 < pred[0, 0] < 5.5


def test_cosine_similarity_np():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [1.0, 0.0]])
    sims = cosine_similarity_np(a, b)
    np.testing.assert_allclose(sims, [1.0, 0.0], atol=1e-6)
