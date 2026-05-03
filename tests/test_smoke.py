"""End-to-end smoke test for the full pipeline."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from iods import IODSModel, evaluate_c2, kmer_frequency_spectrum
from iods.data import (
    SyntheticIODSDataset, collate_synthetic_batch, make_synthetic_dataset,
)
from iods.prediction.losses import (
    cosine_distance, heteroskedastic_nll, kmer_kl_divergence,
)
from iods.validation.phylo_null import Taxonomy, TaxonomicMeanNull, top_k_retrieval


def _build_small_model():
    return IODSModel(
        d_z=64,
        meta_dim=8, phenotype_dim=16, env_dim=8,
        kmer_dim=256, marker_dim=64, pfam_dim=128,
        phenotype_out_dim=16,
        dna_kmer_k=3, dna_d_model=32, dna_n_layers=1, dna_max_len=512,
        n_fusion_layers=1, n_fusion_heads=2,
    )


def _build_context(B, device):
    return {
        "t": torch.linspace(0.0, 5.0, B, device=device),
        "e": torch.randint(0, 4, (B,), device=device),
        "s": torch.randint(0, 4, (B,), device=device),
    }


def test_smoke_model_forward():
    torch.manual_seed(0)
    model = _build_small_model()
    species = make_synthetic_dataset(seed=0)
    dataset = SyntheticIODSDataset(species)
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        collate_fn=collate_synthetic_batch)
    batch = next(iter(loader))
    context = _build_context(B=4, device="cpu")
    out = model(batch, context)
    assert out["z"].shape == (4, 64)
    assert out["magnitudes"].shape == (4, model.n_modalities)
    assert out["forward_pred"]["mu"].shape == (4, 16)
    assert out["inverse_pred"]["kmer_logits"].shape == (4, 256)
    assert torch.all(torch.isfinite(out["z"]))


def test_smoke_one_training_step_decreases_loss():
    torch.manual_seed(42)
    model = _build_small_model()
    species = make_synthetic_dataset(seed=0)
    dataset = SyntheticIODSDataset(species)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_synthetic_batch)
    batch = next(iter(loader))
    context = _build_context(B=8, device="cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def compute_loss():
        out = model(batch, context)
        forward_loss = heteroskedastic_nll(
            out["forward_pred"]["mu"],
            out["forward_pred"]["log_sigma2"],
            batch["phenotype"],
        )
        kmer_loss = kmer_kl_divergence(
            out["inverse_pred"]["kmer_logits"], batch["kmer_target"]
        )
        return forward_loss + kmer_loss, out

    model.train()
    initial_loss, _ = compute_loss()
    initial_loss_value = initial_loss.item()

    for _ in range(10):
        loss, _ = compute_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss, _ = compute_loss()
    assert final_loss.item() < initial_loss_value, (
        f"loss did not decrease: {initial_loss_value:.4f} -> {final_loss.item():.4f}"
    )


def test_smoke_training_modifies_parameters():
    torch.manual_seed(0)
    model = _build_small_model()
    species = make_synthetic_dataset(seed=1)
    dataset = SyntheticIODSDataset(species)
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_synthetic_batch)
    batch = next(iter(loader))
    context = _build_context(B=8, device="cpu")

    initial_w = model.magnitude.w.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    out = model(batch, context)
    forward_loss = heteroskedastic_nll(
        out["forward_pred"]["mu"],
        out["forward_pred"]["log_sigma2"],
        batch["phenotype"],
    )
    forward_loss.backward()
    optimizer.step()
    new_w = model.magnitude.w.detach()
    assert not torch.allclose(initial_w, new_w)


def test_smoke_phylogenetic_baseline_runs():
    species = make_synthetic_dataset(seed=2)
    tax = Taxonomy()
    for sp in species:
        tax.add(sp.species_id, genus=sp.genus, family=sp.family)

    kmer_spectra = np.stack(
        [kmer_frequency_spectrum(sp.dna, k=4).numpy() for sp in species]
    )
    species_ids = [sp.species_id for sp in species]

    held_family = "family_00"
    train_idx = [i for i, sp in enumerate(species) if sp.family != held_family]
    test_idx = [i for i, sp in enumerate(species) if sp.family == held_family]
    train_species = [species_ids[i] for i in train_idx]
    test_species = [species_ids[i] for i in test_idx]
    train_traits = kmer_spectra[train_idx]
    test_traits = kmer_spectra[test_idx]

    null = TaxonomicMeanNull(tax, rank="family")
    null.fit(train_species, train_traits)
    null_pred = null.predict(test_species)
    assert null_pred.shape == test_traits.shape

    acc = top_k_retrieval(null_pred, test_traits, k=5)
    assert 0.0 <= acc <= 1.0


def test_smoke_evaluate_c2_runs():
    rng = np.random.default_rng(0)
    N, D = 20, 16
    targets = rng.standard_normal((N, D))
    iods_pred = targets + 0.1 * rng.standard_normal((N, D))
    null_pred = rng.standard_normal((N, D))
    res = evaluate_c2(iods_pred, null_pred, targets, k=5, delta=0.05)
    assert "surplus" in res
    assert "passes" in res
    assert "iods_topk" in res
    assert "null_topk" in res
