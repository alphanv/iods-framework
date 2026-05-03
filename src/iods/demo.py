"""Quickstart demo: python -m iods.demo"""

from __future__ import annotations

import argparse

import numpy as np
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


def make_context(B, device):
    return {
        "t": torch.linspace(0.0, 5.0, B, device=device),
        "e": torch.randint(0, 4, (B,), device=device),
        "s": torch.randint(0, 4, (B,), device=device),
    }


def run_demo(n_epochs: int = 5, batch_size: int = 16, seed: int = 0, device: str = "cpu"):
    print("=" * 60)
    print("  IODS framework — quickstart demo")
    print("=" * 60)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n[1/4] Building synthetic species dataset ...")
    species = make_synthetic_dataset(seed=seed)
    print(f"      {len(species)} synthetic species generated.")
    print(f"      Modalities per species: DNA, image, audio, time-series, "
          f"phenotype, metadata, environment.")

    dataset = SyntheticIODSDataset(species, modality_dropout_prob=0.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_synthetic_batch)

    print("\n[2/4] Building IODS model ...")
    model = IODSModel(
        d_z=64, kmer_dim=256, marker_dim=64, pfam_dim=128, phenotype_out_dim=16,
        dna_kmer_k=3, dna_d_model=64, dna_n_layers=2, dna_max_len=512,
        n_fusion_layers=2, n_fusion_heads=4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Model has {n_params/1e6:.2f}M parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n[3/4] Training for {n_epochs} epochs ...")
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            B_actual = len(batch["species_id"])
            context = make_context(B_actual, device)
            for k in ("image", "audio", "timeseries", "phenotype",
                      "metadata", "environment", "kmer_target"):
                if isinstance(batch.get(k), torch.Tensor):
                    batch[k] = batch[k].to(device)

            out = model(batch, context)
            forward_loss = heteroskedastic_nll(
                out["forward_pred"]["mu"],
                out["forward_pred"]["log_sigma2"],
                batch["phenotype"],
            )
            kmer_loss = kmer_kl_divergence(
                out["inverse_pred"]["kmer_logits"], batch["kmer_target"]
            )
            loss = forward_loss + kmer_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"      epoch {epoch+1}/{n_epochs}  loss={epoch_loss/n_batches:.4f}")

    print("\n[4/4] Evaluating C2 phylogenetic surplus ...")
    model.eval()
    held_family = "family_00"
    train_species = [sp for sp in species if sp.family != held_family]
    test_species = [sp for sp in species if sp.family == held_family]
    print(f"      held-out family: {held_family}")
    print(f"      train species: {len(train_species)},  test: {len(test_species)}")

    iods_pred = []
    targets = []
    test_dataset = SyntheticIODSDataset(test_species)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_synthetic_batch)
    with torch.no_grad():
        for batch in test_loader:
            B = len(batch["species_id"])
            context = make_context(B, device)
            batch["dna_strings"] = None
            for k in ("image", "audio", "timeseries", "phenotype",
                      "metadata", "environment"):
                if isinstance(batch.get(k), torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch, context)
            kmer_pred = torch.softmax(out["inverse_pred"]["kmer_logits"], dim=-1)
            iods_pred.append(kmer_pred.cpu().numpy())
            targets.append(batch["kmer_target"].cpu().numpy())
    iods_pred = np.concatenate(iods_pred)
    targets = np.concatenate(targets)

    tax = Taxonomy()
    for sp in species:
        tax.add(sp.species_id, genus=sp.genus, family=sp.family)
    train_kmers = np.stack(
        [kmer_frequency_spectrum(sp.dna, k=4).numpy() for sp in train_species]
    )
    null = TaxonomicMeanNull(tax, rank="genus")
    null.fit([sp.species_id for sp in train_species], train_kmers)
    null_pred = null.predict([sp.species_id for sp in test_species])

    res = evaluate_c2(iods_pred, null_pred, targets, k=5, delta=0.05)
    print(f"      IODS top-5 retrieval:    {res['iods_topk']:.3f}")
    print(f"      null top-5 retrieval:    {res['null_topk']:.3f}")
    print(f"      surplus (IODS - null):   {res['surplus']:+.3f}")
    print(f"      C2 (delta={res['delta_threshold']}) passes? "
          f"{'YES' if res['passes'] else 'no'}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("  NOTE: synthetic data; no biological conclusions.")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_demo(n_epochs=args.epochs, batch_size=args.batch_size,
             seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
