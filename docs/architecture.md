# Architecture Overview

A single-page description of how the IODS pipeline fits together. For the formal specification, see the formalisation paper (Vardarlı 2026, *From Theory to Test*); for the section-by-section mapping to source code, see [paper_to_code_map.md](paper_to_code_map.md).

## Pipeline

```
Raw inputs                                        Outputs
─────────                                         ───────

DNA string ─────► DNAEncoder ──┐
image      ─────► ImageEncoder ─┤
audio      ─────► AudioEncoder ─┤
timeseries ─────► TSEncoder ────┼──► ModalityProjector ──► (B, M, d_z)
phenotype  ─────► PhenoEncoder ─┤                                │
metadata   ─────► MetaEncoder ──┤                                │
environment─────► EnvEncoder ───┘                                │
                                                                 │
                 t, e, s ──► ContextEmbedding ──┐                │
                                                ▼                │
                                       MagnitudeFunction         │
                                                │                │
                                                ▼ (B, M)         │
                                       renormalize_for_presence ─┤
                                                │                │
                                                ▼                ▼
                                              CrossAttentionFusion
                                                         │
                                                         ▼ (B, d_z)
                                                       Z_i(t)
                                                ┌────────┴────────┐
                                                ▼                 ▼
                                        ContinuousHead      DNAFeatureHead
                                          │       │              │
                                          ▼       ▼              ▼
                                       (μ, log σ²)         (k-mer, marker, Pfam)
                                       phenotype           DNA features
                                       prediction          prediction
                                       (forward)           (inverse — the
                                                            decisive test)
```

## Three concepts that distinguish IODS from generic multimodal learning

**1. Magnitude function.** Each modality's contribution to the fused representation is gated by `M_m(t, e, s)` — a learned, context-dependent scalar in (0, 1). The same model in different contexts can lean more on DNA, more on phenotype, or more on environment. This formalises the paper's central claim: no modality is permanently privileged, but contextual asymmetry is real.

**2. Modality-presence indicator.** A binary vector `h` distinguishes "modality absent from this sample's data" from "modality present but uninformative." It is concatenated as a learned bias inside the fusion layer and used to mask attention. This resolves a training-test mismatch identified in prior multimodal work: models trained with random modality dropout can fail at inference when modalities are systematically missing.

**3. Inverse translation as decisive test.** The DNAFeatureHead predicts DNA features (k-mer spectra, marker gene embeddings, Pfam domain profiles) from the phenotypic ensemble. This is the direction that gene-centric biology says should not be possible. If inverse translation systematically fails — across all phylogenetic distances, all contexts, all null comparisons — then the IODS mutual-constraint claim is falsified.

## What the architecture does *not* claim

- It does not model mutual constraint mechanistically. It detects whether the empirical signature of mutual constraint (bidirectional information flow modulated by context) is present in real biological data.
- It does not require any biological modality to be ontologically primary. The architecture is symmetric across modalities by construction; any asymmetry that emerges is a property of the data, not of the model.
- It does not require all modalities to be present for any given organism. Modality dropout and the presence vector handle missingness natively.

## Computational footprint

Default configuration (used by tests and demo):
- d_z = 64
- DNA encoder: k=3, d_model=64, n_layers=2, max_len=512
- Fusion: n_heads=4, n_layers=2
- Total: ~1.25M parameters

This runs in seconds on CPU. For the paper's Tier 1 target (~50–100M parameters), scale up d_z to 256, DNA d_model to 256, n_layers to 4, and use pretrained backbones for image and audio.
