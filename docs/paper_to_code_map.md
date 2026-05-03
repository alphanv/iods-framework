# Paper-to-Code Map

This document maps every formula, claim, and procedure in the formalization paper to the source file that implements it. Reviewers and replicators can use this to verify that nothing in the paper is hand-waved in code.

## Formulas

| Formula | Description | File | Function / class |
|---|---|---|---|
| F1 | `M_m(t,e,s) = σ(w_m^T · [t_embed; e_embed; s_embed] + b_m)` | `src/iods/magnitude/magnitude.py` | `MagnitudeFunction.forward` |
| F2 | `φ̃_m = M_m · φ_m` (magnitude weighting) | `src/iods/magnitude/magnitude.py` | `apply_magnitude` |
| F3 | `Z_i(t) = f_enc({φ̃_m}, h, φ_P, B_i, L_i; ψ)` | `src/iods/fusion/cross_attention.py` | `CrossAttentionFusion.forward` |
| F4 | `Z_i^G = GAT(Z_i, G(t))` | *deferred to Phase A2* | — |
| F5 | `Z_i^P = CrossAttention(Z_i^G, E_planet(t))` | *deferred to Phase A2* | — |
| F6 | `Ŷ_i(t+D) = F_θ(Z_i^P, C(t))` | `src/iods/prediction/heads.py` | `ContinuousHead`, `DNAFeatureHead` |
| F7 | `L_total = L_forward + λ_inv·L_inverse + λ_MI·L_MI + λ_reg·L_reg` | `src/iods/prediction/losses.py` | `total_loss` |
| F8 | `L_align = (1/S) Σ_s L_InfoNCE(anchor=s)` | *deferred to Phase A2* | — |
| F9 | `Z_target(epoch) = β·Z_target(epoch-1) + (1-β)·Z_anchor` | *deferred to Phase A2* | — |

## Conditions (C1, C2, C3 — Sect. 2.7)

| Condition | Description | File | Function |
|---|---|---|---|
| C1 | Symmetric cross-modal information: `1/κ ≤ ratio ≤ κ` | `src/iods/validation/conditions.py` | `evaluate_c1` |
| C2 | Phylogenetic surplus: `Acc(IODS) − Acc(phylo) > δ` | `src/iods/validation/conditions.py` | `evaluate_c2` |
| C3 | Context sensitivity: `Acc(Z,L) − Acc(Z) > Δ_context` | `src/iods/validation/conditions.py` | `evaluate_c3` |

## Sections

| Paper section | Topic | Source file |
|---|---|---|
| 2.6 | Magnitude function | `src/iods/magnitude/magnitude.py` |
| 2.7 | C1 / C2 / C3 conditions | `src/iods/validation/conditions.py` |
| 2.8 | Computational pipeline (8 steps) | `src/iods/model.py` |
| 2.8 step 4 | Modality presence vector `h` | `src/iods/fusion/cross_attention.py` (`presence_bias`, `key_padding_mask`) |
| 2.9 | Inverse translation targets (k-mer, marker, Pfam) | `src/iods/prediction/heads.py` (`DNAFeatureHead`) |
| 2.10 | Distance functions (cosine for DNA, etc.) | `src/iods/prediction/losses.py` |
| 3.1 | DNA encoder (k-mer Transformer + attentive pooling) | `src/iods/encoders/dna_encoder.py` |
| 3.1 | Sensor-modality encoders | `src/iods/encoders/sensor_encoders.py` |
| 3.5 | Prediction heads | `src/iods/prediction/heads.py` |
| 3.6 | Heteroskedastic uncertainty (Kendall & Gal 2017) | `src/iods/prediction/heads.py` (`ContinuousHead`), `src/iods/prediction/losses.py` (`heteroskedastic_nll`) |
| 4.1 | Tier-1 data sources | `README.md` data-sources table; ingestion scripts in Phase A2 |
| 4.4 | Phylogenetic null specification | `src/iods/validation/phylo_null.py` |

## Hypotheses tested by code

| Hypothesis | Test | File |
|---|---|---|
| H6 (C1) | InfoNCE symmetry diagnostic + accuracy ratio | `evaluate_c1`, `evaluate_info_nce_symmetry` |
| H7 (C2) | Top-K retrieval against phylogenetic nulls | `evaluate_c2`, `top_k_retrieval` |
| H8 (C3) | Context-ablation comparison | `evaluate_c3` |
| H13 (no-anchor-privilege) | Round-robin alignment | *deferred to Phase A2* |
| H14 (modality-dropout robustness) | Presence-vector handling | `CrossAttentionFusion`, `MagnitudeFunction.renormalize_for_presence` |
| H15 (magnitude-attention separability) | Identifiability diagnostics | *deferred to Phase A3* |

## What is *not* yet implemented (deferred)

The following are scheduled for Phase A2 / A3 in `docs/development_roadmap.md`:

- Graph attention over the symbiosis graph (F4) — the `GAT` layer.
- Planetary context adapter (F5).
- Round-robin alignment with EMA target smoothing (F8, F9) — currently the model trains directly with all losses combined; the no-anchor-privilege guarantee from F8/F9 is not yet empirically verified.
- Pretrained backbones — image and audio encoders are small CNNs in this release; in production they should be replaced with ViT (iNaturalist-pretrained) and BirdNET-style backbones.
- Real-data ingestion pipeline — only synthetic data is supported here; GenBank + iNaturalist + xeno-canto + Open Tree of Life ingestion is the first task of Phase A2.
- Pfam ground-truth targets — the head is implemented but synthetic data has no Pfam labels; real data is required.
- PGLS residual analysis (Sect. 4.4) — implementable in `phylo_null.py` once a real phylogeny is loaded.
- Identifiability diagnostics for Proposition 1 — empirical validation deferred to A3.

## Reproducibility

Every test in `tests/` is deterministic given a fixed random seed. The smoke tests use `torch.manual_seed` and `np.random.default_rng(seed)`. The quickstart demo accepts a `--seed` argument. The pre-registered Tier-1 evaluation (Phase A2) will fix a single commit hash for all reported numbers.
