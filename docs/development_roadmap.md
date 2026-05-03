# Development Roadmap

This document specifies the planned development of the IODS framework across three phases. Phase A1 is complete; A2 and A3 are scheduled.

## Phase A1 — Minimum viable proof ✅ Complete

**Goal:** every component clickable from the README runs end-to-end on synthetic data without errors.

**Delivered:**
- Magnitude function `M_m(t,e,s)` with sigmoid and softmax variants — formula F1.
- Modality-specific encoders: DNA (k-mer Transformer + attentive pooling, real implementation), image (CNN), audio (mel-spectrogram CNN), time-series (TCN), phenotype/metadata/environment (MLPs).
- Cross-attention fusion with magnitude weighting and modality-presence indicators — formulas F2, F3.
- Forward (heteroskedastic continuous) and inverse (k-mer / marker / Pfam) prediction heads.
- Loss functions: heteroskedastic NLL, KL on k-mer distributions, cosine distance, InfoNCE.
- Phylogenetic null models: TaxonomicMean, NearestNeighbour, BrownianMotion.
- C1 / C2 / C3 evaluators.
- Synthetic dataset for end-to-end testing.
- 44 unit and integration tests, all passing.
- Quickstart demo runnable as `python -m iods.demo`.

## Phase A2 — Tier 1 real data (next milestone)

**Goal:** the C1 / C2 / C3 conditions evaluated on real biological data with phylogenetic controls. Bird-species pilot.

### A2.1 — Real-data ingestion pipeline (`src/iods/data/ingestion/`)

Build a pipeline that, given a list of NCBI taxon IDs, downloads and stores:
- GenBank genome assemblies (or RefSeq when available).
- iNaturalist research-grade observation images via GBIF download API.
- xeno-canto recordings.
- Open Tree of Life subtree (Newick).
- GLOBI ecological interactions.

Output: a single HDF5 store with one record per species containing each modality and a presence indicator.

**Scope:** 100 well-studied bird species (high coverage across all three modalities + phylogeny + interactions). This is a meaningful subset that supports Tier 1 of the paper's empirical roadmap.

### A2.2 — Round-robin alignment (`src/iods/fusion/round_robin.py`)

Implement formulas F8 and F9:
- Anchor-rotation across modalities in successive epoch blocks.
- EMA smoothing of the latent target with β ∈ [0.9, 0.99].
- Convergence diagnostic: coefficient of variation of anchor-specific losses across a 10-epoch window.

**Diagnostic test:** train parallel models with fixed single-modality anchors (DNA-only, image-only) and compare representations via Centred Kernel Alignment (CKA). Round-robin should produce representations that are equally close to all single-anchor variants (no privileged modality).

### A2.3 — Graph attention for symbiotic context (`src/iods/fusion/gat.py`)

Implement formula F4 — a 2-layer GAT over the symbiosis graph derived from GLOBI / Web of Life. Edges weighted by `S_{ij}(t)`. Test: ablating partner-species data should reduce prediction accuracy by at least 3% on tasks where partner data is informative.

### A2.4 — Planetary context adapter (`src/iods/fusion/planetary.py`)

Implement formula F5 — cross-attention conditioning on `E_planet(t)`. For Tier 1 birds, planetary context is approximated by season + latitude + climate-zone categorical features.

### A2.5 — Validation suite expansion (`src/iods/validation/`)

- `evaluate_pgls.py` — phylogenetic generalized least squares for residual signal testing (Sect. 4.4).
- `evaluate_magnitude_correlation.py` — does the learned `M_m(t,e,s)` correlate with the predicted stability gradient? Direct test of the magnitude function's interpretability claim. Failure of this test is a falsification condition listed in Sect. 4.7 of the paper.
- Convergence diagnostic for round-robin (CV of anchor losses).

### A2.6 — Pre-registration document (`docs/preregistration.md`)

Pre-register the primary outcome before running the full evaluation:
- Primary outcome: phenotype → DNA inverse retrieval at top-5, against family-level held-out test set.
- α = 0.05.
- Sample size: full bird pilot (100 species).
- Analysis plan: frozen at a specific commit hash; no post-hoc modifications.
- Secondary outcomes: C1 InfoNCE symmetry, C3 context improvement, magnitude–stability correlation.

## Phase A3 — Revision-readiness

**Goal:** when *Theory in Biosciences* returns the inevitable major-revision request, all reviewer concerns are pre-addressed.

### A3.1 — Stronger phylogenetic null (Reviewer 1's main concern)

Make NearestNeighbourNull the default phylogenetic null for k-mer-spectrum comparisons. Brownian-motion-on-compositional-features is too weak; nearest-neighbour-in-tree at family level is the credible baseline.

### A3.2 — Identifiability diagnostics for Proposition 1

Empirical validation of the magnitude–attention separability claim. Notebook in `notebooks/identifiability_diagnostics.ipynb`:
- Ablation against uniform magnitudes (`M_m = 1` for all m) — does fixing magnitudes hurt prediction?
- Comparison with domain-knowledge-fixed magnitudes — does using the predicted stability gradient as fixed magnitudes match learned performance?
- Consistency across random seeds — do learned magnitudes correlate across runs?
- Correlation between learned magnitudes and the predicted stability gradient.

### A3.3 — Tightened C1 thresholds

Default κ = 1.7 (down from 2.5). Report κ ∈ {1.5, 2.0, 2.5} as sensitivity. InfoNCE asymmetry as the primary criterion; accuracy ratio secondary.

### A3.4 — Regenerable figures

`figures/` directory with one notebook per paper figure. Every figure regenerable from a single commit hash.

### A3.5 — Zenodo archive

Tag v0.1.0 release. Connect repository to Zenodo for DOI assignment. Update `README.md` with the DOI badge.

## Out of scope for this repository

These belong in the companion repository `episteme-spacecraft`:

- Earth-as-1001st-species (Experiment 2 of the original Episteme Spacecraft project).
- DNA-as-operating-system claim and Earth simulation.
- Symbiotic circles as primary dataset unit (rather than auxiliary GAT input).
- Calibrated Earth-DNA latent vector and forward simulation of Earth modalities.

The two-repository structure protects the publishable framework from the more speculative claims while keeping both tracks alive under the same author.
