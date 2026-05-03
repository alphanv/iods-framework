# IODS Framework

**Intra-Organismal Data Symbiosis: Multimodal Biological Translation**

Reference implementation for:

> Vardarlı, A. (2026). "From Theory to Test: Formalizing Life-Specific Data Variables and Bidirectional Translation Protocols for Intra-Organismal Data Symbiosis." *Submitted to Theory in Biosciences.*
>
> Vardarlı, A. (2026). "Life-Specific Data and Intra-Organismal Data Symbiosis: A Biosemiotic Framework for Post-Genomic Biology." *Submitted to Biosemiotics.*

## Overview

IODS formalises the hypothesis that biological modalities — DNA, morphology, vocalisations, behaviour, environment — mutually constrain one another, with each modality's contribution varying by context rather than being permanently privileged. The framework operationalises this hypothesis as a multimodal learning architecture and tests it through the **inverse translation** task: predicting DNA features from phenotypic ensembles, evaluated against phylogenetic null models.

This repository implements:

- **Magnitude function** `M_m(t, e, s)` — sigmoid-gated context-dependent modality weighting (formula F1, Sect. 2.6).
- **Modality-specific encoders** — DNA (Transformer + attentive pooling), image (CNN), audio (mel-spectrogram CNN), time-series (TCN), and MLPs for phenotype, metadata, and environment.
- **Cross-attention fusion** with magnitude weighting and modality-presence indicators (formulas F2, F3, Sect. 2.8).
- **Forward and inverse prediction heads** — heteroskedastic continuous output and three DNA feature targets (k-mer spectra, marker gene embeddings, Pfam domain profiles).
- **Phylogenetic null models** — taxonomic mean, nearest-neighbour-in-tree, and Brownian motion baselines for the C2 phylogenetic-surplus condition.
- **C1 / C2 / C3 evaluators** — symmetric cross-modal information, phylogenetic surplus, and context sensitivity.
- **Synthetic dataset and quickstart demo** — runs end-to-end on CPU in under a minute.

## Status

**Phase A1 (minimum viable proof) — complete.**

This release implements the architectural core of the framework with a runnable end-to-end pipeline on synthetic data. Tier 1 of the empirical roadmap (real GenBank + iNaturalist data, round-robin alignment, GAT for symbiotic context) is the next milestone — see `docs/development_roadmap.md`.

44 unit and integration tests pass; the quickstart demo trains a 1.25M-parameter model and reports a C2 phylogenetic-surplus evaluation in roughly 30 seconds on CPU.

## Installation

```bash
git clone https://github.com/alphanv/iods-framework
cd iods-framework
pip install -e .
```

Requires Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24.

## Quickstart

```bash
python -m iods.demo --epochs 5
```

Expected output: synthetic dataset construction, model build, short training loop, and C2 evaluation against a taxonomic-mean phylogenetic null. The demo confirms the pipeline runs end-to-end. Results on synthetic data are not biological — see `docs/paper_to_code_map.md` for what each component tests.

## Programmatic use

```python
import torch
from iods import IODSModel
from iods.data import make_synthetic_dataset, SyntheticIODSDataset

# Build a model
model = IODSModel(d_z=64)

# Build a tiny dataset
species = make_synthetic_dataset(seed=0)
dataset = SyntheticIODSDataset(species)

# Forward pass on one sample
sample = dataset[0]
batch = {
    "dna_strings": [sample["dna_string"]],
    "image": sample["image"].unsqueeze(0),
    "phenotype": sample["phenotype"].unsqueeze(0),
}
context = {
    "t": torch.tensor([1.0]),
    "e": torch.tensor([0]),
    "s": torch.tensor([0]),
}
out = model(batch, context)
print("integrated latent Z:", out["z"].shape)
print("magnitudes per modality:", out["magnitudes"])
print("inverse k-mer prediction:", out["inverse_pred"]["kmer_logits"].shape)
```

## Repository structure

```
iods-framework/
├── src/iods/
│   ├── magnitude/        # M_m(t,e,s)  [formula F1, Sect. 2.6]
│   ├── encoders/         # DNA, image, audio, TS, phenotype, metadata, env
│   ├── fusion/           # cross-attention + presence indicator [F2, F3]
│   ├── prediction/       # forward + inverse heads, losses
│   ├── validation/       # phylogenetic nulls + C1/C2/C3 evaluators
│   ├── data/             # synthetic dataset for smoke tests
│   ├── model.py          # IODSModel: full pipeline integration
│   └── demo.py           # quickstart demo (run via `python -m iods.demo`)
├── tests/                # 44 unit + integration tests
├── docs/
│   ├── paper_to_code_map.md       # paper section -> source file mapping
│   ├── development_roadmap.md     # Phase A1 / A2 / A3 plan
│   └── architecture.md            # high-level architecture overview
├── configs/                       # placeholder for Tier 1/2/3 configs
├── data/examples/                 # data-download instructions (no data)
├── notebooks/                     # placeholder for analysis notebooks
└── figures/                       # placeholder for paper figure regeneration
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 44 tests should pass in under 10 seconds on CPU.

## Paper-to-code mapping

Every component traces to a section of the formalisation paper. See `docs/paper_to_code_map.md` for the complete mapping. A brief summary:

| Paper section / formula | Source file |
| --- | --- |
| F1: M_m(t,e,s) sigmoid gate | `src/iods/magnitude/magnitude.py` |
| F2: φ̃_m = M_m · φ_m | `src/iods/magnitude/magnitude.py::apply_magnitude` |
| F3: cross-attention fusion | `src/iods/fusion/cross_attention.py` |
| Sect. 2.8 step 4: presence vector h | `src/iods/fusion/cross_attention.py` |
| Sect. 3.1: DNA encoder | `src/iods/encoders/dna_encoder.py` |
| Sect. 3.5: prediction heads | `src/iods/prediction/heads.py` |
| Sect. 2.10: distance functions | `src/iods/prediction/losses.py` |
| Sect. 4.4: phylogenetic null models | `src/iods/validation/phylo_null.py` |
| Sect. 2.7: C1, C2, C3 conditions | `src/iods/validation/conditions.py` |

## Data sources (for Tier 1 — not yet ingested)

All real data drawn from public databases. No new collection.

| Database | Modality | URL |
| --- | --- | --- |
| GenBank / NCBI | Genomes | https://www.ncbi.nlm.nih.gov/genbank/ |
| iNaturalist / GBIF | Images | https://www.inaturalist.org / https://www.gbif.org |
| xeno-canto | Vocalisations | https://xeno-canto.org |
| Macaulay Library | Sounds | https://www.macaulaylibrary.org |
| Movebank | Movement | https://www.movebank.org |
| Web of Life / GLOBI | Interactions | https://www.web-of-life.es / https://www.globalbioticinteractions.org |
| Open Tree of Life | Phylogenies | https://opentreeoflife.github.io |

## Development roadmap

- **Phase A1 — minimum viable proof. ✅ Complete.** Architecture, magnitude function, encoders, fusion, prediction heads, phylogenetic nulls, C1/C2/C3 evaluators, synthetic-data smoke test.
- **Phase A2 — Tier 1 real data.** Bird-species pilot (100 species, GenBank + iNaturalist + xeno-canto + Open Tree of Life), full round-robin alignment, GAT for symbiotic context, planetary context adapter, pre-registration of primary outcome.
- **Phase A3 — revision-readiness.** Stronger phylogenetic null (nearest-neighbour-in-tree as default), reframed Proposition 1, tightened C1 thresholds, regenerable figures, Zenodo archive.

See `docs/development_roadmap.md` for the full plan.

## Citation

```bibtex
@article{vardarli2026iods,
  title={From Theory to Test: Formalizing Life-Specific Data Variables and
         Bidirectional Translation Protocols for Intra-Organismal Data Symbiosis},
  author={Vardarl{\i}, Alphan},
  journal={Theory in Biosciences},
  year={2026},
  note={Submitted}
}

@article{vardarli2026biosemiotics,
  title={Life-Specific Data and Intra-Organismal Data Symbiosis:
         A Biosemiotic Framework for Post-Genomic Biology},
  author={Vardarl{\i}, Alphan},
  journal={Biosemiotics},
  year={2026},
  note={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Author

Alphan Vardarlı — Independent Researcher, Çanakkale, Türkiye.
ORCID: [0009-0007-1581-2190](https://orcid.org/0009-0007-1581-2190)
