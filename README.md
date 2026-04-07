# IODS Framework

**Intra-Organismal Data Symbiosis: Multimodal Biological Translation**

Reference implementation for:

> Vardarlı, A. (2026). "From Theory to Test: Formalizing Life-Specific Data Variables and Bidirectional Translation Protocols for Intra-Organismal Data Symbiosis." *Theory in Biosciences*.

## Overview

IODS formalizes the hypothesis that biological modalities (DNA, morphology, vocalizations, behavior, etc.) mutually constrain one another. The framework builds a guessing algorithm that learns cross-modal relationships from freely available biological databases, compares its guesses against actual data, and iteratively improves.

### Core Components

- **Magnitude function** `M_m(t,e,s)`: Context-dependent modality weighting (sigmoid-gated)
- **Modality-specific encoders**: DNA (transformer), image (ViT), audio (mel-spectrogram CNN), time-series (TCN), metadata (MLP)
- **Round-robin alignment**: Anchor-rotating contrastive alignment with EMA smoothing
- **Cross-attention fusion**: Magnitude-weighted multimodal fusion with modality presence indicators
- **Graph attention network**: Symbiotic context from ecological interaction databases
- **Forward & inverse translation**: Bidirectional genotype↔phenotype prediction
- **Validation suite**: Phylogenetic null models, C1–C3 condition testing, PGLS residual analysis

## Architecture

```
Input modalities (whatever is available per species)
    │
    ├── DNAEncoder ──────────┐
    ├── ImageEncoder ─────────┤
    ├── AudioEncoder ─────────┤──→ Magnitude weighting ──→ Cross-attention fusion ──→ Z_i(t)
    ├── TSEncoder ────────────┤         M_m(t,e,s)              + presence vector h
    └── MetaEncoder ──────────┘
                                                                      │
                                                    ┌─────────────────┼─────────────────┐
                                                    │                 │                 │
                                               GAT context    Forward heads    Inverse head
                                             (symbiosis graph)  (phenotype)    (DNA features:
                                                                                k-mer spectra,
                                                                                marker genes,
                                                                                Pfam domains)
```

## Installation

```bash
git clone [INSERT_GITHUB_URL_HERE]
cd iods-framework
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torch-geometric >= 2.3
- transformers >= 4.30
- timm >= 0.9

See `requirements.txt` for full list.

## Quick Start

```python
from iods import IODSModel, RoundRobinTrainer
from iods.data import SpeciesDataset

# Load from public databases — missing modalities handled natively
dataset = SpeciesDataset(
    genbank_dir="data/genbank/",
    images_dir="data/inaturalist/",
    audio_dir="data/xenocanto/",
    taxonomy="data/ncbi_taxonomy.tsv"
)

model = IODSModel(
    dna_dim=512, img_dim=512, audio_dim=256,
    latent_dim=512, n_modalities=3, context_dim=32
)

trainer = RoundRobinTrainer(model=model, ema_decay=0.99, modality_dropout=0.2)
trainer.fit(dataset, phases=[1, 2, 3])

# Evaluate C1-C3 conditions
from iods.validation import evaluate_conditions
results = evaluate_conditions(model, dataset.test_split, phylo_tree="data/opentree.nwk")
```

## Validation: Testing C1–C3

```python
from iods.validation import PhylogeneticNull, evaluate_c1, evaluate_c2, evaluate_c3

c1 = evaluate_c1(model, test, kappa_values=[1.5, 2.0, 2.5, 3.0])
c2 = evaluate_c2(model, test, null=PhylogeneticNull(tree, models=["BM","OU","EB"]))
c3 = evaluate_c3(model, test, context_ablation=True)
```

## Magnitude Diagnostics

```python
from iods.diagnostics import magnitude_diagnostics

diag = magnitude_diagnostics(model, test, n_seeds=5,
    stability_gradient={"dna": {"generational": 1.0, "acute": 0.2},
                        "metabolome": {"generational": 0.2, "acute": 1.0}})
```

## Data Sources

All data from freely available databases. No new collection required.

| Database | Modality | URL |
|----------|----------|-----|
| GenBank/NCBI | Genomes | https://www.ncbi.nlm.nih.gov/genbank/ |
| iNaturalist/GBIF | Images | https://www.inaturalist.org / https://www.gbif.org |
| xeno-canto | Vocalizations | https://xeno-canto.org |
| Macaulay Library | Sounds | https://www.macaulaylibrary.org |
| Movebank | Movement | https://www.movebank.org |
| Web of Life / GLOBI | Interactions | https://www.web-of-life.es / https://www.globalbioticinteractions.org |
| Open Tree of Life | Phylogenies | https://opentreeoflife.github.io |

## Repository Structure

```
iods-framework/
├── src/
│   ├── model.py                 # IODSModel: full pipeline
│   ├── encoders/                # Modality-specific encoders
│   ├── fusion/                  # Cross-attention + round-robin alignment
│   ├── magnitude/               # M_m(t,e,s) magnitude function
│   ├── prediction/              # Forward/inverse heads + uncertainty
│   └── utils/                   # Losses, phylo nulls, metrics
├── configs/                     # Tier 1/2/3 configurations
├── tests/                       # Identifiability, convergence, C1-C3 tests
├── data/examples/               # Data download instructions
├── figures/                     # Figure generation for paper
└── docs/                        # Architecture-to-IODS mapping
```

## Citation

```bibtex
@article{vardarli2026iods,
  title={From Theory to Test: Formalizing Life-Specific Data Variables
         and Bidirectional Translation Protocols for Intra-Organismal
         Data Symbiosis},
  author={Vardarl{\i}, Alphan},
  journal={Theory in Biosciences},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
