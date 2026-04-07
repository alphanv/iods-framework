# Architecture-to-Paper Mapping

| Source file | Paper section | Component |
|-------------|--------------|-----------|
| `src/magnitude/magnitude.py` | Section 2.6 | M_m(t,e,s) |
| `src/encoders/dna.py` | Section 3.1 | DNAEncoder |
| `src/encoders/image.py` | Section 3.1 | ImageEncoder |
| `src/encoders/audio.py` | Section 3.1 | AudioEncoder |
| `src/fusion/cross_attention.py` | Section 3.2 | CrossAttentionFusion |
| `src/fusion/round_robin.py` | Section 3.2 | RoundRobinTrainer |
| `src/prediction/inverse.py` | Section 3.5 | InverseHead (k-mer, marker, Pfam) |
| `src/prediction/uncertainty.py` | Section 3.6 | HeteroskedasticHead |
| `src/utils/losses.py` | Section 2.10 | Distance functions |
| `src/utils/metrics.py` | Section 4.6 | C1, C2, C3 evaluation |
| `src/utils/phylo.py` | Section 4.4 | Phylogenetic null models |
| `tests/test_magnitude.py` | Section 2.6 | Identifiability diagnostics |
| `tests/test_conditions.py` | Section 4.7 | Falsification conditions |
