# Data Download Instructions

All data is drawn from freely available biological databases. No new collection required.

## Tier 1 (Minimum)

1. **Genomes**: Download marker gene sequences from GenBank
   ```bash
   # Example: COI sequences for birds
   esearch -db nucleotide -query "COI[gene] AND Aves[organism]" | efetch -format fasta > data/genbank/aves_coi.fasta
   ```

2. **Images**: Download from iNaturalist export or GBIF
   - Use iNaturalist API: https://api.inaturalist.org/v1/
   - Filter for research-grade observations
   - Match to NCBI taxonomy via species name

## Taxonomy Reconciliation

Use NCBI Taxonomy as reference. Map iNaturalist taxon IDs → NCBI taxon IDs.

```python
from Bio import Entrez
Entrez.email = "your@email.com"
handle = Entrez.esearch(db="taxonomy", term="Parus major")
```
