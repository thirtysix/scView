# Validation

What scView's analysis methods were checked against, how, and what the checks found. Each page covers
one method or one family of methods, so the pages can later be combined into a single methods document.

## Pages

| Page | Question | Decision it supports |
|---|---|---|
| [Pseudobulk DE](pseudobulk-de.md) | Which pseudobulk model holds its false discovery rate on real donor data, and how to build the replicates | limma-voom is the default (v0.61.0); DESeq2 carries an FDR caution below 6 donors per side |

## Template for a page

1. **Question and decision**: what was tested, and what scView does because of it.
2. **Data**: which datasets, how many cells, donors and cell types, and why those.
3. **Design**: null and truth construction, what counts as a hit, the score for each fit.
4. **Methods compared**: the scView code and the reference implementations with versions.
5. **Results**: the main tables first. Any number quoted elsewhere (README, CHANGELOG, site) comes from here.
6. **Port equivalence**: how closely scView's implementation matches the reference.
7. **Limits**: what the design cannot show.
8. **Reproducing**: harness versions, inputs, where the outputs live.

## Equivalence checks still to write up

The code already records these checks against reference tools (see the test suite and CHANGELOG). Each
needs a page in this format:

- Activity scores: ULM = decoupler; AUCell = decoupler 2.2
- Composition: propeller-style test with robust eBayes = R limma
- Ligand-receptor: LIANA consensus rank aggregate = liana-py 1.10
- Milo neighbourhood DA = pertpy's pydeseq2 solver
- WNN = Seurat FindMultiModalNeighbors
- RNA velocity: steady-state and stochastic = scVelo 0.3.3
- ATAC: LSI, LinkPeaks, TSS enrichment, nucleosome signal = Signac; motif hits = motifmatchr; chromVAR
- Spatial: Moran's I and co-occurrence = squidpy 1.6.5
- QC: Scrublet (bitwise = scanpy), barcode ranks = DropletUtils, HTODemux and hashedDrops, SoupX autoEstCont rho
