# Pseudobulk differential expression: model choice and replicate rules

Measured November 2026 on 18 public CELLxGENE datasets. Applies to scView 0.61.0 and later.

## Question and decision

Pseudobulk DE sums each donor's counts per condition and tests the sums, so the replicate is the donor and
not the cell. Two choices are left: which count model tests the sums, and how the pseudobulk samples are
built (minimum cells per sample, weights, which unit counts as the replicate).

**Decision.** limma-voom is the default model. DESeq2 and voomByGroup stay available. A DESeq2 run with
fewer than 6 donors on its smaller side carries a caution that its adjusted p-values are optimistic.
Replicates are donors, samples with fewer than 10 cells are dropped and listed, samples are never weighted
by cell count, and a donor's cells are never split into several pseudo-replicates. A side with fewer than
3 donors gets a warning.

**Why.** When the true answer is known, DESeq2 called about 1 false gene in 3 at a nominal FDR of 5% with
2 donors per side, and 1 in 6 at 4 donors. limma-voom and edgeR's quasi-likelihood test (v4) called 1 in 7
at 2 donors and stayed near 1 in 20 from 3 donors on. All three rank genes about equally well (AUC
0.89-0.90), so DESeq2's higher sensitivity comes from a looser effective threshold, not from better
ranking. The pattern held in every one of the 18 datasets.

## Data

18 datasets from the CELLxGENE Census, chosen to span size, assay, tissue and design. Two more pilot
datasets with only 3 donors were left out. Cell types: the author's `cell_type`, those with at least 200
cells and 4 donors, the 12 largest per dataset, giving 82 cell types. Within each dataset one condition was
used (the value with the most donors), so the two sides of every comparison come from one biological group.

| Dataset | DOI | Assay | Tissue | Cells | Donors | Cell types | ID |
|---|---|---|---|---:|---:|---:|---|
| Supercluster: MGE-derived interneurons | 10.1126/science.adf6812 | 10x 3' v3 | cortex (8 regions) | 185,477 | 6 | 3 | `0b8a9eb3` |
| NSCLC epithelial cells | 10.1016/j.ccell.2021.09.008 | 10x 3' v2/v3 | lung and metastases | 9,778 | 42 | 7 | `25f2c5ad` |
| Infiltrating neoplastic cells, human glioblastoma | 10.1016/j.celrep.2017.10.030 | Smart-seq2 | cerebral cortex | 3,589 | 4 | 3 | `2e357b12` |
| Placenta infection | 10.1016/j.cels.2024.04.002 | 10x 3' v3 | placenta | 158,978 | 20 | 7 | `38b64997` |
| Kidney cortex, donors with and without diabetes | 10.1038/s41467-022-32972-z | 10x 3' v3, 5' v1 | kidney cortex | 39,176 | 11 | 8 | `473d2e1c` |
| Cardiac interventricular septum survey | 10.1111/cpr.13557 | modified STRT-seq | interventricular septum | 1,971 | 9 | 1 | `47a98d37` |
| Tangle-bearing neurons in Alzheimer's disease (excitatory) | 10.1016/j.neuron.2022.06.021 | 10x 3' v2/v3 | prefrontal cortex | 96,129 | 16 | 1 | `5015051f` |
| Time-resolved systems immunology, COVID-19 | 10.1016/j.cell.2021.02.018 | 10x 5' v1 | blood | 125,117 | 46 | 8 | `60cc18b6` |
| Choroid plexus single-nucleus, COVID-19 | 10.1038/s41586-021-03710-0 | 10x 3' v3 | choroid plexus | 27,092 | 14 | 5 | `6dd04cc1` |
| Selectively vulnerable neurons in Alzheimer's disease | 10.1038/s41593-020-00764-7 | 10x 3' v2 | entorhinal cortex | 5,572 | 10 | 1 | `8580d236` |
| Oligodendrocytes in multiple sclerosis | 10.1038/s41586-019-0903-2 | 10x 3' v2 | brain white matter | 17,799 | 9 | 3 | `8bbea7c5` |
| Peripheral immune response to SARS-CoV-2 | 10.1038/s41591-020-0944-y | Seq-Well | blood | 44,721 | 13 | 8 | `8d7fad46` |
| Tangle-bearing neurons in Alzheimer's disease (inhibitory) | 10.1016/j.neuron.2022.06.021 | 10x 3' v2/v3 | prefrontal cortex | 23,197 | 16 | 1 | `9b87c411` |
| Colorectal liver metastasis atlas | 10.1186/s12943-025-02430-7 | BD Rhapsody WTA | liver | 75,104 | 6 | 7 | `a3bcb68b` |
| Alpha cells of pancreatic islets | 10.1038/s44318-026-00744-w | 10x 3' v2/v3 | islet of Langerhans | 74,812 | 48 | 1 | `bf1557d3` |
| Medial orbital frontal cortex single-nucleus, COVID-19 | 10.1038/s41586-021-03710-0 | 10x 3' v3 | medial orbital frontal cortex | 38,217 | 16 | 8 | `c4ee3adc` |
| ASPC-to-adipocyte differentiation, visceral fat | 10.1038/s41588-024-02048-3 | 10x 3' v3 | visceral fat | 29,078 | 10 | 2 | `d16c862c` |
| Mouse and human prostate fibroblasts | 10.1002/path.5751 | 10x 3' v2/v3 | prostate | 83,451 | 9 | 8 | `e6b9fd9a` |

ID = the first 8 characters of the CELLxGENE dataset version id.

## Design

- **Null.** Random splits of one condition's donors into k vs k, with k = 2, 3, 4, 5, 8, 12 and half the
  donors; 50 splits per k (all of them when fewer exist). No gene truly differs by design, so the share of
  genes at p < 0.05 should be 0.05.
- **Truth.** The same splits with known effects added by binomial thinning of the pseudobulk counts (the
  approach of seqgendiff, Gerard 2020): 5% of expressed genes, |log2 fold change| uniform on 0.5-2.5, half
  up and half down. The effects ride on real donor-to-donor variation, which simulated counts lack.
- **Gene filter**, applied after spiking and identical for every method: scView's own, at least 10 counts in
  at least as many samples as the smaller side has.
- **Scores per fit.** Null: share of genes at p < 0.05 and < 0.01, and whether any gene passes BH FDR 0.05.
  Truth: true false discovery proportion (FDP) and true positive rate (TPR) at BH 0.05 and 0.10, AUC of
  spiked vs unspiked genes, and log2 fold change error. Seconds per fit, on one CPU.
- **Scale.** Round 2: 120,300 fits of 18 methods. Head-to-head rerun (scView's own engines next to R, same
  inputs and node per dataset): 46,008 fits.

## Methods compared

- **scView**: limma-voom (TMM, voom, eBayes), voomByGroup, and variants with robust eBayes or sample
  weights; DESeq2 through pydeseq2 0.5.4 (the default up to 0.60) and scView's vectorised DESeq2 Wald test;
  a clean-room edgeR v4 quasi-likelihood pipeline.
- **R (Bioconductor 3.21)**: DESeq2 1.48.2 (default, and with glmGamPoi 1.20.0), edgeR 4.6.3 (QL v4, QL
  legacy, QL robust, LRT), limma 3.64.3 (voom, voomWithQualityWeights, limma-trend).

## Results

### True FDP at a nominal FDR of 5%, by donors per side

| Method | 2 | 3 | 4 | 6 | 8+ |
|---|---:|---:|---:|---:|---:|
| limma-voom (scView) | 0.13 | 0.07 | 0.06 | 0.06 | 0.05 |
| edgeR v4 QL (R; scView's port identical) | 0.15 | 0.07 | 0.07 | 0.06 | 0.05 |
| limma-trend (R) | 0.16 | 0.05 | 0.04 | 0.03 | 0.03 |
| DESeq2 (R; pydeseq2 and scView's vectorised build within 0.01) | 0.34 | 0.20 | 0.16 | 0.11 | 0.10 |
| edgeR LRT (R) | 0.38 | | | | 0.10 |
| DESeq2 + glmGamPoi (R) | 0.51 | | | | 0.12 |

### Overall, all splits

scView's engines and R DESeq2 / edgeR v4 QL from the head-to-head run; the other rows from round 2 (pydeseq2 on a
subset of 2,069 of the 3,834 splits).

| Method | Null p < .05 | Null splits with any FDR hit | True FDP @ .05 | TPR @ .05 | AUC | s / fit |
|---|---:|---:|---:|---:|---:|---:|
| limma-voom (scView) | 0.056 | 0.15 | 0.091 | 0.249 | 0.894 | 0.03 |
| limma-voom (R) | 0.056 | 0.15 | 0.092 | 0.248 | 0.895 | 0.36 |
| edgeR v4 QL (R) | 0.051 | 0.15 | 0.097 | 0.243 | 0.901 | 0.20 |
| edgeR v4 QL (scView port) | 0.051 | 0.15 | 0.097 | 0.243 | 0.901 | 0.07 |
| edgeR v4 QL robust (R) | 0.051 | 0.17 | 0.102 | 0.248 | 0.901 | 0.22 |
| limma-trend (R) | 0.046 | 0.15 | 0.089 | 0.207 | 0.877 | 0.05 |
| voomWithQualityWeights (R) | 0.070 | 0.22 | 0.130 | 0.293 | 0.894 | 1.38 |
| DESeq2 (R) | 0.040 | 0.70 | 0.235 | 0.342 | 0.895 | 2.16 |
| DESeq2 (pydeseq2) | 0.040 | 0.70 | 0.238 | 0.343 | 0.894 | 11.9 |
| DESeq2 (scView vectorised) | 0.038 | 0.69 | 0.232 | 0.339 | 0.895 | 0.15 |

DESeq2 is conservative on the share at p < 0.05 and yet passes FDR in 70% of null splits: its p-value
distribution has a heavy lower tail. The genes in that tail are real donor-level differences between two
random groups of people (XIST and Y-chromosome genes, mitochondrial, interferon and stress genes,
immunoglobulins), which limma-voom and edgeR QL weigh against the between-donor variance and DESeq2 does not.

### Power with uneven cells per sample

Power arm: truth from the full data, then cells subsampled; 5 datasets, 30 cell types. TPR / share of hits
outside the truth:

| Scenario | limma-voom, ≥ 10 cells | DESeq2, ≥ 10 cells | limma-voom weighted by cells |
|---|---|---|---|
| All cells | 0.37 / 0.20 | 0.32 / 0.36 | 0.48 / 0.42 |
| Uneven cells (3-300 per sample) | 0.18 / 0.18 | 0.14 / 0.28 | 0.25 / 0.34 |
| 50 cells per sample | 0.23 / 0.17 | 0.19 / 0.32 | 0.25 / 0.20 |
| 25 cells per sample | 0.15 / 0.16 | 0.12 / 0.29 | 0.15 / 0.17 |

## Replicate rules

Null share of genes at p < 0.05 with limma-voom (round 1: 18 datasets, 106 cell types, 227,000 fits):

| Rule | Null p < .05 |
|---|---:|
| Donor as unit, any minimum cells (1, 5, 10, 20, 50) | 0.054-0.055 |
| Weights by cell count (dreamlet's default idea) | 0.13-0.15 |
| Weights by √cells | 0.08 |
| Weights n / (n + 20) | 0.06-0.075 |
| Samples as units when samples repeat donors | 0.10 |
| A donor's cells split into 2 or 3 pseudo-replicates | 0.15-0.22 |

- Dropping samples under 10 cells costs nothing in calibration and gains power with uneven cells (TPR 0.28
  vs 0.18 keeping every sample, Kang and Wilk). Above 20 cells the rule starts to lose donors.
- Weights raise TPR only together with hits outside the truth: the gain is the null inflation above.

## Port equivalence

- **limma-voom**: scView matches R limma-voom (null 0.056 vs 0.056, FDP 0.091 vs 0.092, AUC 0.894 vs 0.895).
- **edgeR v4 QL** (clean-room, built from edgeR's documentation and outputs only): median |Δlog10 p| vs R
  3 × 10⁻⁵ (99th percentile 4 × 10⁻⁴); calls at p < 0.05 agree on 99.997% of tests; all summary scores equal
  R to three decimals; 3x faster than R.
- **Vectorised DESeq2**: median |Δlog10 p| vs R 0.0002 (pydeseq2 0.0028); 14x faster than R and about 80x
  faster than pydeseq2.

## Limits

- "False" means a gene that was not spiked. Genes that truly differ between two random groups of donors
  also count as false, so every method's FDP is an upper bound. That inflates all methods alike, so the
  ranking holds even though the absolute levels are high.
- Only two-group unpaired comparisons within one condition were calibrated. Paired designs and designs with
  covariates were not.
- The power arm is smaller (5 datasets, 30 cell types) than the calibration arm.
- The spike-in effects are multiplicative on the counts; effects that change variance or detection are not
  modelled.

## Reproducing

Harness: bundle builder (donor sums and tasks per dataset x cell type, HDF5), Python and R fitters on the same
bundles, scorer and summariser. Fits ran on CSC Roihu, one CPU per fit, Apptainer images of scView 0.60.x and
Bioconductor 3.21. The harness and the raw per-fit tables are not yet in this repository.
