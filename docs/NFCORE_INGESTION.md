# Ingesting nf-core/scrnaseq output into scView

[nf-core/scrnaseq](https://github.com/nf-core/scrnaseq) is a Nextflow pipeline that
processes **raw FASTQs → count matrices** using one of several aligners (STARsolo,
alevin-fry/simpleaf, kallisto|bustools, CellRanger). scView's own pipeline starts from a
**count matrix**, so the two are *complementary stages*: run nf-core to get counts, then
explore them in scView.

## Recommended ingestion path: the `mtx_conversions/` h5ad files

Whichever aligner you choose, nf-core writes aligner-agnostic converted matrices to:

```
results/<aligner>/mtx_conversions/
    <sample>_raw_matrix.h5ad
    <sample>_filtered_matrix.h5ad
    <sample>_cellbender_filter_matrix.h5ad   # if cellbender ran
    combined_matrix.h5ad                      # all samples concatenated
    <sample>_*_matrix.rds                      # Seurat objects
```

These `.h5ad` files are **standard AnnData** and load directly through scView's Data tab —
no conversion needed. Drop them into the importer (or the `/ingest` wizard):

- **One sample** → drop `<sample>_filtered_matrix.h5ad`. scView labels the dataset with the
  **sample name** (the `_{raw,filtered,cellbender_filter}_matrix` suffix is stripped
  automatically; `combined_matrix.h5ad` → `combined`).
- **All samples at once** → drop `combined_matrix.h5ad` (already concatenated by nf-core),
  **or** drop several per-sample `*_matrix.h5ad` files together to let scView's merge flow
  combine them (it flags the multi-unit bundle as a merge).
- **Seurat** `*_matrix.rds` → routed to scView's R converter.
- **CellRanger** `filtered_feature_bc_matrix.h5` → loads via scView's 10x HDF5 reader.

After ingest, open **Data Assessment** to see the QC distributions and run
normalization / clustering / etc. (nf-core does not normalize or cluster — it stops at counts).

## What scView handles today

| nf-core output | scView ingestion |
|---|---|
| `*_matrix.h5ad`, `combined_matrix.h5ad` | ✅ native AnnData loader (sample-name labelling) |
| CellRanger `*.h5` | ✅ 10x HDF5 loader |
| 10x MEX triplet (matrix/barcodes/features) | ✅ MEX loader |
| `*_matrix.rds` (Seurat) | ✅ via R converter |
| raw STARsolo / alevin / kallisto dirs | ⚠ prefer the `mtx_conversions/` h5ad instead |

## Deferred: running nf-core *from* scView (heavy path)

Driving the Nextflow run from inside scView (submit FASTQs → alignment → counts) is **not**
implemented and is out of scope for the in-app pipeline: FASTQ alignment is HPC-scale and
needs Nextflow + a container/conda engine, not a Docker-compose service. The natural home for
that is the `csc-puhti-job` workflow (submit on HPC, then ingest the resulting
`mtx_conversions/*.h5ad` back into scView). Revisit if there's demand.
