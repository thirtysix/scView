# scView API Reference

**Base URL**: `http://localhost:8000/api/v1`

All endpoints are prefixed with `/api/v1`. The frontend Nginx proxy transparently routes requests, so browser calls to `/api/v1/...` reach the backend.

---

## Datasets

### Upload a Dataset

```
POST /datasets/upload
Content-Type: multipart/form-data
```

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | Yes | `.h5ad`, `.rds`, or `.rdata` file |

**Response** (201):
```json
{
  "dataset_id": "abc123",
  "filename": "pbmc3k.h5ad",
  "status": "ready",
  "message": "Dataset uploaded and loaded successfully."
}
```

For `.rds` files, the response may return `"status": "converting"` while the R converter processes the file.

### List Datasets

```
GET /datasets
```

**Response** (200):
```json
[
  {
    "dataset_id": "abc123",
    "filename": "pbmc3k.h5ad",
    "status": "ready",
    "n_cells": 2700,
    "n_genes": 13714,
    "embeddings": ["X_umap", "X_pca"],
    "created_at": "2026-02-12T10:30:00"
  }
]
```

### Get Dataset Info

```
GET /datasets/{dataset_id}
```

**Response** (200):
```json
{
  "dataset_id": "abc123",
  "filename": "pbmc3k.h5ad",
  "status": "ready",
  "n_cells": 2700,
  "n_genes": 13714,
  "embeddings": [
    {"name": "X_umap", "dimensions": 2},
    {"name": "X_pca", "dimensions": 50}
  ],
  "obs_columns": [
    {"name": "leiden", "dtype": "category", "n_unique": 8},
    {"name": "n_genes", "dtype": "int64", "n_unique": 1352}
  ]
}
```

### Delete a Dataset

```
DELETE /datasets/{dataset_id}
```

**Response** (200):
```json
{"message": "Dataset deleted."}
```

---

## Embeddings

### List Available Embeddings

```
GET /datasets/{dataset_id}/embeddings
```

**Response** (200):
```json
{
  "embeddings": [
    {"name": "X_umap", "dimensions": 2},
    {"name": "X_tsne", "dimensions": 2},
    {"name": "X_pca", "dimensions": 50}
  ]
}
```

### Get Embedding Coordinates

```
GET /datasets/{dataset_id}/embeddings/{embedding_name}
```

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `color_by` | string | (none) | Obs column name to include as color values |

**Response**: `application/vnd.apache.arrow.stream` (binary)

The Arrow IPC response contains a RecordBatch with columns:
- `x` (Float32) — first dimension coordinates
- `y` (Float32) — second dimension coordinates
- `color` (Int32 or Float32) — category codes or continuous values (if `color_by` specified)

**Example usage** (JavaScript):
```javascript
const response = await fetch(`/api/v1/datasets/${id}/embeddings/X_umap?color_by=leiden`);
const buffer = await response.arrayBuffer();
// Decode in Web Worker using apache-arrow tableFromIPC()
```

---

## Metadata

### List Metadata Columns

```
GET /datasets/{dataset_id}/metadata
```

**Response** (200):
```json
[
  {"name": "leiden", "dtype": "category", "n_unique": 8, "values": ["0","1","2","3","4","5","6","7"]},
  {"name": "n_genes", "dtype": "int64", "n_unique": 1352, "values": null},
  {"name": "percent_mito", "dtype": "float64", "n_unique": null, "values": null}
]
```

### Get Metadata Summary

```
GET /datasets/{dataset_id}/metadata/summary
```

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `groupby` | string | auto-detect | Obs column to group by (tries leiden, louvain, seurat_clusters, cluster, celltype) |

**Response** (200):
```json
{
  "groupby": "leiden",
  "counts": {"0": 723, "1": 508, "2": 480, "3": 351, "4": 300, "5": 168, "6": 130, "7": 40}
}
```

### Get Metadata Column Values

```
GET /datasets/{dataset_id}/metadata/{column}
```

**Response**: `application/vnd.apache.arrow.stream` (binary)

Arrow IPC RecordBatch with a single column containing per-cell values.

---

## Gene Expression

### Get Expression Values

```
GET /datasets/{dataset_id}/expression
```

**Query parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `genes` | string | Yes | Comma-separated gene names (e.g., `CD3D,CD8A,MS4A1`) |

**Response**: `application/vnd.apache.arrow.stream` (binary)

Arrow IPC RecordBatch with one Float32 column per requested gene (columns named by gene symbol). Only genes found in the dataset are included.

### Get Violin Plot Data

```
GET /datasets/{dataset_id}/expression/violin
```

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `gene` | string | (required) | Gene name |
| `groupby` | string | `leiden` | Obs column to group by |

**Response** (200):
```json
{
  "gene": "CD3D",
  "groupby": "leiden",
  "groups": {
    "0": [0.0, 1.2, 3.4, ...],
    "1": [0.0, 0.0, 0.5, ...],
    ...
  }
}
```

### List All Genes

```
GET /datasets/{dataset_id}/genes
```

**Response** (200):
```json
{"genes": ["A1BG", "A1CF", "A2M", ...]}
```

### Search Genes (Autocomplete)

```
GET /datasets/{dataset_id}/genes/search
```

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q` | string | (empty) | Search prefix |
| `limit` | int | 20 | Max results (1–100) |

**Response** (200):
```json
{"query": "CD3", "results": ["CD3D", "CD3E", "CD3G", "CD300A", "CD300C"]}
```

---

## Marker Genes

### Get Marker Gene Table

```
GET /datasets/{dataset_id}/markers
```

**Query parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `groupby` | string | (all groups) | Filter to a specific group |
| `format` | string | `arrow` | Response format: `arrow` or `json` |

**JSON Response** (200):
```json
{
  "columns": ["group", "names", "scores", "logfoldchanges", "pvals", "pvals_adj", "pct_nz_group", "pct_nz_reference"],
  "data": [
    {"group": "0", "names": "IL32", "scores": 8.45, "logfoldchanges": 2.1, "pvals": 1e-15, "pvals_adj": 3e-12, ...},
    ...
  ],
  "n_rows": 800
}
```

**Arrow Response**: `application/vnd.apache.arrow.stream` (binary) — same columns as JSON but in Arrow IPC format.

---

## Data Assessment

### Get Preprocessing State

```
GET /datasets/{dataset_id}/assessment
```

**Response** (200):
```json
{
  "qc_metrics": {"done": true, "confidence": "high", "details": "Found n_genes_by_counts, n_counts, pct_counts_mt in obs"},
  "filtering": {"done": true, "confidence": "medium", "details": "Min genes/cell appears to be ~200 based on distribution"},
  "normalization": {"done": true, "confidence": "high", "details": "X contains float values and raw layer exists"},
  "log_transform": {"done": true, "confidence": "medium", "details": "Max value in X is 9.2 (< 20), suggesting log-transformed"},
  "highly_variable_genes": {"done": true, "confidence": "high", "details": "Found 2000 highly_variable genes in var"},
  "scaling": {"done": false, "confidence": "high", "details": "Gene means are not centered around 0"},
  "pca": {"done": true, "confidence": "high", "details": "X_pca found in obsm with 50 components"},
  "neighbors": {"done": true, "confidence": "high", "details": "connectivities found in obsp"},
  "clustering": {"done": true, "confidence": "high", "details": "leiden found in obs with 8 clusters"},
  "embeddings": {"done": true, "confidence": "high", "details": "X_umap found in obsm"},
  "marker_genes": {"done": true, "confidence": "high", "details": "rank_genes_groups found in uns"},
  "cell_cycle": {"done": false, "confidence": "high", "details": "S_score and G2M_score not found in obs"}
}
```

### Run Pipeline Steps

```
POST /datasets/{dataset_id}/assessment/run
Content-Type: application/json
```

**Request body:**
```json
{
  "steps": ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes", "pca", "neighbors", "clustering", "umap", "markers"],
  "params": {
    "min_genes": 200,
    "min_cells": 3,
    "max_pct_mt": 20.0,
    "target_sum": 10000,
    "n_top_genes": 2000,
    "n_neighbors": 15,
    "resolution": 1.0
  }
}
```

**Response** (200):
```json
{
  "steps_run": ["qc_metrics", "filtering", "normalization", "log_transform", "highly_variable_genes", "pca", "neighbors", "clustering", "umap", "markers"],
  "n_cells_before": 2700,
  "n_cells_after": 2638,
  "n_genes_before": 13714,
  "n_genes_after": 2000,
  "output_path": "uploads/abc123_v2.h5ad",
  "version": 2
}
```

### Get AI Suggestions

```
POST /datasets/{dataset_id}/assessment/suggest
```

**Response** (200):
```json
{
  "suggestions": [
    {
      "step": "filtering",
      "recommended": true,
      "reasoning": "Dataset shows cells with very low gene counts. Filtering cells with fewer than 200 genes and genes expressed in fewer than 3 cells is recommended.",
      "suggested_params": {"min_genes": 200, "min_cells": 3, "max_pct_mt": 20.0}
    },
    {
      "step": "normalization",
      "recommended": true,
      "reasoning": "Raw counts detected. Library-size normalization to 10,000 counts per cell is standard practice.",
      "suggested_params": {"target_sum": 10000}
    }
  ],
  "raw_response": "Based on the preprocessing state..."
}
```

If no language model key is configured (`DEEPINFRA_API_KEY`, or `LLM_API_KEY` for another provider), returns rule-based suggestions with the same format.

---

## Gene Sets

### List MSigDB Collections

```
GET /datasets/{dataset_id}/genesets/collections
```

### Search Gene Sets

```
GET /datasets/{dataset_id}/genesets/search?q=interferon
```

### Score a Gene Set

```
POST /datasets/{dataset_id}/genesets/score
Content-Type: application/json
```

**Request body:**
```json
{
  "genes": ["ISG15", "IFI6", "IFI44", "IFIT1", "MX1"],
  "name": "interferon_response"
}
```

**Response** (200):
```json
{
  "name": "interferon_response",
  "scores": [0.12, -0.05, 0.34, ...],
  "n_cells": 2700
}
```

---

## Enrichment

### Get Pre-computed Enrichment

```
GET /datasets/{dataset_id}/enrichment
```

### List Available Groups

```
GET /datasets/{dataset_id}/enrichment/groups
```

### Compute Enrichment

```
POST /datasets/{dataset_id}/enrichment/compute
Content-Type: application/json
```

**Request body:**
```json
{
  "group": "0",
  "gene_sets": "GO_Biological_Process_2021",
  "top_n": 20
}
```

---

## Trajectory

PAGA, diffusion pseudotime (DPT) and the genes that change along a pseudotime
(`core/trajectory.py`, `core/pseudotime_genes.py`). A pseudotime column is any numeric obs
column whose name holds `pseudotime`, `dpt_`, `latent_time` or `monocle`.

### List Pseudotime Columns

```
GET /datasets/{dataset_id}/trajectory
```

**Response** (200):
```json
{
  "pseudotime_columns": ["scview_dpt_pseudotime", "scview_dpt_Ductal"],
  "n_columns": 2,
  "computed": {
    "column": "scview_dpt_pseudotime", "root_named_column": "scview_dpt_Ductal",
    "root_column": "clusters", "root_value": "Ductal", "root_cell": "GACGCGTCATTTCAGG",
    "graph": "existing neighbour graph (use_rep=X_pca, k=15)", "n_unreachable": 0,
    "n_detached": 0, "warnings": [], "stale": null
  },
  "runs": {"scview_dpt_Ductal": {"...": "the same record"}},
  "velocity_layers": true
}
```

`computed` is the latest in-app run (null when none), `runs` one record per root's column.
`stale` says why a run no longer matches the data (a graph, PCA or data step since, or another
neighbour graph now), else null. `velocity_layers`: spliced and unspliced counts are attached.

### Compute Diffusion Pseudotime

```
POST /datasets/{dataset_id}/trajectory/dpt
Content-Type: application/json
```

**Request body:** `{"root_column": "celltype", "root_value": "Blood progenitors 1"}`

Writes `scview_dpt_pseudotime` (the latest run) and `scview_dpt_<root>` (kept when another root
runs). The response is the run record above, without `stale`. Cells the root cannot reach get no
pseudotime (`n_unreachable`). Cells past an extreme gap at the far end (at most 5% of the cells,
the gap wider than the range below it) are a weakly linked island: they are left out
(`n_detached`) and DPT is computed again without them. `warnings` says so, and flags a root that
sits inside the saved PAGA tree of its column or runs against another pseudotime column
(Spearman rho <= -0.5). 400 when no cell has the root value or there is no graph or PCA.

### PAGA

```
POST /datasets/{dataset_id}/trajectory/paga
GET  /datasets/{dataset_id}/trajectory/paga?embedding=X_umap
```

**POST body:** `{"groupby": "celltype"}`. Returns the record: `groups`, `sizes`, `edges`
(`[i, j, connectivity]`), `tree` (maximum spanning tree edges), `graph`, `n_cells`.
GET returns `{"paga": record | null, "positions": [[x, y], ...] | null, "stale": str | null,
"root_group": str | null}`: each group's median position on `embedding`, why the record is out
of date (its grouping or graph changed), and the group holding the pseudotime root cell.

### Get Pseudotime Values

```
GET /datasets/{dataset_id}/trajectory/{column}
```

**Response** (200): `{"column", "n_cells", "values": [0.01, null, ...], "min", "max"}`; null for
a cell without a finite pseudotime.

### Get Gene Expression Along Pseudotime

```
GET /datasets/{dataset_id}/trajectory/{column}/genes?genes=Hba-x,Gata2&n_bins=20
```

Up to 10 genes. `n_bins` (10-500, default 100) is the most bins: bins hold equal cell counts
(quantile bins, at least 10 cells each), each at its median pseudotime, as the trend-genes
ranking bins.

**Response** (200):
```json
{
  "pseudotime_column": "scview_dpt_pseudotime",
  "n_bins": 20,
  "binning": "equal cell counts along pseudotime (quantile bins); each bin at its median pseudotime",
  "genes": {
    "Hba-x": {
      "found": true,
      "scatter_pseudotime": [0.01, ...], "scatter_expression": [0.0, ...],
      "binned_pseudotime": [0.11, ...], "binned_expression": [0.11, ...], "binned_counts": [490, ...]
    },
    "NOPE": {"found": false}
  }
}
```

### Rank Genes Along Pseudotime

```
POST /datasets/{dataset_id}/trajectory/{column}/trend-genes
Content-Type: application/json
```

**Request body:** `{"n_top": 50, "n_bins": 40, "groups_column": "clusters", "groups": ["Ductal", "Ngn3 low EP"]}`
(the last two optional: one branch). Each gene's log-normalised expression is fitted with a
natural cubic spline of pseudotime order and tested against a constant (F); genes are ranked
by F. Returns `genes` (`f_stat`, `pval`, `pval_adj`, `r2`, `amplitude`, `peak_pseudotime`,
`pattern` rising | falling | transient, binned `mean` and `fitted`) and `bins`.

---

## Export

### Export Data

```
POST /datasets/{dataset_id}/export
Content-Type: application/json
```

**Request body:**
```json
{
  "type": "markers",
  "format": "csv",
  "options": {
    "groupby": "0"
  }
}
```

**Response**: `application/octet-stream` (file download)

Supported export types: `markers`, `metadata`, `expression`
Supported formats: `csv`, `xlsx`

---

## Streaming runs (SSE)

The long runs below answer with `Content-Type: text/event-stream`. Each event is
`event: <type>` + `data: <json>`: `start` (the plan and its estimate), `progress` or `stage`,
`watchdog` (memory warnings), `result` (the saved result), then `complete`, or `error`
(`{"error": "<one sentence>"}`). Lines starting with `:` are keep-alives. Closing the
connection stops the run and nothing is saved. A refusal before the run starts (no ATAC,
another run going) is a plain 4xx with `{"detail": ...}`.

---

## Multi-file Import (ingest sessions)

| Route | Body | What |
|-------|------|------|
| `POST /ingest/session` | none | `{"session_id"}` |
| `POST /ingest/session/{sid}/files` | multipart: `files` (repeat), optional `paths` (one per file, its path inside an uploaded folder) | stage files; answers with what was detected (bundle, side files, issues); BAM, Loupe and HTML files are skipped |
| `GET /ingest/session/{sid}` | | the detection, bundle and validation |
| `GET /ingest/session/{sid}/merge-plan` | | several samples: genes and cells shared |
| `POST /ingest/session/{sid}/options` | `{"name", "join", "sample_label", "streamed"}` | |
| `POST /ingest/session/{sid}/convert` | | convert R objects now (background) |
| `POST /ingest/session/{sid}/commit` | | `{"dataset_id"}` |
| `DELETE /ingest/session/{sid}` | | discard |

```bash
SID=$(curl -s -X POST $BASE/api/v1/ingest/session | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')
curl -F "files=@outs/filtered_feature_bc_matrix.h5" -F "files=@outs/atac_fragments.tsv.gz" \
     -F "files=@outs/atac_fragments.tsv.gz.tbi" -F "files=@outs/per_barcode_metrics.csv" \
     $BASE/api/v1/ingest/session/$SID/files
curl -X POST $BASE/api/v1/ingest/session/$SID/commit
```

---

## Protein (CITE-seq)

The antibodies live beside the genes (obsm `protein_counts`, `protein_clr`, `protein_dsb`)
and read as expression layers `protein` (CLR), `protein_counts` and `protein_dsb` in
`GET /expression`, `/expression/by-group` and the violin routes.

| Route | Body / query | What |
|-------|--------------|------|
| `POST /datasets/{id}/de` | `{"indices", "label", "method", "min_pct", "layer"}` | `layer`: `protein` (CLR) or `protein_counts` tests the proteins instead of the genes |
| `POST /datasets/{id}/de/compare` | `{"split_by", "group_a", "group_b", "within_column", "within_value", "method", "layer", ...}` | A vs B; `layer` as above, cell-level methods only (pseudobulk is refused with `layer`) |
| `GET /datasets/{id}/expression/per-donor` | `gene`, `groupby`, `sample`, `condition`, `method`, `conditions`, `reference`, `groups`, `all_cells`, `layer` | one value per donor; `all_cells=true`: the whole dataset as one group; `layer=protein`: a protein's mean CLR per donor |
| `GET /datasets/{id}/protein/dsb` / `POST` | `{"mode", "rna_window", "prot_window", "isotypes", "denoise"}` | DSB normalisation (empty droplets or ModelNegativeADTnorm) |
| `GET /datasets/{id}/wnn` | `groupby` | WNN RNA + protein: what a run would use, the saved result |
| `POST /datasets/{id}/wnn/run-stream` | `{"protein_pcs", "resolution"}` | SSE; saves `X_umap_wnn`, `wnn_rna_weight`, `wnn_leiden` |
| `GET /datasets/{id}/wnn/weights` | `groupby` | per-cell RNA weight (and by group) |

---

## Perturbation (CRISPR screens)

| Route | Body | What |
|-------|------|------|
| `GET /datasets/{id}/perturbation` | | guide sources, the suggested one, the saved assignment, the last Mixscape run |
| `POST /datasets/{id}/perturbation/preview` | `{"source": "counts"\|"cellranger"\|"column", "column", "min_umi": 5, "min_ratio": 2, "mapping": {guide: target}, "control"}` | what `assign` would write; nothing changes |
| `POST /datasets/{id}/perturbation/assign` | same | saves `guide_call`, the perturbation column, `n_guides` |
| `POST /datasets/{id}/perturbation/mixscape/run-stream` | `{"labels", "control", "split_by", "de_genes": "all"\|"variable", "n_dims": 15, "n_neighbors": 20, "logfc_threshold", "pval_cutoff", "min_de_genes": 5, "iter_num", "lda", "guide_column", "replicate_column"}` | SSE; saves `mixscape_class`, `mixscape_class_global`, `mixscape_p_ko`; the result has `targets`, `per_guide`, `per_replicate` |
| `GET /datasets/{id}/perturbation/mixscape` | | the last run on the current cells |

---

## Cell Hashing

| Route | Body | What |
|-------|------|------|
| `GET /datasets/{id}/hashing` | | tags (obsm `hto_counts`), antibody features that look like hashtags (`suggested`), the saved call, a suggested `split_by` |
| `POST /datasets/{id}/hashing/demux` | `{"tags", "method": "htodemux"\|"hashedDrops", "quantile": 0.99, "split_by", "allow_warnings": false}` | saves `hash_call`, `hash_class`, `hash_max`, `hash_second`, `hash_margin`; a call the checks find unsafe is a 409 naming why, unless `allow_warnings` |
| `POST /datasets/{id}/hashing/samples` | `{"mapping": {tag: sample}, "column": "hash_sample"}` | a sample column from the call |
| `GET /datasets/{id}/hashing/plot-data` | | per-tag CLR histograms with cut-offs, tag x call heatmap |

---

## ATAC (10x Multiome)

| Route | Body / query | What |
|-------|--------------|------|
| `GET /datasets/{id}/atac` | `groupby`, `allow_empty` | peaks, genome, LSI / activity / WNN runs, what a run would need |
| `POST /datasets/{id}/atac/run-stream` | `{"n_lsi": 50, "dims_to", "drop_first": "auto"\|"yes"\|"no", "resolution", "activity": true, "activity_source": "auto"\|"fragments"\|"peaks"}` | SSE: TF-IDF, LSI, graph, `X_umap_atac`, `atac_leiden`, gene activity (fragments = Signac GeneActivity) |
| `POST /datasets/{id}/atac/wnn/run-stream` | `{"lsi_first": 2, "lsi_last", "resolution"}` | SSE: WNN RNA + ATAC; saves `X_umap_wnn_atac`, `wnn_atac_rna_weight`, `wnn_atac_leiden` |
| `GET /datasets/{id}/atac/activity` | `gene`, `groupby` | one gene's activity per cell, by group, and against its RNA |
| `GET /datasets/{id}/atac/peaks` | `offset`, `limit`, `sort`, `descending`, `search`, `selected_only` | every peak's totals, nearest gene, class |
| `GET /datasets/{id}/atac/qc` | `groupby` | the ATAC QC columns per cell |
| `POST /datasets/{id}/atac/metrics` | multipart `file` (per_barcode_metrics.csv), `sample`, `column` | adds FRiP, TSS and mito fractions |
| `GET /datasets/{id}/atac/differential-estimate` | `column`, `test`, `group`, `vs` | time and memory of a differential run |
| `POST /datasets/{id}/atac/differential` | `{"column", "groups", "vs", "test": "wilcoxon"\|"lr", "min_pct": 0.05, "max_peaks", "lrt_peaks", "standard_only": true, "fold_change": "signac"\|"tfidf"}` | SSE: each group vs the rest, or `groups: [A]` vs `vs: B`; kept by id; `fold_change` = the log2fc formula (Signac's FoldChange, or mean TF-IDF + 1 as before 0.79) |
| `GET /datasets/{id}/atac/differential[/{rid}[/csv]]` | `group`, `offset`, `limit` | the last result, a page, or the CSV |
| `GET /datasets/{id}/atac/export` | | an h5ad of the peak counts |
| `GET /datasets/{id}/atac/fragments` | | the attached fragments file, its QC estimate and saved QC |
| `POST /datasets/{id}/atac/fragments?name=` | raw body (`curl --data-binary @atac_fragments.tsv.gz`), then the `.tbi` | attach the fragments file and its index |
| `DELETE /datasets/{id}/atac/fragments` | | remove them |
| `POST /datasets/{id}/atac/fragments/qc-stream` | none | SSE: TSS enrichment, nucleosome signal, fragments per cell (obs columns) |
| `GET /datasets/{id}/atac/fragments/values`, `/lengths` | `group_by` | per-cell QC values and length histograms by group |
| `GET /datasets/{id}/atac/coverage` | `gene` or `region` (chr1:1000-5000, up to 2 Mb), `group_by`, `groups`, `bins` | depth per group per bin, peaks and genes in the window |
| `POST /datasets/{id}/atac/links/estimate` | `{"genes": "variable"\|"all"\|"list", "gene_list", "cells_column", "cells_value"}` | what a links run needs (a first genome download) |
| `POST /datasets/{id}/atac/links` | same | SSE: Signac LinkPeaks; kept by id |
| `GET /datasets/{id}/atac/links[/{rid}[/csv\|/locus]]` | `gene`, `search`, `sort`, `offset`, `limit` | the last result, a page, the CSV, one gene's arcs |
| `GET /datasets/{id}/atac/motifs` | | what the scan and chromVAR need, whether each is done |
| `POST /datasets/{id}/atac/motifs/scan` (+ `/cancel`) | none | SSE: JASPAR 2024 x peaks (motifmatchr); the scan outlives the request |
| `POST /datasets/{id}/atac/motifs/enrichment` | `{"da_id", "groups", "p_column", "p_max": 0.005, "min_pct": 0.05, "min_log2fc", "background": 40000, "seed"}` | Signac FindMotifs on a differential result; kept by id |
| `GET /datasets/{id}/atac/motifs/enrichment[/{rid}[/csv\|/export]]` | `group`, `offset`, `limit` | the last result, a page, the CSV, a zip with logos |
| `GET /datasets/{id}/atac/motifs/logo/{motif_id}` | | a motif's logo (SVG) |
| `POST /datasets/{id}/atac/motifs/chromvar` | `{"seed"}` | SSE: chromVAR deviations |
| `GET /datasets/{id}/atac/motifs/chromvar[/groups\|/motif\|/csv]` | `column`; `motif_id`, `groupby`, `save` | the run, groups x motifs, one motif per cell, the ranking |

The differential, links, motif enrichment and chromVAR results are also in the results store:
`GET /datasets/{id}/results?kind=atac_differential` (`atac_links`, `atac_motif_enrichment`,
`atac_chromvar`), `GET /datasets/{id}/results/{rid}` (settings) and `.../table.tsv.gz`.

---

## RNA Velocity

| Route | Body / query | What |
|-------|--------------|------|
| `GET /datasets/{id}/velocity` | | spliced / unspliced counts attached, what matched, the last run, its estimate |
| `POST /datasets/{id}/velocity/attach` | multipart `files` (a velocyto .loom, an .h5ad with the layers, or STARsolo Velocyto files; one loom per sample together) | matched to the cells; replaces earlier counts |
| `POST /datasets/{id}/velocity/run-stream` | `{"method": "stochastic"\|"steady_state", "n_top_genes", "n_neighbors", "group_by"}` | SSE: scVelo 0.3.3's steady-state or stochastic model |
| `GET /datasets/{id}/velocity/grid` | `embedding`, `density` | arrows on an embedding |
| `GET /datasets/{id}/velocity/phase` | `gene`, `color_by` | one gene's phase portrait |
| `GET /datasets/{id}/velocity/genes` | | the fitted genes: gamma, r2, whether each passed |

---

## MCP

An AI agent's tools over these routes at `/mcp` (Streamable HTTP; `MCP_ENABLED`, on in
private mode). The tools for the routes above: `guide_assignment`, `mixscape`, `get_mixscape`,
`hashing_demux`, `hashing_samples`, `atac_status`, `atac_fragment_qc`, `atac_lsi`, `wnn`,
`atac_gene_activity`, `atac_differential`, `atac_links`, `atac_motif_enrichment`, `atac_chromvar`,
`plot(kind="coverage")`, and `layer="protein"` on `differential_expression` /
`per_donor_expression`. `upload_instructions` gives the curl recipes above;
`import_server_path` imports a file or folder on the server under `MCP_IMPORT_DIRS` (off when
unset, always off with `DEPLOYMENT_MODE=public`).

---

## Health Check

```
GET /health
```

**Response** (200):
```json
{"status": "healthy"}
```

---

## Response Formats

### JSON
Standard JSON responses for metadata, summaries, search results, and structured data.

### Arrow IPC
Binary responses for large numerical data (embeddings, expression, metadata columns). Content type: `application/vnd.apache.arrow.stream`.

To decode in JavaScript:
```javascript
import { tableFromIPC } from 'apache-arrow';

const response = await fetch(url);
const buffer = await response.arrayBuffer();
const table = tableFromIPC(new Uint8Array(buffer));
const xColumn = table.getChild('x').toArray();  // Float32Array
```

### Error Responses

All errors follow this format:
```json
{"detail": "Human-readable error message"}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (missing parameters, invalid input) |
| 404 | Resource not found (dataset, gene, column) |
| 413 | File too large (exceeds MAX_UPLOAD_SIZE_MB) |
| 500 | Internal server error |
