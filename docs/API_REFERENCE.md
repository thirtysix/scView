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

If no DeepInfra API key is configured, returns rule-based suggestions with the same format.

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

### List Pseudotime Columns

```
GET /datasets/{dataset_id}/trajectory/columns
```

### Get Pseudotime Values

```
GET /datasets/{dataset_id}/trajectory/values?column=dpt_pseudotime
```

### Get Gene Expression Along Pseudotime

```
GET /datasets/{dataset_id}/trajectory/expression?column=dpt_pseudotime&genes=CD3D,CD8A
```

**Response** (200):
```json
{
  "column": "dpt_pseudotime",
  "genes": {
    "CD3D": {
      "pseudotime": [0.01, 0.02, 0.03, ...],
      "expression": [0.0, 1.2, 0.5, ...],
      "smoothed": [{"x": 0.0, "y": 0.1}, {"x": 0.05, "y": 0.3}, ...]
    }
  }
}
```

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
