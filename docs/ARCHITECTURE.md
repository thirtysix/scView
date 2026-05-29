# scView Architecture

## System Overview

scView is a three-service Docker Compose application for interactive single-cell RNA-seq visualization.

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  React 18 + TypeScript + Vite                            │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │   │
│  │  │ deck.gl   │  │ Plotly   │  │ Zustand + React Query │  │   │
│  │  │ (WebGL)   │  │ (Charts) │  │ (State Management)    │  │   │
│  │  └──────────┘  └──────────┘  └───────────────────────┘  │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │ Web Workers (Arrow decode + color mapping)       │    │   │
│  │  └──────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │ HTTP (JSON + Arrow IPC binary)                        │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│  Docker │Compose                                                │
│         ▼                                                       │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │  Frontend     │──proxy─│  Backend     │                       │
│  │  (Nginx)      │  /api  │  (FastAPI)   │                       │
│  │  Port 3000    │        │  Port 8000   │                       │
│  └──────────────┘        └──────┬───────┘                       │
│                                 │                               │
│                          ┌──────┴───────┐                       │
│                          │ anndata /    │                       │
│                          │ scanpy       │                       │
│                          └──────┬───────┘                       │
│                                 │                               │
│                          ┌──────┴───────┐    ┌──────────────┐   │
│                          │ Shared Volume│◄───│  Converter   │   │
│                          │ /data        │    │  (R/Seurat)  │   │
│                          └──────────────┘    │  Port 8001   │   │
│                                              └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Three Services

### 1. Backend (Python/FastAPI)

**Image size**: ~1 GB
**Port**: 8000
**Role**: Data loading, analysis, and API serving

Key components:
- **FastAPI** app with async endpoints, CORS, GZip middleware
- **AnnData adaptor** — lazy-loads h5ad files, extracts embeddings/metadata/expression
- **Arrow serializer** — converts numpy/pandas to Apache Arrow IPC binary
- **Dataset manager** — LRU cache (max 3 datasets), JSON sidecar metadata
- **Assessor** — detects 12 preprocessing steps with confidence levels
- **Pipeline** — executes missing scanpy steps, saves versioned output
- **LLM advisor** — DeepInfra integration for analysis suggestions (optional)

### 2. Frontend (React/Nginx)

**Image size**: ~100 MB (production)
**Port**: 3000 (production) / 5173 (dev)
**Role**: UI rendering, WebGL visualization, user interaction

Key components:
- **deck.gl** ScatterplotLayer with OrthographicView for 2D scatter (200k cells at 60fps)
- **Web Workers** for Arrow IPC decoding and color mapping (off main thread)
- **Zustand** stores for UI state (panel selection, settings, selection)
- **React Query** for server state caching and refetching
- **Plotly.js** for violin plots, bar charts, line plots
- **Nginx** reverse proxy (production) routes `/api/*` to backend

### 3. Converter (R/Seurat)

**Image size**: ~1.5 GB
**Port**: 8001
**Role**: Seurat .rds → .h5ad conversion

Key components:
- **httpuv** HTTP server with `/health` and `/convert` endpoints
- **sceasy** for standard Seurat→AnnData conversion
- **Manual fallback** for Seurat v5 and edge cases (direct matrix extraction)
- Supports Seurat v3, v4, and v5 object structures

## Data Flow

### Upload → Visualization Pipeline

```
User drops file
      │
      ▼
POST /api/v1/datasets/upload
      │
      ├── .h5ad file ──────────────────────┐
      │                                    │
      └── .rds file                        │
           │                               │
           ▼                               │
    POST converter:8001/convert            │
           │                               │
           ▼                               │
    sceasy / manual fallback               │
           │                               │
           ▼                               │
    Writes .h5ad to shared volume          │
           │                               │
           ▼                               │
    AnnData adaptor loads h5ad  ◄──────────┘
           │
           ▼
    Dataset metadata cached (JSON sidecar)
           │
           ▼
    Frontend fetches embedding
           │
           ▼
    GET /api/v1/datasets/{id}/embeddings/{name}
           │
           ▼
    Arrow IPC binary response (~1.6 MB for 200k cells)
           │
           ▼
    Web Worker decodes Arrow → Float32Array
           │
           ▼
    deck.gl renders ScatterplotLayer (WebGL)
```

### Arrow IPC Binary Transfer

This is the performance-critical path. Instead of JSON:

```
Traditional:  numpy → JSON serialize → HTTP → JSON parse → typed array
              200k cells: ~6 MB, ~200ms parse

scView:       numpy → Arrow IPC binary → HTTP → Arrow decode → typed array
              200k cells: ~1.6 MB, ~5ms decode
```

The Arrow IPC format maps directly to typed arrays, enabling near-zero-copy transfer from Python to JavaScript. deck.gl consumes typed arrays natively, so the decode-to-render path has minimal overhead.

## Backend Architecture

### Directory Structure

```
backend/src/scview/
├── main.py                 # FastAPI app entry point
├── config.py               # Pydantic BaseSettings
├── dependencies.py         # FastAPI dependency injection
├── api/
│   ├── router.py           # Aggregates all v1 routers
│   └── v1/
│       ├── datasets.py     # Upload, list, info, delete
│       ├── assessment.py   # Preprocessing state, pipeline, suggestions
│       ├── embeddings.py   # Coordinates as Arrow IPC
│       ├── metadata.py     # Obs column values and summaries
│       ├── expression.py   # Gene expression as Arrow IPC
│       ├── markers.py      # Marker gene table
│       ├── genesets.py     # Gene set scoring
│       ├── enrichment.py   # Pathway enrichment
│       ├── trajectory.py   # Pseudotime data
│       ├── export.py       # CSV/Excel export
│       └── ws.py           # WebSocket progress
├── core/
│   ├── anndata_adaptor.py  # h5ad → numpy/pandas interface
│   ├── arrow_serializer.py # numpy → Arrow IPC binary
│   ├── dataset_manager.py  # LRU cache, metadata sidecar
│   ├── conversion.py       # Orchestrate Seurat→h5ad
│   ├── assessor.py         # Detect preprocessing state
│   ├── pipeline.py         # Run scanpy preprocessing
│   └── llm_advisor.py      # DeepInfra LLM integration
└── models/
    ├── schemas.py          # Pydantic response models
    └── enums.py            # DatasetStatus, EmbeddingType
```

### AnnData Adaptor

The adaptor is the core data access layer. It wraps an `anndata.AnnData` object and provides clean methods for extracting data:

- **Lazy loading**: Files >2 GB use `backed='r'` mode (memory-mapped)
- **Sparse-aware**: Detects scipy sparse matrices and calls `.toarray()` only on requested slices
- **Embedding extraction**: Reads from `adata.obsm` (X_umap, X_tsne, X_pca, etc.)
- **Expression extraction**: Reads from `adata.X` (or `adata.raw.X`), returns float32 numpy arrays
- **Gene search**: Prefix matching on `adata.var_names` for autocomplete

### Dataset Manager

Manages loaded datasets with an LRU eviction policy:

- **Max 3 datasets** in memory simultaneously
- **JSON sidecar** (`<dataset_id>_meta.json`) caches cell/gene counts, available embeddings, and obs column info — avoids re-scanning the h5ad on every request
- **Async loading** with asyncio locks to prevent duplicate loads

### Assessor (Data Assessment)

Inspects 12 preprocessing steps:

| Step | Detection Method |
|------|-----------------|
| QC metrics | `n_genes_by_counts` in `adata.obs` |
| Filtering | Heuristic: min genes per cell, min cells per gene |
| Normalization | `adata.X.dtype` is float AND (`adata.raw` exists OR `counts` in layers) |
| Log transform | `adata.X.max() < 20` AND float dtype |
| HVGs | `highly_variable` in `adata.var` |
| Scaling | Per-gene mean ~ 0, std ~ 1 (sample check) |
| PCA | `X_pca` in `adata.obsm` |
| Neighbors | `connectivities` in `adata.obsp` |
| Clustering | `leiden` or `louvain` in `adata.obs` |
| Embeddings | `X_umap` or `X_tsne` in `adata.obsm` |
| Markers | `rank_genes_groups` in `adata.uns` |
| Cell cycle | `S_score` and `G2M_score` in `adata.obs` |

Each returns a `StepStatus` with `done` (bool), `confidence` (high/medium/low), and `details` (human-readable explanation).

### Pipeline Runner

Executes missing scanpy steps in canonical dependency order:

```
QC metrics → Filter → Normalize → Log1p → HVG → Scale → PCA → Neighbors → Cluster → UMAP → Markers
```

Key design choices:
- **Won't run PCA without normalization** — enforces dependency order
- **Saves as new version** (`dataset_v2.h5ad`) — original always preserved
- **Configurable parameters** via `PipelineParams` Pydantic model with sensible defaults
- **Clustering → Enrichment linking** — when clustering runs, the active clustering column name is stored in `adata.uns["scview_active_clustering"]`. Both `_run_marker_genes()` and `_run_enrichment()` auto-inject this column into their target lists, ensuring markers and enrichment are always computed for newly created clusters even if the frontend prediction doesn't match.

## Frontend Architecture

### Directory Structure

```
frontend/src/
├── main.tsx                    # React 18 entry with QueryClientProvider
├── App.tsx                     # Root component
├── index.css                   # Tailwind v4 theme (teal accent, dark sidebar)
├── api/
│   ├── client.ts               # apiFetch, apiFetchBinary, apiUpload
│   ├── types.ts                # TypeScript interfaces
│   ├── datasets.ts             # Dataset API calls
│   └── embeddings.ts           # Embedding API calls
├── stores/
│   ├── datasetStore.ts         # Current dataset, upload state
│   ├── viewStore.ts            # Active panel, sidebar collapsed
│   ├── selectionStore.ts       # Selected cells, highlight groups
│   ├── settingsStore.ts        # Embedding, colorBy, pointSize, opacity
│   └── unifiedViewStore.ts     # Unified View scatter overlay, violin, active tab
├── hooks/
│   └── useEmbedding.ts         # Fetch → decode → positions + colors
├── workers/
│   └── arrowDecoder.worker.ts  # Arrow IPC → typed arrays (Web Worker)
├── lib/
│   ├── constants.ts            # Panel IDs, API base URL
│   ├── colors.ts               # Categorical/continuous color palettes
│   ├── formatting.ts           # Number/p-value formatters
│   └── arrow.ts                # Synchronous Arrow decode utilities
├── components/
│   ├── layout/                 # AppLayout, Sidebar, Header, PanelContainer
│   ├── panels/                 # Feature panels (Overview, DataAssessment, UnifiedView, etc.)
│   ├── unified/                # Unified View subtab components
│   │   ├── UnifiedMarkersSubtab.tsx      # Marker genes table + |FC| filter
│   │   ├── UnifiedEnrichmentSubtab.tsx   # Pathway enrichment + MSigDB + scoring
│   │   ├── UnifiedGeneSetsSubtab.tsx     # Gene set browsing, search, scoring
│   │   └── UnifiedExpressionSubtab.tsx   # Gene search + expression overlay
│   ├── plots/                  # EmbeddingScatter, ViolinPlot, PlotControls, etc.
│   ├── tables/                 # (Future: TanStack Table components)
│   └── common/                 # GeneSearch, ExportMenu, ErrorBoundary, LoadingSpinner
```

### State Management

**Zustand** (UI state — no server round-trip):
- `datasetStore`: current dataset ID, dataset info, upload progress
- `viewStore`: active panel, sidebar collapsed
- `selectionStore`: selected cell indices (Set), selection mode, highlighted group
- `settingsStore`: embedding name, color-by column, point size, opacity, background

**React Query** (server state — cached, refetchable):
- Embedding coordinates (Arrow binary)
- Gene expression data (Arrow binary)
- Metadata columns (Arrow binary)
- Marker genes, enrichment results, assessment state (JSON)

### Web Worker Pipeline

```
Main Thread                          Worker Thread
     │                                    │
     │  postMessage(arrayBuffer)          │
     │ ──────────────────────────────►    │
     │                                    │  tableFromIPC(buffer)
     │                                    │  Extract Float32Array columns
     │                                    │
     │    postMessage(typedArrays,        │
     │      [transferables])              │
     │ ◄────────────────────────────────  │
     │                                    │
     │  Zero-copy: ArrayBuffers           │
     │  transferred, not copied           │
     ▼                                    │
  deck.gl renders                         │
```

The Arrow decoding worker receives raw ArrayBuffer, uses `tableFromIPC()` from the `apache-arrow` library, extracts typed arrays (Float32Array for coordinates, Int32Array for categories), and transfers them back via the `Transferable` API. This avoids copying potentially large arrays (200k cells × 2 floats = 1.6 MB).

### Rendering Pipeline

```
useEmbedding hook
      │
      ├── React Query: fetch ArrayBuffer from /api/v1/.../embeddings/X_umap
      │
      ├── Web Worker: decode Arrow IPC → { x: Float32Array, y: Float32Array, color?: ... }
      │
      ├── Interleave: [x0, y0, 0, x1, y1, 0, ...] → Float32Array (positions)
      │
      └── Color mapping: category/value → Uint8Array RGBA (4 bytes per cell)
            │
            ▼
      EmbeddingScatter.tsx
            │
            ├── DeckGL component with OrthographicView
            ├── ScatterplotLayer (positions + colors as typed arrays)
            ├── Auto-fit initial view from data bounds
            ├── Hover: deck.gl picking → tooltip
            └── Selection: lasso polygon → point-in-polygon → selectionStore
```

### Color System

- **Categorical**: Tableau 20 palette (20 distinct colors), wraps for >20 categories
- **Continuous**: Viridis colormap (perceptually uniform), supports inferno/magma/plasma
- **Selection dimming**: non-selected cells rendered at 20% opacity
- **Group highlighting**: clicked legend item highlights matching cells

## Unified View Panel

The Unified View is a Kana-inspired single-screen workspace that combines scatter plot visualization with analysis tools in a single panel, avoiding the context-switching of separate panels.

### Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  Toolbar: embedding select, colorBy, pointSize, opacity, reset  │
├─────────────────────────────┬─┬──────────────────────────────────┤
│                             │ │  Tabs: Markers│Enrichment│       │
│     deck.gl Scatter         │◀│        GeneSets│Expression│      │
│     (resizable width)       │▶│                                  │
│                             │ │  [Active subtab content]         │
│                             │ │                                  │
├─────────────────────────────┴─┴──────────────────────────────────┤
│  Collapsible Violin Plot (toggleable per gene)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Resizable split pane**: Mouse-drag handle between scatter and tabs (30–80% range). State stored as `splitFraction` with pointer event listeners for smooth dragging.
- **Scatter overlay system**: `scatterOverlay` in `unifiedViewStore` holds a `Float32Array` of continuous values (expression or gene-set scores) that override categorical coloring. Automatically cleared when `colorBy` changes.
- **Enrichment scoring feedback**: When an enrichment term is clicked, the term row shows a spinner + "Scoring..." label while the gene-set score is computed server-side. Other rows are dimmed during scoring.
- **Marker |FC| filter**: Range slider (0–5, step 0.25) filters the marker gene table by absolute log fold-change threshold.
- **Keyboard shortcuts**: 1–4 switch tabs, Escape clears overlay.

### Data Assessment Pipeline Linking

When the user selects both Clustering and Pathway Enrichment in the Data Assessment pipeline:

1. **Frontend prediction**: `DataAssessmentPanel` computes the predicted clustering column name (`scview_{method}_r{resolution}`) and dynamically injects it into the marker and enrichment column checkbox lists with a blue "from Clustering step" badge. A sync effect tracks parameter changes and auto-updates the selections.

2. **Backend safety net**: `pipeline.py` reads `adata.uns["scview_active_clustering"]` (set by the clustering step) and auto-injects it into both `_run_marker_genes()` and `_run_enrichment()` column lists, ensuring the dependency chain works even if the frontend prediction mismatches.

## Converter Architecture

```
POST /convert
    │
    ├── Read .rds file
    │
    ├── Detect Seurat version (v3/v4/v5)
    │   ├── v3/v4: standard slot-based structure
    │   └── v5: layer-based structure (new in Seurat 5)
    │
    ├── Try sceasy::convertFormat()
    │   │
    │   ├── Success → .h5ad written
    │   │
    │   └── Failure → Manual fallback
    │       ├── Extract count matrix from assay
    │       ├── Extract metadata from obj@meta.data
    │       ├── Extract reductions (PCA, UMAP, tSNE)
    │       ├── Build AnnData via reticulate
    │       └── Write .h5ad via anndata$write_h5ad()
    │
    └── Return { status, output_path }
```

## Security Considerations

- **File validation**: Upload accepts only `.h5ad`, `.rds`, `.rdata` extensions
- **CORS**: Configurable via environment variable, defaults to localhost only
- **No authentication**: Designed for local/single-user use (authentication needed for public deployment)
- **Data isolation**: Each dataset gets a unique ID; files stored in a Docker volume
- **API key**: DeepInfra key stored in `.env`, never exposed to the frontend
