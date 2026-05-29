# scView Build Log

## Session: 2026-02-12 — Initial Build (Phases 1–5)

### Context

scView was conceived as a modern replacement for Cerebro, a discontinued R/Shiny/Electron single-cell RNA-seq visualization tool. The goal: a Docker-based, browser-based viewer that accepts both Seurat (.rds) and Scanpy (.h5ad) inputs, with an AI-assisted data assessment system as a key differentiator.

### Research Phase

Before writing any code, three parallel research tasks were conducted:

1. **Current project state** — Confirmed the scView directory was empty (greenfield project).
2. **Cerebro feature mining** — Explored the [romanhaa/Cerebro](https://github.com/romanhaa/Cerebro) GitHub repository to catalog its panel architecture, visualizations, and data flow. Key takeaways: modular panel design (10 panels), Plotly for interactivity, R/Shiny limits on performance.
3. **Modern landscape survey** — Evaluated cellxgene, Vitessce, UCSC Cell Browser, CytoAnalyst, Kana, and BBrowser. Findings: React + WebGL (deck.gl or regl) is the dominant pattern, Python backends with FastAPI/Flask, Apache Arrow for binary data transfer, Docker Compose for deployment.

### Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backend framework | FastAPI (Python) | Native anndata/scanpy support; async; modern |
| Frontend framework | React 18 + TypeScript + Vite | Ecosystem dominance for data viz |
| WebGL rendering | deck.gl ScatterplotLayer | 200k+ cells at 60fps; OrthographicView for 2D |
| Charts | Plotly.js | Built-in violin, bar, heatmap; interactive |
| Data transfer | Apache Arrow IPC (binary) | ~10x faster than JSON for large arrays |
| State management | Zustand (UI) + React Query (server) | Lightweight; clean separation |
| Styling | Tailwind CSS + shadcn/ui primitives | Modern, utility-first, customizable |
| Seurat conversion | Separate R Docker container | Clean separation; avoids bloating Python image |
| LLM integration | DeepInfra API (OpenAI-compatible) | Cheap hosted LLMs; optional dependency |
| Orchestration | Docker Compose (3 services) | Modular, reproducible, shareable |

### Phase 1: Scaffold — Docker + Backend + Frontend + Converter

**Objective:** Running Docker Compose stack with all three services wired together.

**Files created (21):**
- `docker-compose.yml` — 3 services (backend, frontend, converter) with shared named volume
- `docker-compose.dev.yml` — Development overrides with bind mounts and hot-reload
- `.env.example` — Configuration template (DATA_DIR, CORS, DEEPINFRA_API_KEY)
- `Makefile` — Convenience targets: `make dev`, `make build`, `make test`
- `.gitignore` — Python, Node, Docker, IDE, data files
- `backend/Dockerfile` — Multi-stage (dev/production) with uv for fast installs
- `backend/pyproject.toml` — Dependencies: fastapi, anndata, scanpy, pyarrow, openai, gseapy
- `backend/src/scview/main.py` — FastAPI app with CORS, GZip middleware, lifespan handler
- `backend/src/scview/config.py` — Pydantic BaseSettings loading from env
- `backend/src/scview/dependencies.py` — FastAPI DI: singleton DatasetManager, cached Settings
- `backend/src/scview/api/router.py` — Aggregates all v1 endpoint routers
- `backend/src/scview/api/v1/datasets.py` — File upload, list, get info, delete
- `backend/src/scview/models/schemas.py` — DatasetInfo, DatasetUploadResponse, EmbeddingInfo, ObsColumnInfo
- `backend/src/scview/models/enums.py` — DatasetStatus, EmbeddingType
- `converter/Dockerfile` — rocker/r-ver:4.4.0 + Seurat + sceasy + httpuv
- `converter/install_packages.R` — R package installation with tryCatch resilience
- `converter/server.R` — httpuv HTTP server with /health and /convert endpoints
- `converter/convert.R` — Seurat v3/v4/v5 → h5ad conversion with sceasy + manual fallback
- `frontend/Dockerfile` — Multi-stage: dev (Vite) / prod (Nginx)
- `frontend/nginx.conf` — API proxy, WebSocket proxy, SPA fallback, asset caching
- 10 stub API endpoint files + frontend scaffold (App, layout, panel stubs, stores)

### Phase 2: Data Loading Pipeline

**Objective:** Load h5ad files, serve data as Arrow IPC binary to the frontend.

**Key implementations:**
- `anndata_adaptor.py` (233 lines) — Lazy-loads AnnData, extracts embeddings from `obsm`, metadata from `obs`, expression from `X` (sparse-aware), marker genes from `uns['rank_genes_groups']`, pseudotime detection. Backed mode for files >2GB.
- `arrow_serializer.py` (117 lines) — Converts numpy arrays and pandas DataFrames to Arrow IPC stream format. Functions: `embedding_to_arrow_ipc()`, `expression_to_arrow_ipc()`, `dataframe_to_arrow_ipc()`, `series_to_arrow_ipc()`.
- `dataset_manager.py` (151 lines) — LRU cache (max 3 datasets), auto-populates metadata JSON sidecar with cell/gene counts and available embeddings on first load.
- `conversion.py` — Orchestrates Seurat→h5ad via HTTP POST to the R converter service.
- Updated API endpoints: `embeddings.py` (Arrow IPC binary responses with optional color_by), `metadata.py` (column info, summaries, Arrow column values), `expression.py` (Arrow IPC expression, violin data, gene autocomplete).

### Phase 2b: Data Assessment & AI Advisor

**Objective:** Auto-detect preprocessing state, run missing steps, LLM-powered suggestions.

**Key implementations:**
- `assessor.py` (~550 lines) — Inspects 12 preprocessing steps: QC metrics, filtering, normalization, log transform, HVGs, scaling, PCA, neighbors, clustering, embeddings, markers, cell cycle. Each returns a `StepStatus` with `done`, `confidence` (high/medium/low), and `details`. Handles sparse matrices, missing columns, and edge cases gracefully.
- `pipeline.py` (~300 lines) — Runs scanpy preprocessing steps in canonical dependency order. Each step (e.g., `sc.pp.normalize_total`, `sc.tl.pca`, `sc.tl.leiden`) is a separate function with configurable parameters via `PipelineParams`. Saves results as a new version (`_v2.h5ad`), preserving the original.
- `llm_advisor.py` (~540 lines) — DeepInfra integration using OpenAI-compatible client with `meta-llama/Meta-Llama-3.1-8B-Instruct`. Sends structured preprocessing state as context, parses LLM response into `LLMSuggestion` objects. Falls back to `get_rule_based_suggestions()` when no API key is configured — deterministic best-practice recommendations based on dataset size.
- `assessment.py` (API) — GET assessment state, POST to run pipeline steps, POST for LLM/rule-based suggestions.
- `DataAssessmentPanel.tsx` (~475 lines) — Visual pipeline stepper with green/yellow/gray status indicators, confidence badges, expandable per-step details, "Run Missing Steps" button, and "Get AI Suggestions" button.

### Phase 3: Core Visualization

**Objective:** WebGL scatter plot with 200k cells at 60fps, Arrow decoding, lasso selection.

**Key implementations:**
- `arrowDecoder.worker.ts` (75 lines) — Web Worker that decodes Arrow IPC binary, extracts Float32Array/Int32Array columns, transfers via Transferable objects (zero-copy).
- `useEmbedding.ts` (159 lines) — React hook orchestrating: React Query fetch → Worker decode → interleaved positions array + color array.
- `EmbeddingScatter.tsx` (320 lines) — deck.gl `DeckGL` with `OrthographicView`, `ScatterplotLayer` with `radiusUnits: 'pixels'`, auto-fit initial view, precomputed RGBA color buffer, selection dimming, hover tooltip, resize observer.
- `PlotControls.tsx` (193 lines) — Embedding selector, color-by dropdown (categorical-first sorting), point size/opacity sliders, background toggle, lasso toggle, reset view.
- `ColorLegend.tsx` (106 lines) — Categorical (clickable colored circles) and continuous (CSS gradient bar with viridis).
- `LassoSelector.tsx` (153 lines) — SVG overlay for freeform polygon drawing, screen-to-data coordinate conversion, ray-casting point-in-polygon test.
- `OverviewPanel.tsx` (302 lines) — Composes scatter + controls + legend with 70/30 layout, keyboard shortcuts (L for lasso, Escape to clear), selection bar, dataset summary card.

### Phase 4: Analysis Panels

**Objective:** Gene expression, samples, clusters, marker genes.

**Key implementations:**
- `GeneExpressionPanel.tsx` (315 lines) — Gene search with autocomplete, embedding scatter colored by expression (continuous), Plotly violin plot split by group, group-by selector.
- `SamplesPanel.tsx` (366 lines) — Auto-detects sample columns, stacked bar chart of cluster composition per sample, summary table with inline distribution bars.
- `ClustersPanel.tsx` (370 lines) — Auto-detects cluster columns, summary table with click-to-highlight on embedding, stacked bar chart of sample distribution per cluster.
- `MarkerGenesPanel.tsx` (443 lines) — Fetches marker genes as JSON, sortable/filterable table with logFC coloring, significance highlighting, CSV export, "View" button linking to Gene Expression panel.
- `GeneSearch.tsx` (223 lines) — Reusable autocomplete with debounced search, keyboard navigation, click-outside-to-close.
- `ViolinPlot.tsx` (92 lines) — Reusable Plotly violin wrapper with categorical colors and box plot overlay.

### Phase 5: Advanced Features

**Objective:** Gene sets, enrichment, trajectory, export.

**Key implementations:**

Backend:
- `genesets.py` — MSigDB collection listing, gene set search (stub), gene set scoring via `sc.tl.score_genes()`.
- `enrichment.py` — Read pre-computed enrichment from `adata.uns`, compute via gseapy.enrich() with marker genes, group listing.
- `trajectory.py` — Pseudotime column detection, per-cell values, gene expression along pseudotime with sub-sampling and binned smoothing.
- `export.py` — CSV/Excel export of markers, metadata, or expression as file download.

Frontend:
- `GeneSetPanel.tsx` — Manual gene set input textarea, scoring endpoint call, embedding scatter colored by score, violin plot by group.
- `EnrichmentPanel.tsx` — Group selector, compute button, horizontal bar chart of top enriched terms, results table with sorting, CSV export.
- `TrajectoryPanel.tsx` — Pseudotime column selector, embedding colored by pseudotime, multi-gene expression-along-pseudotime plot with smoothed lines and raw scatter.
- `ExportMenu.tsx` — Reusable dropdown with CSV/Excel/PNG export buttons.

### Final Statistics

| Category | Files | Lines of Code |
|----------|-------|---------------|
| Python (backend + tests) | 31 | ~3,400 |
| TypeScript/TSX (frontend) | 41 | ~5,900 |
| R (converter) | 3 | ~510 |
| Config/Docker/Build | 19 | ~600 |
| **Total** | **94** | **~10,400** |

### What Remains (Phase 6)

- Visual polish: Inter font, panel transition animations, skeleton loading states, glassmorphism on floating controls
- Performance: React.lazy for panel code splitting, React.memo for expensive renders, deck.gl DataFilterExtension
- Error handling: ErrorBoundary per panel, corrupt file detection, network retry with backoff
- Testing: pytest + httpx (backend), Vitest + RTL (frontend), Playwright (E2E)
- Documentation: README with screenshots (after UI is running)
- Git initialization and first commit
