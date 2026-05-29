# scView Technical Decisions Log

This document records the rationale behind key architectural and technology choices.

---

## D-001: Docker Compose with 3 Separate Services

**Decision:** Use three separate Docker containers (Python backend, Node/Nginx frontend, R converter) rather than a monolithic container.

**Alternatives considered:**
1. Single container with Python + R + Node — bloated (5GB+), slow builds, fragile dependency conflicts
2. Two containers (Python+R backend, Node frontend) — still heavy, rpy2 is fragile and version-sensitive
3. Three containers (chosen) — clean separation, each ~500MB-1.5GB, independent caching

**Rationale:** The R converter is only needed during Seurat import. Isolating it means the Python backend stays lean (~1GB with scanpy), builds are faster (Docker layer caching per service), and the R container can be skipped entirely when only using .h5ad files.

---

## D-002: Apache Arrow IPC for Data Transfer

**Decision:** Use Apache Arrow IPC (binary) instead of JSON for embedding coordinates and expression data.

**Alternatives considered:**
1. JSON — simple but 10x slower for 200k float32 values, plus JSON parse overhead
2. MessagePack — faster than JSON but still requires deserialization
3. Protocol Buffers — good performance but adds schema complexity
4. Arrow IPC (chosen) — native typed array output, near-zero parse cost in JS

**Rationale:** For a 200k-cell UMAP, JSON would be ~6MB and take ~200ms to parse. Arrow IPC is ~1.6MB and decodes to typed arrays in ~5ms. deck.gl accepts typed arrays directly, so the decode-to-render pipeline is essentially zero-copy. The `apache-arrow` JS library is well-maintained and the `pyarrow` Python library is already a dependency of anndata.

---

## D-003: deck.gl with OrthographicView for Scatter Plots

**Decision:** Use deck.gl's `ScatterplotLayer` with `OrthographicView` rather than alternatives.

**Alternatives considered:**
1. regl (raw WebGL) — maximum performance, used by cellxgene, but much more code to write
2. Three.js — powerful but overkill for 2D scatter; geo-oriented
3. Plotly WebGL scatter — easy but limited interactivity (no lasso, limited hover)
4. deck.gl (chosen) — high-level API with excellent typed array support, OrthographicView for 2D

**Rationale:** deck.gl provides the best balance of performance and developer productivity. `OrthographicView` avoids the Mercator projection issues of the default MapView. `ScatterplotLayer` with `radiusUnits: 'pixels'` gives pixel-perfect control. The picking system enables hover tooltips and lasso selection without custom WebGL code. Performance target: 200k cells at 60fps.

---

## D-004: Separate R Container via sceasy for Seurat Conversion

**Decision:** Convert Seurat objects to h5ad using sceasy in a dedicated R container, rather than using rpy2 in Python.

**Alternatives considered:**
1. rpy2 in the Python backend — fragile R/Python bridge, version conflicts, bloated image
2. SeuratDisk — depends on hdf5r, has known issues with Seurat v5
3. sceasy (chosen) — uses reticulate to bridge R→Python→anndata, well-tested on v3/v4

**Rationale:** sceasy handles the R→AnnData conversion in pure R with a reticulate bridge to Python's anndata. By isolating this in a separate container, we avoid polluting the Python backend with R dependencies. The container exposes a simple HTTP API (`POST /convert`), making it easy to replace with a different converter later.

**Seurat v5 note:** Seurat v5 changed its internal structure (layers instead of slots). The converter includes detection logic for v3/v4/v5 and a manual fallback that extracts matrices, metadata, and reductions directly when sceasy fails.

---

## D-005: Zustand + React Query for State Management

**Decision:** Use Zustand for UI state and React Query for server state.

**Alternatives considered:**
1. Redux + RTK Query — powerful but heavy boilerplate for this app size
2. Zustand + SWR — SWR is simpler but React Query has better mutation/cache control
3. Zustand + React Query (chosen) — minimal boilerplate, hooks-native

**Rationale:** scView has two distinct categories of state:
- **UI state** (active panel, point size, selected cells, sidebar collapsed) — changes locally, no server round-trip → Zustand
- **Server state** (embedding data, gene expression, metadata) — fetched from API, needs caching and refetch logic → React Query

This separation prevents the anti-pattern of putting API responses into global stores.

---

## D-006: Web Workers for Arrow Decoding and Color Mapping

**Decision:** Decode Arrow IPC binary and compute color arrays in Web Workers.

**Rationale:** For 200k cells, Arrow decoding takes ~5ms and color mapping (applying a palette to 200k values) takes ~10ms. While these are fast, they block the main thread during a critical render path. Running them in Web Workers keeps the UI responsive during data transitions. The Worker communicates via `Transferable` objects (ArrayBuffer transfer), so the typed arrays move to the main thread without copying.

---

## D-007: Data Assessment as a Core Differentiator

**Decision:** Build an intelligent data assessment system that auto-detects preprocessing state and offers to run missing steps.

**Rationale:** Most existing tools (cellxgene, Vitessce) assume data is fully preprocessed. scView's assessment panel fills a gap: users upload raw or partially processed data, see a visual checklist of what's done/missing, and can run the full scanpy pipeline from the browser. The optional LLM advisor (DeepInfra) adds parameter suggestions — but the system works without it via rule-based fallback.

**Pipeline versioning:** When the pipeline modifies data (filtering reduces cell count, normalization changes expression values), it saves as a new version (`dataset_v2.h5ad`) rather than overwriting. The original is always preserved.

---

## D-008: DeepInfra for LLM Integration

**Decision:** Use DeepInfra's OpenAI-compatible API with Llama 3.1 8B for analysis suggestions.

**Alternatives considered:**
1. OpenAI GPT-4 — excellent quality but expensive for a local tool
2. Local Ollama — no API key needed but requires user to install Ollama
3. Anthropic Claude API — excellent but overkill for structured parameter suggestions
4. DeepInfra Llama 3.1 8B (chosen) — cheap (~$0.001/query), fast, sufficient for structured suggestions

**Rationale:** The LLM advisor's job is structured: given a preprocessing state summary, suggest parameters and next steps. This doesn't require frontier model capability. Llama 3.1 8B via DeepInfra costs fractions of a cent per query and responds in <2 seconds. The feature is entirely optional — the rule-based fallback provides the same suggestions without an API key.

---

## D-009: Tailwind CSS + shadcn/ui for Styling

**Decision:** Use Tailwind CSS v4 with shadcn/ui component primitives.

**Rationale:** Tailwind provides utility-first CSS that's fast to write and produces small bundles. shadcn/ui (not a component library — it's copy-paste components) gives accessible, well-designed primitives (buttons, cards, dialogs) that can be fully customized. The combination produces a clean, modern UI without the weight of a full component framework like Material UI or Ant Design.

**Design choices:**
- Dark sidebar (`slate-900`) with light content area
- Teal accent (`#0D9488`) — evokes scientific rigor without being sterile
- Inter font — clean, excellent at small sizes for data labels
