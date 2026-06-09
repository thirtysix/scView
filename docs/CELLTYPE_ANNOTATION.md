# Cell-type annotation — design & options

*Written 2026-06-08. Goal: add cell-type annotation that **plugs into scView's existing Docker
backend** (Python/scanpy, CPU, no GPU) **without a heavy processing environment**. Benchmarks below
were measured in the running backend container on the Kang IFN-β PBMC hero dataset (13,836 cells).*

## Status (as built, 2026-06-09)
- **CellTypist method shipped**: pipeline step `cell_type_annotation` (`PipelineParams.annotation_*`),
  registered in `ALL_STEPS` / `_STEP_RUNNERS` / `_STEP_PROVENANCE`. Builds CellTypist input from raw
  counts (normalize to 1e4 + log1p), runs majority-voting consensus over the existing clustering
  (`over_clustering=groupby`), and writes `obs['cell_type']` (+ `cell_type_percell`,
  `cell_type_confidence`); the method + model are recorded in provenance. `celltypist` is in
  `backend/pyproject.toml` (rebuild the backend image to bake it in).
- **Model picker endpoint**: `GET /api/v1/annotation/celltypist-models` returns the catalog (61
  models, name + description) and the default. **Choosing a model:** CellTypist models are
  tissue/system specific — there is *no* universal model — so the UI should offer a **dropdown** of
  this catalog, defaulting to **`Immune_All_Low`** (the common PBMC/immune case, and what we
  benchmarked). The user picks the model matching their sample's tissue; a mismatch mislabels (e.g.
  on the immune Kang set, erythrocytes/megakaryocytes get myeloid/T calls). `Immune_All_High` is the
  coarse-grained immune alternative.
- **LLM-from-markers shipped** (`method="llm"`): names each cluster's cell type from its top marker
  genes via the DeepInfra LLM — **no reference model to pick, any tissue**. On Kang it matched the
  authors' labels and *beat* CellTypist on the non-immune clusters (**Erythroblast**, **Megakaryocyte**)
  that the immune model mislabeled. Optional `annotation_tissue` hint; writes `obs[target]` + the
  per-cluster mapping to `uns[<target>_llm_mapping]`. Reuses precomputed markers when present, else
  computes them. Requires `DEEPINFRA_API_KEY`; non-deterministic, so present as a reviewable pass.
  *(The earlier "8B is noisier" caveat was mitigated here by a tissue hint + a single structured
  per-cluster prompt.)*
- **Pending**: the frontend annotation control (method choice + CellTypist model picker + "Annotate"
  button on Data Assessment / Unified View), and the offline `marker_score` method.

## TL;DR
Ship a small **tiered annotator** behind one "Annotate cell types" step — pick a method, sensible
default:

1. **CellTypist** *(default; reference-based)* — pretrained logistic-regression models, **CPU,
   pip-installable, ~nil new deps** (all its requirements already ship in scView), **2.8 MB** of
   models. Best accuracy/footprint. Strongest recovery of major lineages on Kang.
2. **AI (LLM-from-markers)** *(any-tissue, AI-native)* — annotate clusters from their top markers via
   the **DeepInfra** LLM scView already uses. **Zero local footprint**, works for any tissue, and is
   the natural piece for the co-pilot and the **"Auto-analyze" autopilot**. Non-deterministic →
   present as reviewable.
3. **Marker-score (decoupler + PanglaoDB/CellMarker)** *(offline fallback)* — deterministic,
   interpretable, **no model download / no network**.

All three run **per-cluster on the user's existing clustering**, write an `obs['cell_type']` column,
**record method+model in provenance**, and surface in the co-pilot grounding + the Unified-View
grouping selector. This also **completes the "Auto-analyze" feature** (annotation was its missing piece).

## Constraints
scView's backend is Python + scanpy in a Docker image, CPU-only, "not a huge processing
environment." So: prefer pip-installable, scanpy/AnnData-native, small models, no GPU, no large
reference datasets.

## Landscape — what fits vs what to avoid

| Option | New footprint | Speed (14k cells) | Determinism | Fit |
|---|---|---|---|---|
| **CellTypist** | deps already in scView + **2.8 MB** models | ~46 s w/ consensus voting; **seconds** per-cell | deterministic | ⭐ default |
| **LLM-from-markers** (GPTCelltype-style) | **zero local** (reuses DeepInfra) | ~seconds (one call/cluster) | non-deterministic | ⭐ AI-native / any tissue |
| **decoupler** + marker DB (PanglaoDB/CellMarker) | pip-light, **no model DL** | seconds | deterministic | offline fallback |
| scanpy `score_genes` + curated markers | **zero new deps** | instant | deterministic | crude baseline (infra already present) |

**Avoid (too heavy for the constraint):** scANVI/scvi-tools (pulls PyTorch, ~GB), Azimuth / SingleR
(R + large reference datasets), foundation models scGPT/Geneformer (GPU). Overkill here.

## Measured benchmarks (Kang IFN-β PBMC, 13,836 cells, in-container, CPU)

### CellTypist 1.7.1 (`Immune_All_Low.pkl`)
- **Install:** trivial — requirements (`scanpy, scikit-learn, leidenalg, numpy, pandas, openpyxl,
  click, requests`) already satisfied by scView; only the small `celltypist` package is new.
- **Model cache:** **2.8 MB** total (download 1.1 s).
- **Runtime:** 45.7 s for 13,836 cells *with* majority-voting (the cost is its internal
  over-clustering; per-cell prediction alone is seconds — and we can pass scView's existing clusters
  as `over_clustering` to skip it entirely).
- **Concordance with the authors' labels** (author cluster → CellTypist majority label):
  B → **B cells (98%)**, B activated → B cells (98%), CD16 Mono → **Non-classical monocytes (98%)**,
  pDC → **pDC (95%)**, CD14 Mono → Intermediate macrophages (94%, myeloid), NK → **CD16+ NK cells
  (91%)**, T activated → Tcm/Naive helper T (66%), CD4 Naive → **Tcm/Naive helper T (60%)**, Eryth →
  **Late erythroid (60%)**, DC → Intermediate macrophages (58%), Mk → Tcm/Naive helper T (56%, miss),
  CD4 Memory → Tem/Effector helper T (51%), CD8 T → **Tem/Trm cytotoxic T (39%)**.
  Major lineages recovered; misses are the expected ones (rare Mk; fine T-subtype splits) — i.e. a
  good *reviewable* first pass, not ground truth.

### LLM-from-markers (DeepInfra, Llama-3.1-8B, top-10 markers/cluster)
Zero local footprint; recovers many (B → B cells, CD4 Memory → CD4+ T, CD4 Naive → Naive T, CD8 T →
CD8 T, CD16 Mono → Monocytes, DC → Dendritic cells, Eryth → Erythrocytes) but noisier than CellTypist
with an 8B model (Mk → Neutrophils, NK → Cytotoxic T, pDC → Cytotoxic T). The GPTCelltype paper used
GPT-4; accuracy scales with model size — offer a "stronger model" toggle.

## Integration design

A new pipeline step `cell_type_annotation`:
- **params:** `method` ∈ {`celltypist`, `llm`, `marker_score`}; `celltypist_model` (e.g.
  `Immune_All_Low`); `llm_model` (DeepInfra id); `markerset` (PanglaoDB/CellMarker); `groupby`
  (defaults to the active clustering); `target` obs column (default `cell_type`).
- **behaviour:** annotate **per the user's existing clusters** (consensus), so labels align to the
  clustering and CellTypist skips its own re-clustering (`over_clustering=adata.obs[groupby]`).
  CellTypist input must be log1p-normalised to 1e4 — build it from `adata.raw` counts on the fly.
- **outputs:** `obs['cell_type']` (+ a per-cluster confidence/score), recorded into
  `uns['scview_provenance']` (method + model). Surfaced in the co-pilot grounding (`result:groups`),
  the Unified-View grouping/colour selector, and exportable.
- **ties:** completes the **Auto-analyze** autopilot; the co-pilot can then answer in named cell
  types and explain the annotation; users can **edit/override** a label and re-run (recorded).

### Docker / deps impact: negligible
Add `celltypist` (and optionally `decoupler`) to `backend/pyproject.toml` — deps already satisfied, so
the image barely grows. Bundle a few MB of CellTypist models at build, or download-on-demand into a
volume. LLM path needs no new deps.

## Caveats (important for the AI-in-medicine framing)
- **Reviewable first pass, not truth** — show confidence, let users edit/override (and re-run records it).
- **Model–tissue match matters** — Immune model for immune data, pan-tissue otherwise; the step should
  warn/guide. (On Kang, the immune model is appropriate.)
- **No clinical/diagnostic claims** from automated annotation.

## References
- CellTypist — Domínguez Conde *et al.*, *Science* 2022; https://www.celltypist.org · https://pypi.org/project/celltypist/
- GPTCelltype / GPT-4 annotation — Hou & Ji, *Nature Methods* 2024; https://www.nature.com/articles/s41592-024-02235-4
- decoupler — Badia-i-Mompel *et al.*, 2022; PanglaoDB / CellMarker marker databases.
- See also `docs/AI_ASSISTANT.md` (co-pilot + Auto-analyze) and `docs/FUTURE.md`.
