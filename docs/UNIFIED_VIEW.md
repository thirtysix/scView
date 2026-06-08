# Toward a Kana-like Unified View — design (2026-05-29)

**Goal:** make a single-screen, linked-view workspace the *primary* analysis surface
in scView — the way [Kana](https://www.kanaverse.org/kana/) presents an embedding, a
marker table, and diagnostic plots together and keeps them interactively linked.

## What Kana does (and why it's the model)
From the Kana paper/app ([biorxiv](https://www.biorxiv.org/content/10.1101/2022.03.02.482701v2.full.pdf),
[github](https://github.com/kanaverse/kana)):
- **One screen, three linked regions:** top-left = low-dimensional **embedding**;
  right = **marker table for the selected cluster**; bottom-left = **gallery of
  diagnostic plots** (QC, PCA variance, etc.).
- **One-click workflow** from count matrix → QC → normalization → PCA → clustering →
  UMAP/t-SNE → marker detection, then drop you into that explorable screen.
- **Linked interactivity:** pick a cluster → its markers; click a gene → recolor the
  embedding; iterate by re-running steps with new parameters.
- Compute runs **in-browser via WebAssembly** (no backend; data stays local).

scView's tradeoff is different and fine: it keeps a **Python/scanpy backend** (richer
algorithms, big shared MSigDB, LLM assist) and streams Arrow to a deck.gl frontend. We
adopt Kana's *UX model*, not its WASM compute.

## Where scView already is
`frontend/src/components/panels/UnifiedViewPanel.tsx` (+ `stores/unifiedViewStore.ts`,
`components/unified/Unified*Subtab.tsx`) is **already a Kana-inspired workspace**: a
resizable split pane with the deck.gl scatter on the left, a tabbed analysis panel on the
right (Markers / Expression / Gene Sets / Enrichment), and a collapsible violin
("Distribution") along the bottom. This is a strong foundation — the work is to finish,
fix, and promote it rather than build anew.

Gaps today (from a 2026-05-29 UX audit):
1. **🔴 The Unified View scatter renders blank** while the standalone Visualizations panel
   renders the same UMAP. Same `useEmbedding()` hook and same `EmbeddingScatter` props, so
   it's a deck.gl rendering issue in the Unified DOM context — prime suspects: the
   Orthographic `viewState` not re-fitting to data inside the split pane, or a WebGL
   context handoff when switching from the Overview deck.gl instance. **This is the #1
   prerequisite — the unified view is unusable until it renders.** Debug by logging
   `containerSize`, `viewState`, and `positions?.length` live in the Unified mount, and by
   loading the Unified View *without* visiting Visualizations first (to test the
   context-handoff theory).
2. It's **one tab among ten** in the sidebar, not the default surface.
3. Linked interactions are **partial** (overlay state exists; cluster-select → markers and
   lasso → on-the-fly markers are not wired).

## Proposed direction (phased)

**U1 — Make it work and make it primary**
- Fix the blank scatter (#1 above).
- After a dataset loads, **land in the Unified View by default** (currently → Overview /
  Data Assessment). Keep Visualizations/Expression/etc. as deep-dive panels.
- Treat the unified right-hand tabs as the canonical home for Markers / Expression /
  Gene Sets / Enrichment (they already exist as `Unified*Subtab`), shrinking sidebar sprawl.

**U2 — Linked views (Kana parity)**
- ✅ **Gene click → recolor + violin (2026-06-01):** clicking a marker row or an expression
  hit recolors the embedding by that gene (viridis + colorbar) and updates the Distribution
  violin. Was already wired in `unifiedViewStore`; unblocked by the blank-scatter fix +
  a Markers-subtab race fix (markers now auto-populate for the active clustering, so
  marker-gene clicks recolor too).
- ✅ **Cluster-map click-to-select + linked camera (2026-06-06):** clicking a point in the
  cluster reference map highlights that cluster (dims the rest) like its legend chip
  (`EmbeddingScatter` opt-in `onClick`, deck.gl click-not-drag). The cluster map and the main
  plot also **share one camera** (bidirectional, 3D OrbitView + 2D), each keeping its own zoom.
  *Next:* make the click also drive the Markers subtab group (`/markers?groupby_column=…`).
- **Lasso → on-the-fly markers:** Kana's signature — lasso a region, compute markers for
  the selection vs rest. Lasso selection already exists; wire it to a markers call.
- ✅ **Tab-switch persistence + expression-units selector (2026-06-05/06):** the gene/score
  overlay + violin persist across panel switches (lifted into `unifiedViewStore`); a **Layer**
  dropdown in the toolbar sets the expression units for the overlay and violin.

**U3 — One-click workflow + diagnostics gallery**
- A prominent **"Analyze"** action that runs the Data Assessment pipeline
  (QC→norm→HVG→PCA→neighbors→clustering→UMAP→markers) and drops the user into the unified
  view — Kana's one-click path. The pipeline + immutable `derived/` layer already exist.
- A **bottom diagnostic gallery** (QC violins, PCA variance, n-genes/counts) mirroring
  Kana's bottom-left — generalising today's single "Distribution" strip.

**U4 — Iterative re-runs in place**
- Change clustering resolution / embedding params from the unified view and re-run,
  updating the embedding + markers without leaving the screen. Safe because pipeline
  output goes to the regenerable `derived/` layer (originals stay immutable).

## Why this is low-risk
The unified view is a **frontend composition over endpoints that already exist**
(embeddings, markers, expression, genesets, enrichment, assessment pipeline). No backend
redesign. The one true blocker is the deck.gl blank-scatter bug; everything else is
incremental wiring of interactions scView can already serve.

## Suggested order of work
1. Fix the blank scatter (U1) — unblocks everything.
2. Make Unified the default post-load panel (U1) — cheap, high impact.
3. Cluster-select → markers and gene-click → recolor (U2) — the core Kana feel.
4. One-click Analyze (U3), then lasso-markers and the diagnostics gallery.
