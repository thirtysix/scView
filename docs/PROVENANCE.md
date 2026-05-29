# Data provenance & history — design (2026-05-29)

**The problem scView exists to solve, applied to scView itself:** a user opens an
h5ad they inherited, downloaded, or made long ago and can't remember *what's been
done to it*. scView must (a) tell them what it can, for **any** file, and (b) never
become a new source of forgotten transformations — every operation scView performs
must be recorded *in the data* so the next person (or future-you) sees exactly what
happened.

## Two sources of truth, combined
1. **Recorded history (authoritative, new):** a replayable log written into the
   h5ad whenever scView transforms it. Travels with the file; reproducible; honest.
2. **Inferred state (heuristic, already exists):** `core/assessor.py` already
   *infers* what's been done from the data itself (QC columns present? X looks
   normalized? `X_pca` in obsm? clustered?) — this powers the Data Assessment panel.

For a **scView-touched** file we show the exact recipe (1). For an **inherited /
foreign** file with no scView history, we still show inferred state (2). Displaying
both — "recorded" vs "inferred (and do they agree?)" — is the whole answer.

## Where it lives: one evolving file + internal history (not many files)

**Decision: a single canonical h5ad per dataset, with an append-only history inside
`uns['scview']`. Do NOT version by filename.**

Why:
- The **original is already immutable** (`uploads/`), and pipeline output goes to the
  regenerable `derived/` layer. So "start over" is always possible without snapshots.
- The history is a **replayable recipe** (each entry = tool + params), so any
  intermediate state can be *reconstructed* by replaying `history[:k]` on the
  original — no need to keep GB-scale intermediate files.
- Filenames encoding state (`pbmc.norm.pca.leiden.h5ad`) get unwieldy, break
  references, and bloat disk/Dropbox. **State belongs inside the file, not the name.**

**File naming:** keep the stable `derived/{id}/{stem}.h5ad` working file (name never
encodes state). Generate a descriptive name only on **export/download**
(e.g. `pbmc_1k_v3.scview.h5ad`), with the full history embedded.

**Intermediate snapshots:** none by default. Offer instead:
- **Reset to original** — re-derive from `uploads/`.
- **Replay to step k** — re-run `history[:k]` from the original (enabled by the recipe).
- **Named checkpoints** (optional, later) — only when a user explicitly wants to
  branch/compare; opt-in so disk bloat is a deliberate choice.

## Structure: `uns['scview']`

```jsonc
uns['scview'] = {
  "schema_version": 1,
  "source": {
    "origin": "ingested" | "uploaded" | "converted",
    "original_filename": "pbmc.rds",
    "format": "10x_mex" | "10x_h5" | "h5ad" | "loom" | "dense_csv" | "seurat_rds",
    "ingested_at": "2026-05-29T15:30:00Z",
    "n_cells": 1222, "n_genes": 33538,
    "merged_from": [                      // only for merges
      {"sample": "GSM4711", "n_cells": 1200, "n_genes": 33000}
    ],
    "merge": {"join": "inner", "identifier_basis": "ensembl", "reconciled": true}
  },
  "history": [                            // ordered, append-only, replayable
    {
      "step": "normalization",
      "tool": "scanpy.pp.normalize_total",
      "params": {"target_sum": 1e4},
      "timestamp": "2026-05-29T15:31:02Z",
      "scview_version": "0.1.0",
      "effect": {"n_cells": 1222, "n_genes": 33538},  // state AFTER the step
      "note": null                        // optional human annotation
    }
    // … qc_metrics, log_transform, pca, clustering(res=0.5), umap, markers(SubclusterID) …
  ],
  "current": {                            // denormalized summary for fast display
    "qc": true, "normalized": true, "log1p": true, "scaled": false,
    "pca": true,
    "clustering": {"method": "leiden", "resolution": 0.5, "column": "scview_leiden_r0.5"},
    "embeddings": ["X_pca", "X_umap"],
    "markers_for": ["SubclusterID"],
    "enrichment_for": ["SubclusterID"]
  }
}
```

Design choices that maximise usable information:
- **`history` is the recipe** — `step` + `tool` + `params` make each entry both a
  human sentence ("Normalized to 10,000 counts/cell") *and* reproducible.
- **`effect` per step** records the *impact* (e.g. filtering: 1500 → 1222 cells) so
  users see consequences, not just actions.
- **`source`** answers "where did this come from?" — critical for inherited files and
  for merges (which samples, what join, was identifier reconciliation applied).
- **`current`** is a cheap denormalized snapshot so the UI renders instantly without
  replaying history — and is **reconciled against the real adata** on load (does
  `X_umap` actually exist? is the clustering column present?) to flag files edited
  outside scView. "Recorded vs actual" honesty.
- **`schema_version`** for forward-compatibility; everything namespaced under
  `scview` to avoid colliding with other tools' `uns`.
- Written **defensively** — a malformed `uns['scview']` must never break loading.

## How it's shown in scView (on load)
- A **Provenance** banner/panel: a plain-language timeline — *"pbmc_1k_v3 · originally
  a 10x MEX of 1,222 cells. scView has run: QC ✓ · normalize (10k) ✓ · log1p ✓ · PCA ✓
  · Leiden clustering (res 0.5) ✓ · UMAP ✓ · markers for SubclusterID ✓"* with
  timestamps and per-step effects.
- For files with **no scView history**: *"No scView history found — here's what we can
  infer from the file"* + the assessor's inferred state (reusing Data Assessment).
- **Reconciliation flags** when recorded `current` disagrees with the actual data.
- **Export the recipe** (history as JSON) so a method can be reproduced or re-applied.

## Implementation sketch (where the hooks go)
- `pipeline.py`: after each step, append a history entry to `adata.uns['scview']`
  (the step list + `PipelineParams` already carry everything needed). Update `current`.
- Ingestion (`core/ingestion`): write the initial `source` block + first history
  entries (format, transpose, var_names reset, merge) at commit time.
- A small `core/provenance.py` helper: `record_step()`, `init_source()`,
  `read_provenance(adata)`, `reconcile(adata)` — keeps `uns` writes in one place and
  defensive.
- Backend: expose provenance via the dataset detail endpoint (or a dedicated
  `/datasets/{id}/provenance`), merging recorded history with the assessor's inference.
- Frontend: a Provenance panel/banner (consumes the above); reuse Data Assessment's
  inference rendering for the no-history case.

## Net
One file, one growing internal log that is *also* a reproducible recipe; names stay
stable; intermediate states are replayable rather than stored; and on load scView
always answers "what's been done?" — from its own records when present, from inference
otherwise. That closes the loop the tool is meant to close, including for scView's own
outputs.
