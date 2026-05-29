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

## Undo / branching without staged files

The fair objection to "one evolving file": if a user wants to **undo a choice** (say,
re-cluster at a different resolution), a staged-files workflow lets them grab the
file from one step prior and proceed. With one file they'd seemingly have to replay
*everything* from the original. We address this so that, in practice, **you rarely
replay from the original and you only recompute what actually changed.**

### Why a full replay is almost never needed
Changing a step only invalidates the steps **downstream** of it; upstream steps stay
valid. And the expensive upstream artifacts are **already materialised in the current
file**:
- `obsm['X_pca']`, the neighbour graph (`obsp['connectivities']`/`['distances']`) —
  so changing clustering resolution just re-runs Leiden on the graph that's *already
  there*. No PCA, no neighbours, no replay. (This is also scientifically correct: you
  shouldn't recompute PCA to try a new resolution.)

So the most common "undo" — re-cluster, re-embed, re-detect markers — is a fast,
in-place re-run against artifacts the file already holds.

### Anchor layers — the restore points for *earlier* changes
For changes upstream of PCA (e.g. different normalisation), keep small **anchor
layers** in the file so we restore from the nearest anchor, not the original:
- `layers['counts']` — raw counts: the restore point for the whole
  normalisation/transformation branch (and standard scanpy practice).
- `layers['lognorm']` — log-normalised, pre-scaling: restore point for HVG / scaling /
  PCA (scaling overwrites `X` destructively, so this anchor is what makes that branch
  re-runnable).

Anchors are cheap relative to checkpoints: `X_pca` is `n_cells × ~50`, the graph is
sparse, counts/lognorm are one matrix each — all far smaller than storing a full h5ad
per step.

### "Branch from step k" as a first-class action
Model the pipeline as a small **dependency DAG**
(`normalize → log → hvg → scale → pca → neighbors → {clustering, umap} → markers →
enrichment`). The recipe (history) records the order; the DAG records what depends on
what. When the user edits step *k*'s parameters, scView:
1. computes the **invalidation set** (step *k* + its descendants),
2. restores the **nearest valid anchor** (existing graph for a clustering change; the
   `lognorm`/`counts` layer for an upstream change),
3. re-runs only the invalidated steps with the new params,
4. appends the change to `history` (e.g. *"re-clustered res 0.5 → 0.8; superseded
   markers, enrichment"*), recording random seeds so it's reproducible.

The UI surfaces the scope and cost up front: *"Changing resolution will re-run
clustering, markers, enrichment (~8 s); PCA and the neighbour graph are kept."* Each
completed step in the provenance/assessment view gets an **"edit & re-run from here"**
affordance.

This is **better than manual staged files**: the user doesn't track which file is
which step, scView computes the *minimal* correct re-run automatically (no accidental
PCA recompute), there's one file to manage and share, and the full history is intact.

### When you genuinely want parallel branches
For true side-by-side comparison (res 0.5 *and* 0.8 at once), offer **explicit, named
checkpoints/branches** — opt-in. Even these need not be full h5ad copies: a branch can
be stored as *(recipe up to step k) + the counts/lognorm anchor*, and re-materialised
on demand; a full snapshot is available when instant access is worth the disk. Keeping
this opt-in means the simple path stays one file.

## Net
One file, one growing internal log that is *also* a reproducible recipe; names stay
stable; intermediate states are replayable rather than stored. Undo is handled by
re-running the *minimal* invalidated subset from the nearest in-file anchor — usually
without touching the original — and optional named branches cover true comparison. On
load scView always answers "what's been done?" — from its own records when present,
from inference otherwise. That closes the loop the tool is meant to close, including
for scView's own outputs.
