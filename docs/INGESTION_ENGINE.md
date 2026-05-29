# Ingestion Engine — Design (roadmap)

**Status:** proposed (2026-05-29) · **Goal:** let *any* user — including wet-lab
scientists and PIs with no command-line experience — bring almost any single-cell
RNA-seq dataset into scView, with the tool actively guiding them through multi-file
formats and explaining, in plain language, exactly what to do when something is missing
or wrong.

### Decisions locked (2026-05-29)
- **Phase A covers all supported formats** — 10x MEX (v2/v3), 10x HDF5, dense CSV/TSV,
  Loom, Zarr, h5ad (and Seurat/SCE via the existing R converter). No format is deferred.
- **Excel (`.xlsx`/`.xls`) is NOT accepted** — not a legitimate scRNA-seq interchange
  format. If a user tries one, reject with guidance to export as CSV (see §4 messaging).
- **Multi-sample merge is in Phase A** — combining several 10x samples into one dataset
  with a `sample` label is a day-one wet-lab need. Semantics in §3a below.
- **Merge gene-axis = intersection by default**, union on request — plus active
  **identifier reconciliation** when the intersection is suspiciously low (§3a).
- **Readers run in the Python backend** — scanpy provides every reader natively; the R
  converter stays scoped to Seurat/SingleCellExperiment only.
- **Ambiguous detection → best guess the user can correct** (not block-and-ask), except
  for destructive ambiguity (merge basis, raw-vs-normalized) which always asks.
- **Ingested output lives in a new immutable `ingested/` layer** (parallel to
  `converted/`); processing output still goes to `derived/`.
- **Next:** refine this design before writing code.

Today scView ingests **one file at a time**, accepting only `.h5ad`, `.rds`, `.Rdata`
(`LoadDataPanel.tsx`; converter handles Seurat→h5ad). The common real-world cases —
a 10x **three-file** matrix downloaded from GEO, a CellRanger `.h5`, a plain expression
`.csv` — are unsupported, and there's no guidance when a user gets it wrong.

---

## 1. Format matrix — what we must accept

| Format | What the user has | File(s) | Canonical loader | Notes |
|---|---|---|---|---|
| **AnnData** | `.h5ad` | 1 file | `sc.read_h5ad` | already supported |
| **AnnData (Zarr)** | `.zarr` dir | directory | `sc.read_zarr` | directory upload |
| **10x HDF5** | `filtered_feature_bc_matrix.h5` | 1 file | `sc.read_10x_h5` | CellRanger ≥3; single file, easy |
| **10x MEX (v3)** | matrix + barcodes + features | **3 files** | `sc.read_10x_mtx` | `matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz` |
| **10x MEX (v2)** | matrix + barcodes + genes | **3 files** | `sc.read_10x_mtx` | older: `genes.tsv` (not `features`) |
| **Matrix Market** | bare `.mtx` | 1 + companions | `scipy.io.mmread` | needs barcodes + features to be usable |
| **Dense table** | `.csv` / `.tsv` / `.txt` | 1 (+ optional meta) | `sc.read_csv` / pandas | **orientation ambiguous** (genes×cells?) |
| ~~Excel~~ | `.xlsx` / `.xls` | — | **rejected** | not a valid scRNA-seq format; guide user to export CSV |
| **Loom** | `.loom` | 1 file | `sc.read_loom` | row/col attrs carry gene/cell names |
| **Seurat** | `.rds` / `.RData` / `.h5seurat` | 1 file | R converter → h5ad | already supported (rds/RData) |
| **SingleCellExperiment** | `.rds` | 1 file | R converter → h5ad | extend converter |
| **STARsolo / alevin / bustools** | output dir | MEX-like | `sc.read_10x_mtx` variants | treat as MEX |

The hard cases that justify this whole effort are the **multi-file** ones: 10x MEX and
bare `.mtx`. Everything else is "detect → load".

---

## 2. Architecture — a forgiving pipeline

Four backend stages plus a content library. Each stage emits **structured issues**
(`{severity: info|warn|error, code, message, suggestion, fix_action}`) rather than raw
exceptions, so the frontend can render help instead of a stack trace.

```
uploaded file(s)
   │
   ▼
[1] DETECT      classify each file → FileKind (by extension + content sniff)
   │
   ▼
[2] BUNDLE      group files into an ingest "bundle"; pick the recipe;
   │            report what's present / missing / extra
   ▼
[3] VALIDATE    recipe-specific checks (dims match, orientation, gzip ok…)
   │            → list of IngestIssue
   ▼
[4] LOAD        once complete + valid: read via scanpy/converter → canonical .h5ad
                stored as the dataset (originals stay immutable — see derived/ rule)
```

### [1] Detect — never trust the extension alone
Sniff the first bytes so we catch the classic mistakes:
- HDF5 magic `\x89HDF\r\n\x1a\n` → AnnData or 10x-H5 (disambiguate by inspecting top-level
  keys: `matrix/` group ⇒ 10x H5; `X`,`obs`,`var` ⇒ h5ad).
- gzip magic `\x1f\x8b` → peek inner content.
- MatrixMarket banner `%%MatrixMarket matrix coordinate` → `.mtx`.
- First line `barcode`-like (AAAC…-1) one column ⇒ barcodes file; two/three tab columns
  with Ensembl IDs ⇒ features/genes file.
- Comma/tab-delimited grid with a header row of cell-or-gene names ⇒ dense table.
- Renamed files are common (GEO appends `GSM…_`); detection is content-first so
  `GSM123_barcodes.tsv.gz` and `barcodes.tsv.gz` both resolve.

### [2] Bundle — units vs bundles
Two concepts, kept distinct:
- **Unit** = one experiment/sample. **10x MEX = up to 3 files** (matrix + barcodes +
  features/genes); 10x H5 / h5ad / CSV / loom / zarr = **1 file (or dir)**. So the
  per-unit file maximum is 3.
- **Bundle** = the output of one ingest session: either **a single unit** (one dataset) or
  **several units merged** (one dataset with a `sample` label, §3a).

A 10x MEX unit is **complete** iff it has exactly one of each {matrix, barcodes,
features/genes}. If the user drops just `matrix.mtx.gz`, the unit is *incomplete* and we
know precisely which companions to request.

**Adding samples to merge — one unit at a time (recommended).** The wizard completes and
validates one unit, then offers "Add another sample to merge?". Each experiment is a
discrete, checked step — far more forgiving than dumping 9 loose files and guessing which
barcodes pair with which matrix. **No hard cap** on number of units; instead a **soft,
honest warning** on large merges (see note below).

> **Size warning, not a cap.** scView's render path is deck.gl GPU-instanced scatter,
> which handles 1M+ points easily — *rendering is not the bottleneck*. The real limits, in
> the order they bite: (1) **interaction latency** (lasso hit-testing, hover/picking are
> per-point CPU work) — the first thing to feel sluggish; (2) **browser tab RAM** (~2–4 GB
> ceiling, holding decoded Arrow buffers for embeddings + each materialized expression /
> obs column); (3) **backend RAM** — steady state is cheap (backed mode + per-gene Arrow
> slices), but **merge itself materializes the full matrix**, so a big *merged* dataset is
> the heaviest case. The actual ceiling on this hardware is **not yet measured** (Phase 6
> §4 profiling is pending). So the UI shows a soft message — *"Large merged datasets
> (hundreds of thousands of cells) can make selection and recoloring feel sluggish"* — and
> a real numeric threshold is set only after §4 measures it.

**Bulk drop fallback.** If a user drops a whole GEO folder at once (`GSM1_*`, `GSM2_*`,
`GSM3_*`), auto-group by shared prefix / directory into units and show the detected
grouping **for confirmation** before merging. Same engine, just a convenience path.

### [3] Validate — the friendly-error stage
Recipe-specific, all phrased for a non-technical reader:
- **Dimension agreement:** `matrix` rows = features count, cols = barcodes count (or
  transposed — detect and fix silently, warn if ambiguous).
- **Orientation (dense tables):** decide genes-in-rows vs cells-in-rows by overlap of
  axis labels against a known gene vocabulary; if uncertain, **ask** rather than guess.
- **Missing axis labels (dense tables) — the axes aren't symmetric:**
  - *Cell labels missing* → **auto-generate** `cell_0…cell_N` (warn). Barcodes are
    fungible identifiers; synthetic ones break nothing downstream.
  - *Gene labels missing* → **never auto-generate** — genes are semantic (no names ⇒ no
    markers, no expression lookup, no search). Either accept an optional **sidecar**
    `genes.txt`/`features.txt` (one gene per line, row-aligned — mirrors the MEX model) or
    reject: *"This table's columns are cell barcodes but the rows have no gene names — I
    can't identify genes without labels. Add a `genes.txt` (one gene per line, matching row
    order), or re-export with gene names in the first column."*
- **Gzip integrity / truncated upload.**
- **Duplicate or zero-variance genes, empty barcodes.**
- **Gene identifiers:** Ensembl IDs vs symbols; offer to keep both (`var['gene_ids']`).

### [4] Load — into the immutable-source model
The loader writes a canonical `.h5ad`. It must honour the safety rule established this
session: the **user's uploaded bytes are never modified**. Ingest output is a *new*
canonical file stored under a new **`ingested/{id}/`** layer (parallel to `converted/`,
which stays R-conversion-only), marked read-only. Resolution precedence becomes
**derived > ingested > converted > uploads**; all later processing still writes to
`derived/`. The raw uploaded files (the 10x triplet, the CSV, …) are retained under
`uploads/{id}/` as the untouched original.

---

## 3a. Multi-sample merge (Phase A)

When the bundler detects **>1 complete dataset** (e.g. several 10x triplets with distinct
GEO prefixes, or several `.h5` files dropped together), offer to merge. This is the most
error-prone part of ingestion, so the engine must make the safe choice obvious.

**Merge model** (built on `anndata.concat`):
- **Axis:** concatenate on cells (obs). Each sample's cells are stacked.
- **Cell-name collisions:** 10x barcodes repeat across samples (`AAACCTGAGAAACCAT-1`
  occurs in every run). Prefix each barcode with its sample id (`GSM4711:AAAC…-1`) and set
  `index_unique=":"`, so cells stay globally unique.
- **Sample label:** add an obs column (default name `sample`, user-editable) carrying the
  source id — drives downstream batch/violin/grouping immediately.
- **Gene-axis alignment — the key decision:**
  - **Intersection (`join="inner"`)** — keep only genes present in *all* samples. Safest,
    smaller, no fabricated zeros. **Default**, because mixing gene sets silently is how
    users get burned.
  - **Union (`join="outer"`)** — keep all genes, fill missing with 0. Offered with an
    explicit warning ("genes absent from a sample become zeros, which can look like real
    non-expression"). Only sensible when sets are near-identical (same reference).
  - Surface the overlap up front: *"Sample A has 36,601 genes, Sample B has 33,000;
    32,890 are shared. Keep the 32,890 shared genes (recommended) or all 36,711 (missing
    ones filled with zero)?"*

- **Identifier reconciliation — detect & fix the "tiny intersection" trap:** the most
  common silent disaster is merging a sample whose `var` index is **Ensembl IDs**
  (`ENSG00000160791`) with one indexed by **gene symbols** (`CCR5`). A naive intersection
  is then near-zero and the merge looks broken (or, with union, doubles the gene count
  with two encodings of the same gene). The engine actively guards against this:
  1. After computing the intersection, flag it as **suspicious** if it's **< 50% of the
     smaller sample's var set**. The *smaller* set is the right denominator: a legitimate
     subset (Sample A pre-filtered to 3,000 DEGs, or ~20k protein-coding, vs Sample B's
     60–100k all-genes) still yields intersection ≈ 100% *of the smaller set* and is
     **not** flagged — whereas a basis mismatch collapses the intersection toward 0%.
  2. On a flag, **diagnose the cause** — don't auto-fix:
     - **Identifier mismatch** (different index basis): classify each sample's basis by
       pattern (Ensembl `ENSG…`/`ENSMUSG…` vs symbol). Look **across the other `.var`
       columns** of both samples for a column in the *other's* basis — Sample A (symbols)
       often carries `var['gene_ids']` (Ensembl); Sample B (Ensembl) often carries
       `var['feature_name']`/`var['gene_symbols']`. If a matching basis exists, **propose
       resetting one dataset's `var_names` to that column**, recompute the intersection,
       and show before/after: *"These samples label genes differently (A: symbols,
       B: Ensembl IDs) — only 312 matched. Sample B has a `gene_symbols` column; using it
       lifts the overlap to 31,884. Use gene symbols for the merge?"*
     - **Genuinely different gene sets** (same basis, still low overlap — e.g. one is
       DEG-filtered or coding-only): **not** fixable by reconciliation. Don't reset
       anything; just **inform**: *"Only 3,000 genes are shared because Sample A is
       pre-filtered to 3,000 genes. The merge will keep those 3,000 (intersection) — or
       keep all genes and fill the rest with zeros (union)."*
  3. Prefer Ensembl as the merge basis when both can supply it (robust to symbol aliasing);
     fall back to symbols. **Only reset a `var_names` index when the basis actually
     differs** — never to force a legitimate subset into a union. Index resets are a
     destructive-ambiguity case, so they always ask (per the detection policy).

**Gene vocabulary — patterns + a bundled biomart map.** Two layers, because pattern
inference alone can't rescue the hardest merge:
- **Pattern classification (cheap, first pass):** decide each index's basis from shape —
  `ENSG…`/`ENSMUSG…` ⇒ Ensembl (and species), word-like tokens ⇒ symbols.
- **Bundled biomart mapping (translation + vocabulary):** ship a static Ensembl↔symbol map
  for **human + mouse** (≈60–70k rows each, gzipped ~1–2 MB — same pattern as the mounted
  `msigdb/` reference; **pin the Ensembl release** and record it; handle many-to-one
  aliases). This is what bridges the case patterns can't: Sample A has *only* symbols,
  Sample B has *only* Ensembl IDs, and **neither carries a cross-reference `.var` column** —
  the map translates one basis to the other directly. It also serves as the known-gene
  vocabulary for dense-table orientation detection. Species is chosen from the detected
  `ENSG`/`ENSMUSG` prefix.
- **Raw vs normalized:** only merge comparable layers; refuse to merge a normalized matrix
  with a raw-counts one without a clear warning.
- **Memory guard:** estimate merged size before committing; warn / stream if it would blow
  the per-dataset budget (ties into the 200k-cell target + LRU cache).

Merge is **opt-in** — the default for a multi-dataset drop is still "load separately",
with merge offered as the alternative. A user merging by accident is worse than one extra
click.

## 3. UX — a dedicated "Add Data" tab (wizard)

A new top-level tab, separate from the current single-file `LoadDataPanel`. Flow:

1. **Choose your starting point** — a grid of plain-language format cards:
   - "A single AnnData file (`.h5ad`)"
   - "10x CellRanger output (the `.h5` file, or the 3 matrix files)"
   - "An expression table (`.csv` / Excel)"
   - "A Seurat object (`.rds`)"
   - "**I'm not sure** — let me drop my files and you tell me" ← detection-first path

2. **Drop zone (one experiment at a time)** — accepts **multiple files** and **folders**
   (`webkitdirectory`). As each file lands, show a live **checklist** for the current unit:
   ```
   Sample 1 — 10x MEX
     ✓  matrix.mtx.gz        (expression matrix · 33 k × 5 k)
     ✓  barcodes.tsv.gz      (5 000 cell IDs)
     ✗  features.tsv.gz      ← still needed
   ```
   When the unit is complete, offer **"Add another sample to merge"** (→ a new unit) or
   **"Continue"**. A bulk folder drop is auto-grouped by prefix into multiple units, shown
   for confirmation. Multiple units ⇒ the merge step (§3a) appears before confirm.

3. **Validation summary** — issues rendered as cards with a suggested fix and, where
   possible, a one-click action (toggle orientation, drop duplicate, choose gene-id col).

4. **Confirm & name** — orientation toggle for tables, gene-id vs symbol choice, dataset
   name. Show a 5-row × 5-col preview of the parsed matrix so the user sees it's right.

5. **Ingest** — progress, then land in the viewer.

Each format card links to a short **"Where do I get this file?"** explainer (CellRanger
`outs/` layout, how to download a GEO supplementary set, `SaveH5Seurat()` from R, etc.).

---

## 4. Messaging — examples (the heart of "forgiving & helpful")

These are the literal tone/spec for `IngestIssue.message` + `.suggestion`:

- **Lone matrix:** *"You've uploaded `matrix.mtx.gz` — the expression matrix from a 10x
  experiment. It can't be read on its own. Please also add the two companion files from
  the same folder: `barcodes.tsv.gz` (the cell IDs) and `features.tsv.gz` (the gene
  list). Older datasets call the gene file `genes.tsv` — that's fine too."*

- **Matrix as CSV:** *"This looks like a 10x expression matrix saved as a `.csv`. That
  works, but if you still have the original CellRanger files (`matrix.mtx`, `barcodes`,
  `features`) they load faster and more reliably. To continue with this CSV I just need to
  know: are the **genes** in the rows or the columns?"* (+ a row/col preview).

- **Dimension mismatch:** *"The gene file lists 36,601 genes but the matrix has 33,000
  rows. These need to match — they're probably from different samples. Re-upload the
  `features` file that came from the same folder as this matrix."*

- **Multi-sample GEO drop:** *"I found 3 complete 10x datasets (GSM4711, GSM4712,
  GSM4713) — looks like 3 samples. Merge them into one dataset with a `sample` label, or
  load them separately?"*

- **Wrong object in RDS:** *"This `.rds` contains a `data.frame`, not a Seurat object.
  If you have a Seurat object, save it with `saveRDS(obj, 'data.rds')` and upload that.
  If this is a plain expression table, re-save it as `.csv` and use the table importer."*

- **Truncated/corrupt:** *"This file looks cut off (the gzip ended early). The upload may
  have been interrupted — please try uploading it again."*

- **Excel rejected:** *"Excel files aren't a reliable format for single-cell data —
  spreadsheets silently reformat gene names (e.g. the gene `SEPT2` becomes a date) and
  can't hold a sparse matrix. Please export your data as `.csv` (or, better, use the
  original 10x / `.h5ad` files) and upload that."*

---

## 5. Endpoints (Phase A sketch)

```
POST /ingest/session                      → {session_id}      start a bundle
POST /ingest/session/{sid}/files          → upload file(s); returns detection + bundle state
GET  /ingest/session/{sid}                → current bundle, checklist, issues
POST /ingest/session/{sid}/options        → orientation / merge / gene-id choices
POST /ingest/session/{sid}/commit         → run loader → create dataset → {dataset_id}
DELETE /ingest/session/{sid}              → discard staged files
```

Staging dir: `data/ingest/{session_id}/` (cleaned on commit/discard/TTL). Reuses the
existing converter service for R formats.

---

## 6. Phasing

- **Phase A — backend foundation (all formats + merge):** detect + bundle + validate +
  loaders for 10x H5, 10x MEX (v2/v3), dense CSV/TSV, loom, h5ad/zarr; **multi-sample
  merge** (§3a) incl. GEO-prefix grouping + identifier reconciliation; ingest-session
  endpoints; immutable `ingested/` output. Unit tests with tiny fixtures of each format +
  a 2-sample merge (incl. an Ensembl-vs-symbol reconciliation case).
- **Phase B — the wizard tab:** format cards, multi-file/folder drop, live checklist,
  merge prompt, issue cards, preview, "where do I get this?" content.
- **Phase C — advanced:** gene-id↔symbol mapping, directory ingest from CellRanger
  `outs/`, format auto-suggest, doublet/empty-drop hints, >2-sample integration helpers.

## 7. Decisions & remaining questions

**Resolved 2026-05-29:**
- Formats: all supported ones in Phase A; **Excel rejected** with guidance.
- Multi-sample merge in Phase A; gene-axis **intersection by default**, union on request.
- **Identifier reconciliation** (Ensembl↔symbol) when intersection is suspiciously low,
  by inspecting other `.var` columns and proposing a `var_names` reset (§3a).
- Readers in Python backend; R converter = Seurat/SCE only.
- Detection: **best guess the user can correct**, except destructive-ambiguity cases
  (merge basis, raw-vs-normalized, index reset) which always ask.
- Output: new immutable **`ingested/`** layer; precedence derived > ingested > converted >
  uploads; processing still → `derived/`.
- **Staging:** abandoned ingest sessions live under `data/ingest/{session}/` with a
  **24 h TTL**, swept hourly (committed datasets are never auto-deleted).
- **Bundle model:** *unit* (one experiment; 10x MEX ≤3 files, others 1) vs *bundle* (one
  unit, or several merged). Merge units added **one at a time** (recommended) with a
  bulk-drop prefix-grouping fallback. **No hard cap** on units — **warn** when projected
  merged cells exceed the ~200k render target.
- **Low-intersection trigger:** **< 50% of the smaller** sample's var set → diagnose
  basis-mismatch (reconcile) vs genuinely-different-sets (inform only). Smaller-set
  denominator avoids false-flagging DEG-filtered / coding-only subsets.
- **CSV missing-axis labels:** auto-generate **cell** names; **never** auto-generate gene
  names (accept a `genes.txt` sidecar or reject with guidance).
- **Merged-size handling:** **no hard cap** — a **soft, honest warning** about interaction
  sluggishness on large merges. Bottleneck is interaction latency + browser/backend RAM,
  **not** rendering (deck.gl handles 1M+ points). A numeric threshold awaits §4 profiling.
- **Gene vocabulary:** pattern classification **+ a bundled human/mouse biomart
  Ensembl↔symbol map** (pinned release) for translation and orientation vocabulary.

**Still to settle (Phase-A build-time tuning, not blockers):**
1. Which **Ensembl release** to pin for the bundled biomart map, and whether to add
   species beyond human + mouse (rat, zebrafish?) up front.
