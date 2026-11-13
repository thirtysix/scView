# Changelog

All notable changes to scView are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Spatial statistics over several sections compare each section with itself; joint embeddings filter unfiltered
datasets and take up to 32; per-sample QC gets a genes floor; mouse ATAC on mm39 works; large datasets survive a
crash mid-edit. Found by re-analysing a published heart Visium study (Kuppe 2022) through scView's MCP.

### Added
- MCP: 11 spatial tools (Moran's I, domains, neighbourhood enrichment, co-occurrence, niches, spot composition) and
  the joint embedding (plan, run, cancel), 49 tools in all. Long runs return a job id.
- MCP `plot` draws a multi-section spatial dataset as sections side by side (12 at most, it says of how many), or
  one section alone (`section`); before, all sections were drawn on top of each other. `get_dataset` lists the
  spatial embedding, the sections and the off-tissue record.
- MCP `search_public_datasets` takes an `offset` and says exactly how many records it shows.
- A public spatial dataset whose file holds no spot positions says so at import and on its source card.
- ATAC on mouse mm39 (GRCm39, the build of 10x's current GRCm39-2024-A references): gene activity, TSS
  enrichment, motifs, chromVAR and peak-gene links run on it (GENCODE vM33 and UCSC mm39, fetched on first use).
  Datasets that 0.76.0 imported as mm39 work without importing them again.
- Composition niches on Visium spots, from Spot composition's cell-type shares (k-means on each spot's shares;
  the note names the reference). Without a spot composition, the Niches control says to run one first.
- Co-occurrence can show one section's own curves.
- Data Assessment flags curated cell types that look to have lost the authors' meaning (a CELLxGENE curation
  mapped Kuppe's "Myeloid" to "immature innate lymphoid cell"), with the cross-tab of curated vs author labels.
  Nothing is renamed.
- The joint embedding picker has a filter box, "Select shown", Clear and a selected count.

### Changed
- Moran's I over several sections centres each gene on its own section's mean and lists each section's own I.
  Before, differences between slides counted as spatial pattern: on 8 Kuppe slides the pooled I was higher than
  every single slide for 7 of 8 genes (MALAT1 0.61 vs at most 0.48). One-section results are unchanged.
- Co-occurrence over several sections scores each section against its own label shares and then combines them.
  Before, a pooled baseline turned slide composition into co-location (Fibroblast|Myeloid 1.6 at 4 mm on Kuppe,
  now 0.99, in line with neighbourhood enrichment). One-section results are unchanged.
- Moran's I runs on different sections are kept side by side in History; a run on one section no longer replaces
  another section's.
- Per-sample MAD QC never cuts below a genes floor: 200 genes for cells, 100 for spots and Visium HD cells, none for
  bins and targeted panels (settable). A sample where half the spots are near empty had a MAD cut of 4 genes and
  kept them all. The floor is skipped, and said so, where it would remove more than half of a sample. The preview
  says per sample which cut applied. Recipes from earlier versions replay without it. Kang and paul15: no change.
- The joint embedding filters the datasets scView never filtered (same QC as a single run), shows per dataset how
  many cells that removes, and can be switched off. Datasets filtered before are left as they are.
- The joint embedding takes up to 32 datasets (was 8); the memory estimate refuses a set that cannot fit.
- ATAC input on another species or build (T2T-CHM13, rat, zebrafish and the like) gets one note naming the builds
  scView supports; RNA work is unaffected. Before, rat or zebrafish fragments were read as GRCh38.
- Data Assessment: identical requests for one dataset share one computation.
- Marker database check: a match on one shared gene is "weak support (one shared gene)", whatever its p; it was
  "supported".
- The public dataset check reads a Visium file's header in 10 requests (0.2 to 4 s), not 68 (up to 70 s), and
  now reaches a verdict on CELLxGENE slides instead of "unknown".
- The tissue view's chosen section is kept in session files.

### Fixed
- Large datasets: an edit written into the file (labels, colours, a kept clustering) goes through a rollback
  journal, so a backend killed mid-write leaves the file as it was.
- The joint plan printed the one-condition note once per condition column; it is now one note naming the columns.

## [0.76.0] - 2026-11-13

Joint embeddings get a pipeline run's memory protection; spatial imports leave off-tissue spots out; marker
database suggestions keep to the tissue; NMF programs and chromVAR motifs are kept as scores; two libraries of one
sample that disagree are flagged. Found by re-analysing published studies (kidney, fat, heart Visium) through
scView's MCP.

### Added
- QC per sample flags two libraries of one sample that disagree in cell-type make-up (or depth) more than
  different samples usually do: a "Libraries differ" caution with the numbers. Also in the API
  (`GET /qc/libraries`) and the MCP (`library_concordance`).
- A drawn NMF program's usage and a motif's chromVAR z are saved as cell columns (`program_k<k>_<n>`,
  `chromvar_<motif>_<id>`), offered in Compare with the other scores, and say when they came from an earlier run.
- The joint embedding plan gives a memory and time estimate, and says up front when the run cannot fit even with
  every other dataset closed. A running joint embedding can be cancelled.
- The joint embedding plan lists the annotation columns it leaves out and why (only in some datasets, recomputed,
  an id per cell).

### Changed
- Marker database check: suggested cell types keep to the dataset's tissue (read from its title or set on the
  card; also `tissue` in the API and MCP). On a kidney dataset, 12 suggestions from other organs (colonocyte and
  the like) are gone, and no verdict changed.
- Joint datasets keep numeric annotations every source has (in_tissue, % mito, an author's deconvolution shares),
  not only categories.
- Spatial .h5ad imports, CELLxGENE Visium slides included, leave out spots marked off-tissue (`in_tissue` 0), as
  the Space Ranger reader already did: 713 of 4,992 spots on a Kuppe 2022 heart slide, all empty, which made
  housekeeping genes look spatially patterned. Datasets imported earlier keep their spots; Data Assessment, Moran's
  I, domains, co-occurrence and spot composition say how many off-tissue spots they still hold. The replay script leaves the same spots out of the original file.
- Public dataset search matches part of an assay, tissue or disease name ("Visium") when no name matches whole.
- Drawing the same score again saves nothing new; the same values from a later run are saved again so the record
  names that run.
- Captions name what a layer holds ("Log-normalized (raw)", "Counts (raw)"), not its storage key.
- ATAC input on mouse mm39 (GRCm39) is recognised from its names and chromosome ends and refused like mm9: the
  genome steps need mm10.

### Fixed
- A joint embedding of 8 Visium slides (39,936 spots) killed a 10 GB backend. The joint run now reserves its
  memory like a pipeline run, waits behind earlier runs, and stops a step that cannot fit before it starts, with
  a plain message (the half-made joint dataset is removed; the sources are never changed). It also holds far less:
  the source copies are let go after the merge and only the images scView draws are kept. The same 8 slides now
  finish in 6 GB.
- Saving a dataset's analysis record is now written to a copy and swapped in, so a backend killed mid-save no
  longer leaves an unreadable dataset.
- The dataset title of Kang's h5ad (stored as a list of one) is read, so its tissue is known.
- CELLxGENE spatial files no longer log a second, made-up library.
- Small API fixes: `within_value` takes one value or a list everywhere, `n_donors_column` is accepted, and
  `get_markers` says why there is no result id.

## [0.75.1] - 2026-11-13

### Fixed
- Split by a column, then resizing the window, could grey the grid out with "GPU context lost", and Retry did not
  bring it back. The GPU was fine: the grid's plot was rebuilt with drawing layers the old plot had already used,
  which deck.gl refuses. Each rebuilt plot now gets its own layers, in the split grid and the main plot alike.
- "GPU context lost" now appears only when the browser really drops the graphics context. When the browser
  refuses the page a new one (after a GPU reset it blames on the page), the overlay asks for a page reload instead
  of offering a Retry that cannot work.

## [0.75.0] - 2026-11-13

Scores are kept and offered in Compare; marker genes no longer rank raw counts on CELLxGENE files; cluster QC
sees nuclei contamination and ambient mixes; study-design warnings fit the design; Data Assessment fits in 6 GB
on a 39k-nuclei file. Found by re-analysing two published studies through scView's MCP.

### Added
- Scores are kept: a scored gene set (score_genes, AUCell, ssGSEA, GSVA) or a drawn TF/pathway activity is saved
  as a cell column (`score_<set>`, `activity_<source>`) with how it was made (method, genes used and missing,
  matrix, network). Scoring the same set again replaces it; a column the file brought is never overwritten.
- Compare (toolbar) offers saved scores under Scores, beside the groupings, each drawn with its own colour bar;
  cell-cycle and dissociation-stress scores are offered too. Remove a saved score from the same list. Saved
  scores also appear in Color by (with a violin by group), the histogram's Values, sessions and the h5ad export,
  and the replay script recomputes score_genes and AUCell scores exactly.
- MCP: `score_gene_set` saves the score when given a `name`; new `remove_saved_score`.
- ATAC input on mouse mm9 (NCBIM37) is recognised like hg19: from the reference the file names, the fragments
  header, or peaks past mm10's chromosome ends (mm9's chr7 runs 7 Mb longer). It is recorded as mm9, and gene
  activity, nearest genes, motifs, peak-gene links and TSS enrichment say they need mm10 instead of reading the
  wrong sequence.

- Cluster QC sees contamination in nuclei data: where the median cell has no mito reads, it shows the share of
  cells with any, and a "Mito in nuclei" caution for a cluster far above the rest. % mito is shown with enough
  digits to tell nuclei clusters apart.
- New cluster QC cautions: a cluster with another cluster's markers but far less RNA (an ambient-like mix), and
  marker lists led by non-coding RNAs, pseudogenes or sex genes. They are cautions in the evidence and the report;
  none changes the labels scView decides itself.
- The composition test and pseudobulk DE warn when a donor has under a third of the median donor's cells.
- Combine two to four columns into a new column in one call (API `POST /labels/combine`, MCP `combine_columns`),
  e.g. mouse x depot as a sample column; recorded and replayed.
- The composition test takes covariates (sex, batch, the animal) and pairs on the complete pairs only, saying how
  many pairs were used and left out. The unpaired test without covariates still matches R limma.
- Marker-score annotation keeps to the dataset's tissue (from a Tissue field, the file's tissue column or its
  title): on a kidney dataset, 8 of 21 clusters got types from other organs before, 0 now.
- MCP: `marker_db_check` runs the marker database check (verdicts and top suggestions per label), so
  `cluster_evidence` gets its marker lines without an annotation run first.

### Changed
- Clusters whose markers match no database type better than random now go to the AI with that note instead of
  being labelled "Unidentified" by scView (disease states are missing from the marker databases). Damaged clusters
  are still labelled "Low-quality cells". The random bar is said plainly: "beats 9 of 10 random lists".
- The repeated-donor warning fits the design: silent when the donor is a covariate, "block on it" when the
  compared factor varies within a donor.
- Cell-level DE and the composition test warn when a technology or batch value is on one side only; a covariate
  value held by one sample is named.
- Pseudobulk DE with no significant gene says the smallest adjusted p and how many genes reach nominal p < 0.05.
- Stored results go stale only when something they read changed (a neighbours rebuild no longer stales DE or
  composition; a new label column stales only results that used it), and say why.
- Milo results are labelled with k; age-like covariates can be numeric; per-donor tests name their sides; a note
  explains very large one-sided log2FC; `compare_de_results` speaks API terms.
- The assessment recognises CELLxGENE and Seurat columns: `mitochondrial_percent` and other % mito names,
  cell-cycle scores, `X_harmony` and scVI embeddings, `doublet_id` (also used as a doublet call in filtering).
- Error messages in the older panels (Gene Expression, Gene Sets, Trajectory, Observations, the co-pilot and
  others) show the server's reason instead of the raw `{"detail": ...}` text; validation errors give their
  messages. A test keeps new code from showing raw error bodies.
- The Ambient RNA card on a dataset without raw counts shows a plain note, not a red error.
- Keyboard focus moves into and stays inside the Share link dialog and the gate delete confirm, and the Plot
  reliability panel takes focus too. Escape closes each one and focus returns to its button. The delete confirm
  starts on Cancel.
- Evidence per cluster says what it is doing when the first call is slow (comparing markers with random genes,
  about 10 to 15 seconds the first time).

### Fixed
- Opening Data Assessment on a 39k-nuclei kidney file ran a 6 GB backend out of memory (also in 0.74.0): the
  ambient RNA check copied the counts, and four panels started it at once. It now reads the counts in place and
  runs once per dataset while the others wait: the same page peaks at 3.5 GB.
- Marker genes were ranked on raw counts when a file keeps them in `.raw` beside log-normalised X, as CELLxGENE
  files do: Wilcoxon on counts and fold changes of counts read as logs (a kidney podocyte gene at log2FC 437).
  Markers now use the same log-normalised matrix as gene-set scoring, the table records it, and stored tables
  ranked on counts are recomputed. This also corrects what the AI annotator, cluster evidence, co-pilot facts and
  enrichment (ORA version 7) were given on such files.
- Annotation no longer overwrites the file's own `cell_type` (or any column scView did not write): it writes
  `<grouping>_celltypeAnno` by default and refuses a curated column unless `annotation_overwrite` is set.
- On a file whose `.raw` holds counts, plots default to the log-normalised layer and the raw layer is labelled
  "Counts (raw)".
- The Compare grid header wraps instead of drawing "Cluster distances" over the pager; mouse TF candidates find
  mouse-cased TFs (Gata1); re-picking an already-scored gene set keeps the plot and the saved column in agreement.

## [0.74.0] - 2026-11-12

Each cluster's evidence in words; the co-pilot's claims check reaches Write methods and marks sentences in place;
hg19 ATAC is recognised; marker lists drop nameless genes.

### Added
- Evidence per cluster, in words (Data Assessment > Cluster QC > Evidence per cluster): why each cluster was
  labelled or flagged. Its markers against random genes (the best database p vs what 1 in 10 random lists reach),
  genes and % mito against the typical cluster with any flag that fired, a doublet-like cluster's two parents and
  its RNA, database types as shares of the evidence on named groupings, and the label scView gave it itself. Never
  a probability: a per-cluster probability model did worse than these rules on the annotation benchmark. Also
  under the annotation's "scView labelled..." note (Show the evidence), in the cluster report, for the co-pilot
  ("why is cluster 5 low quality?") and as the MCP tool `cluster_evidence`.
- Write methods checks its own paragraph: each sentence with a number, tool or version is compared with the
  recorded analysis recipe. A changed setting is marked refuted ("30 principal components" when 50 were used,
  Louvain when Leiden ran, a different tool version); a tool or version the recipe does not record is uncertain.
  The claims ledger appears under the methods text with refuted sentences always visible, and Check claims can
  judge the uncertain ones.
- Co-pilot answers mark each checked sentence in place: supported sentences are barely underlined, uncertain ones
  dotted, refuted ones wavy red. Hovering a sentence shows its verdict and the fact it was checked against.
- ATAC input on hg19 (GRCh37) is recognised from the reference the file names, the fragments file header, or peaks
  past GRCh38's chromosome ends (hg19's chr9, chr8, chr21 and chr19 run longer). It is recorded as hg19, and gene
  activity, nearest genes, motifs, peak-gene links and TSS enrichment say they need GRCh38 instead of silently using
  GRCh38 positions.

### Changed
- Marker lists for AI annotation, the co-pilot and enrichment skip genes that have an id but no name (Ensembl ids,
  AC012345.1-style clone ids) and fill up from the next ranks. On Tabula Sapiens liver these were 14% of the Leiden
  markers (68% in one cluster); on PBMC sets about 2%. A dataset named by Ensembl ids keeps them all. The Markers
  table is unchanged; enrichment results are recomputed once.

### Fixed
- The Markers tab no longer asks for markers of a column with one group (a one-library import's `library_id`, which
  answered 400). The default grouping skips such columns, and a dataset with nothing to group by yet says "Run
  clustering first, or add a column that splits the cells".
- Errors in the Markers, Enrichment, Activity and pathway diagram panels show the server's sentence, not its JSON.

## [0.73.0] - 2026-11-12

Bigger, paged small-multiple grids that export as figures; steadier lasso and camera.

### Added
- Split by now takes a column with up to 24 values (was 6), the several-genes grid up to 24 genes (was 9) and
  Compare colourings up to 12 columns (was 4). Choose 4, 6, 9, 12 or all panels per page and step with Prev / Next;
  every page shares one camera, one colour mapping and one colour scale (taken over all cells), so panels on
  different pages read the same. A lasso on any page selects that panel's cells.
- The grids' download menu offers the current page or all panels on one sheet, each as PNG, SVG or PDF, with panel
  titles ("label  N cells") and the legend.
- Sessions remember the grid's panels per page and page; older sessions open at 6 per page, page 1.
- Tissue plots export the cell and nucleus outlines shown on screen (Xenium, Visium HD segmented cells) in SVG and
  PDF, only those in view. Above 50,000 outlines or 600,000 vertices in view they are left out and the file and the
  export menu say so (zoom in to include them).

### Changed
- Opening a dataset from the library says so on its row ("Opening 73.3K cells…", with a seconds count after 2 s)
  while the server reads it (often 5-10 s); the other rows wait until it is open. Before, nothing showed it was busy.
- The lasso button has one tip, which gives the L shortcut (it showed a second, plain tooltip as well).
- A grid's header keeps what it shows ("Split by Sample · 23 panels"); the UMAP distance caveat beside it shortens
  first.
- Exported figures no longer carry the on-screen outline status ("Drawing the cell outlines in view (43).") as a
  caption; the note that outlines were left out stays.

### Fixed
- Resizing the plot (a window resize, full-page toggle, dragging the split pane, or the toolbar wrapping) no longer
  resets a plot you have zoomed or panned; it still fits the cells on first load, an embedding switch and Reset view.
- Drawing or clearing a selection no longer re-wraps the plot toolbar: the selection chip has its own slot.
- Violin plots on a log scale: the ticks near 0 no longer crowd (0 / 0.2 / 0.5); ticks keep a minimum gap.

## [0.72.0] - 2026-11-12

Unified View explains itself: hover a tab, label, heading or column to see what it does, when to use it and how to read it.

### Added
- Plain-language tips across the whole Unified View (184 catalog entries plus the existing explanations): every tab,
  toolbar control, card heading, non-obvious field, column header and plot says what it does, when it is useful and
  how to read it. Hover the thing itself for a moment (or focus its field from the keyboard). Written for biologists
  new to single-cell analysis; defaults and numbers in them are checked against the code.
- The tip texts live in one catalog (`frontend/src/lib/help/`), so the same wording appears wherever a thing is
  shown; a test checks every tip used exists, every entry is used, and every Unified View tab has one.

### Changed
- Tips are larger (12 px) and drawn above everything, so a scrolling panel or card edge no longer cuts them off;
  they open above or below, whichever has room, and stay inside the window.
- No more (i) info icons anywhere: every existing explanation (Data Assessment and QC cards, Gene Sets, import and
  export dialogs, Compare, annotation) now opens on hovering the heading, label or sentence it explains.
- Opening a dialog no longer puts keyboard focus on an info icon (which opened its tip over the dialog).

## [0.71.1] - 2026-11-12

Data Assessment no longer runs out of memory on a dense-ish 10k-cell dataset; split Seurat v5 objects convert.

### Fixed
- Opening Data Assessment on a 10k-cell dataset with dense-ish counts (65M values) could stop a 4 GB scView
  (seen on a hosted instance). The panels ask for the per-cell QC numbers at once, and each first request made
  its own pass, reading the counts as one block: three passes at ~1.3 GB each. Now one pass runs and the others
  wait for it, and the counts are read in blocks of at most 8M values. Same dataset, all panels at once: peak
  5.7 -> 2.5 GiB, per-cell QC 1.3 -> 0.3 GiB.
- Seurat v5 objects whose layers are split by sample (`counts.1`, `counts.2`, ... after `split()` or `merge()`)
  now convert: the converter joins the layers first and keeps the object's cell order. Before, the conversion
  stopped with "data, indices, and indptr should be 1-D".
- Data Assessment names Seurat's own QC columns (`nCount_RNA`, `nFeature_RNA`, `percent.mt`, ...) instead of
  saying none were found. The QC metrics step still runs, since other steps read scanpy's columns.

### Known limits
- AI annotation with DeepSeek-V4-Pro as the annotator model (not the default): since 0.71.0 some doublet clusters
  come back as "Doublet" or "Low-quality cells" instead of naming both cell types. In the annotation benchmark
  V4-Pro named both types for 37 of 96 visible doublet clusters, against 47 with 0.70.0; the default model
  (DeepSeek-V4.1-Flash) and the other models checked held. The replies are the model's own answers, not a reading
  error.

## [0.71.0] - 2026-11-11

AI annotation: scView decides which clusters are damaged cells or name no cell type; the AI names the cell types.

### Changed
- AI annotation (Data Assessment > Cell-type annotation): a cluster that cluster QC flags as damaged (high
  mitochondrial RNA, or few genes with it) is labelled "Low-quality cells" by scView itself, and a cluster whose
  markers match no cell type in the marker databases better than random genes do is labelled "Unidentified". Neither
  is sent to the AI; the panel says which clusters scView labelled and why. In the annotation benchmark the AI named
  a cell type for every random marker list (53 of 162 handled correctly before, 162 of 162 now), and its
  low-quality call moved with the prompt's wording and with the other clusters listed. With these decisions made
  outside the AI, background labels, doublet naming and the low-quality control held or improved for the default
  model (pbmc10k background hF1 0.969 -> 0.951, the largest change, inside its 0.02 gate).
- A flagged cluster whose markers best match a cell type with much mitochondrial RNA by nature (cardiomyocytes,
  hepatocytes, kidney tubule and collecting-duct cells, skeletal muscle, brown adipocytes, gastric parietal cells)
  still goes to the AI with its QC note, which may name the type.

## [0.70.0] - 2026-11-11

The sidebar shows which scView you run and says when a newer release is out.

### Added
- The sidebar shows scView's version. When a newer release is out, a button there opens the update steps with a
  direct link to that release's ZIP (and the release notes). The backend asks GitHub's release list at most once a
  day (no user, dataset or machine details); `UPDATE_CHECK=false` turns it off.
  `GET /api/v1/system/version`.

### Fixed
- A run that failed held its dataset (memory reservation and edit lock) for a moment after the error reached the
  browser, so an action started right away could wait or be refused. The dataset is released first now.
- A backend test compared two stored results including their creation second, and failed when the two runs
  straddled a second (`test_mixscape_stream`).

## [0.69.1] - 2026-11-11

Two fixes for the violin under the Unified View plot.

### Fixed
- Unified View: changing "Group by (violin)" kept the old groups' violins and only recoloured them. The violin on
  screen (a gene, protein, gene-set score, program, gene activity or chromVAR score) is now split by the new column.
- Unified View: after colouring the plot by a column, "Per donor" asked for a gene although the violin and the
  search box still showed one. Per donor now uses the gene the violin shows.

## [0.69.0] - 2026-11-11

The AI features can use another language model provider than DeepInfra, such as Privatemode for unpublished data.

### Added
- The language model provider is a setting: `LLM_BASE_URL` (any OpenAI-compatible endpoint), `LLM_API_KEY` and
  `LLM_CHAT_MODEL` (plus `LLM_PROVIDER_NAME` for display). DeepInfra stays the default and existing `.env` files
  work unchanged. For unpublished data this allows [Privatemode](https://www.privatemode.ai), whose models run in
  confidential-computing enclaves, through Edgeless's `privatemode-proxy` beside scView (compose example in
  docs/AI_ASSISTANT.md). Checked against a live proxy with `glm-5.3-flash`: co-pilot chat, advisor, LLM annotation
  and the claims check all work.
- The co-pilot's Sources tooltip and the AI annotation tab name the provider the questions and marker lists go to;
  the startup log names the provider and model, and `/assistant/llm-status` returns them.

### Changed
- `LLM_API_KEY` falls back to `DEEPINFRA_API_KEY` only while the URL is DeepInfra's, so a DeepInfra key is never
  sent to another provider. Literature search (RAG) embeds and reranks on DeepInfra, so it is off when another
  provider is configured (`/assistant/rag-status` reason `provider`).

## [0.68.1] - 2026-11-09

A bug fix for mixture labels from the AI annotator.

### Fixed
- AI annotator: a mixture answered with one Cell Ontology id per part ("NK cells / monocytes | CL:0000623 /
  CL:0000576") kept "| CL:0000623 /" in the label and took the second part's id as the label's. The label is now
  clean and a mixture gets no single id. Present since 0.64.0.

## [0.68.0] - 2026-11-09

Cluster QC flags more clusters that look like doublets of two others, and the AI annotator names more of them as a mixture.

### Changed
- Cluster QC's "looks like doublets of two clusters" check now needs a fifth of the cluster's top markers from each
  parent (was 30%) and the cluster to be smaller than the larger parent (was both). The old rules missed 7 of 12
  visible doublet clusters in the annotation benchmark, where one parent's genes lead the marker ranking; the new
  ones flagged none of about 210 clean clusters (author labels and Leiden clusters on 8 datasets, paul15 and Kang
  included). The AI annotator sees these flags as QC notes: on the benchmark every model named more visible
  doublets as a mixture (default model 99 -> 101 of 108, Seed-2.0-pro 84 -> 98).

## [0.67.0] - 2026-11-09

The AI annotator waits out a busy model instead of failing, and says on the card when its reply was cut short.

### Added
- **A cut annotator reply now shows a warning under the Annotate button.** When the AI annotator's reply runs out
  of tokens even after its retry, the card says how many clusters were left Unknown and what to do (run again,
  fewer clusters, or a model that reasons less). Before, this showed only as a chip in History.
- **The co-pilot's feature guide has the Cell-type annotation card**, including the `<column>_qualified` column,
  so app questions about qualified labels get a real UI path.

### Changed
- **The AI annotator waits and asks again when the model is busy.** DeepInfra answers 429 "Model busy" for a model
  under load; the annotation step failed on it. It now asks again after 5, 15 and 30 s, and History records
  `llm_busy_retries` when it had to.

## [0.66.1] - 2026-11-08

The AI annotator names both cell types of a two-lineage cluster again, as it did before 0.66.0.

### Fixed
- **0.66.0's low-quality text made the default annotator name one parent of a doublet cluster.** The mixture rule
  sat before the new low-quality text, which ended with "name the cell type it is in". The mixture rule is now the
  last naming rule ("even when one lineage is stronger or the tissue suggests one of them"), and the stress
  sentence says "name the cell type under the stress genes". On synthetic doublets of pairs and datasets the
  problem was not found on (Zheng sorted PBMC and Tabula Sapiens liver; 6 visible doublet groups, 2 tissue-hint
  settings, 3 repeats), DeepSeek-V4.1-Flash named both parents in 14 of 36 runs with 0.66.0, 25 with 0.65.0 and 29
  with this fix. The low-quality answer is unchanged: the mito-high control cluster is still "Low-quality cells"
  in 6 of 6 runs for each DeepSeek model, and clean clusters score as before.

## [0.66.0] - 2026-11-08

Clusters that share a cell-type label, or carry a mixture or no-type label, get the genes that set them apart, the AI annotator can call a cluster "Low-quality cells" with per-cluster QC numbers to go on, and paper cluster ids from CellMarker are no longer offered as cell types.

### Added
- **Marker qualifiers for labels that do not tell clusters apart.** When several clusters get the same cell-type
  label, or a cluster's label is a mixture, doublets, "Unknown" or "Low-quality cells", the annotation step also
  writes `<column>_qualified` with the genes that set each such cluster apart: "Monocytes (CCL8+ RSAD2+)" for
  Kang's stimulated monocytes beside "Monocytes (IL8+ CLEC5A+)" for the controls, and "T cells (DNAJB6+ HSPH1+)"
  for the heat-shock T cells beside "T cells (SH3BGRL3+ CD3D+)". A cluster sharing its label is compared with
  the others of that label, pooled; a lone mixture or no-type label with all other clusters. A gene stands out
  only when it is detected in at least half the cluster, in 30 points more of it than of the clusters compared
  with, and its mean log expression is at least 1 higher; at most two genes, and a gene the compared clusters
  express and this one does not (`CD4-`) only in a slot left free. Ribosomal, mitochondrial and unnamed genes
  (Ensembl ids, LINC, antisense, Gm/Rik) are never used. Nothing standing out leaves the plain label. The labels
  themselves, and their Cell Ontology ids, are unchanged. The record (`uns["<column>_qualifiers"]`) keeps each
  cluster's genes and what it was compared with; renaming a label works them out again, and replay scripts
  write the column from the record.
- **The AI annotator can answer "Low-quality cells".** Clusters that cluster QC flags as damaged cells or
  doublets (low quality, high % mito, few genes, low MALAT1, doublet-rich, doublet-like) are sent with a short QC
  note beside their markers, e.g. "median 31 genes/cell vs 2,046 typical; 97.1% mitochondrial vs 10.0%": the
  marker list leaves mitochondrial genes out, so the model could not see high mito before. The prompt now allows
  "Low-quality cells" for a cluster with no lineage genes whose markers are mostly mitochondrial-encoded or
  nuclear-retained transcripts, or whose QC note says few genes and high mito, and says that a QC note alone is
  not enough (cardiomyocytes, hepatocytes and kidney tubule cells have high mito by nature) and that a
  heat-shock or stress program is a state of living cells, not low quality. The label gets no Cell Ontology id.
  In the annotation benchmark's negative controls (T4) every model had called a mito-high cluster "Activated
  CD14+ monocytes". Live: pbmc 1k without filtering names its 97 %-mito cluster "Low-quality cells"; Kang's
  heat-shock T cells, which an earlier draft of this prompt called low quality, stay T cells.

### Fixed
- **A paper's cluster id is no longer offered as a cell type.** CellMarker 2.0 and 3.0 list 68 "cell names" that
  are one paper's cluster ids, such as "Rheaume et al.Nat Commun.18" and "Lake et al.Science.Ex1", each with a CL
  id. They reached the marker-database annotator and its suggestions (found by the annotation benchmark's shuffled
  marker control). Names containing "et al" are now left out when the database loads; no re-download is needed.

## [0.65.0] - 2026-11-07

Every supported chat model answers without spending its reply budget thinking, a cut annotator reply is asked again, DC2 labels get the cDC2 id, and the variable-gene, merge and compare-methods time estimates are refit.

### Fixed
- **Thinking is turned off the way each chat model accepts it.** The AI annotator, co-pilot, claims judge and QC
  report used to switch thinking off for DeepSeek only, so a `RAG_CHAT_MODEL` or annotator model from another
  family could spend its whole reply budget thinking and answer with nothing. Probed on DeepInfra 2026-11-07:
  GLM-5.2, Kimi, Qwen3.5/3.6/3.8, MiniMax, MiMo, Nemotron 3, Seed 2.0 and Hy4 now get `enable_thinking: false`
  like DeepSeek. GLM-5.3 and GLM-5.3-Flash use low reasoning effort (switching thinking off made them reason
  until the budget ran out), and gpt-oss uses low effort and gets 2,000 more reply tokens. Models that do not think
  (DeepSeek-V3.2, Qwen3-235B Instruct, gemma, unknown models) are sent nothing extra.
- **AI annotator: a reply cut at the token limit is asked again once** with more room, and the reply that names
  more clusters is kept. An unfinished last line no longer becomes a label. If the reply is still cut, History
  names the clusters left Unknown.
- **"DC1" and "DC2" labels get the conventional DC ids.** CellTypist's DC2 is a cDC2, but the Cell Ontology lists
  "DC2" as an exact synonym of plasmacytoid DC, so a DC2 cluster got the pDC id and the marker database check
  compared it with pDC markers. DC1/cDC1 now map to CL:0002394 and DC2/cDC2 to CL:0002399, shown as
  "single-cell name".
- **Windows start script:** the API key and memory prompts keep their indent.

### Changed
- **Time estimates refit** (warm, 2 CPUs, Kang): the variable-gene check 63 s estimated against 67 s measured (was
  122 s), the merge check and count-split markers 31 s against 34 s (was 84 s); each setting is now timed as its
  PCs, graph and clustering. Compare batch-correction methods counts the batches: Kang by donor (8 batches) 16 s
  against 19 s measured (was 6.4 s). `GET /batch/compare` returns `n_batches`. A merge check that turns out not to
  need cluster-count matching still finishes sooner than estimated.

## [0.64.0] - 2026-11-07

Cell-type labels carry Cell Ontology ids, the AI annotator reads numbered clusters by id, and the start scripts never offer more memory than Docker or Podman has.

### Added
- **Cell Ontology ids for cell-type labels.** The AI annotator asks the model for each label's Cell Ontology (CL)
  id and keeps it only when the term exists and the label agrees with its name or an exact synonym; otherwise the
  label is matched by name, or gets no id. Matching is strict: the same name, the same exact synonym, or the same
  once case, plurals, "+" (positive), hyphens and a trailing "cells" are set aside. Abbreviations such as "CD14 Mono"
  get no id rather than a guess. The Cell-type annotation card lists each label of the column it wrote with its CL
  id, linking to the term's page. Any label column can be mapped through
  `GET /datasets/{id}/annotation/ontology?column=` or the MCP tool `cell_ontology_ids`. The ids are stored with the
  dataset, written in by the replay script and named in the co-pilot's cell-type fact. The Cell Ontology download is
  pinned to release 2026-06-08 (sha256-checked). A file put in place by hand is kept and the status says whether it
  is the pinned one; a file fetched unpinned by an earlier version is replaced by the pinned release once, on first
  use (the old file stays in use if that download fails).

### Changed
- **TF and pathway activity on a dataset opened from its file reads the matrix once.** The first request keeps every
  cell's scores (Kang x CollecTRI: about 34 MB), so colouring by one TF after another and the group heatmap no longer
  re-read the expression matrix: 0.17 s to 0.03 s per TF on Kang. The room for the scores is held with that first
  read; they are kept only below 512 MB and 1/16 of the memory limit, and only when they fit without closing another
  dataset. Otherwise requests behave as before.
- **Time estimates.** The "first run" notes on Compare methods, the variable-gene check, the merge check, the
  count-split check and layout stability count numba's compile only while it is still to come: any neighbour graph
  or UMAP built in the server compiles it, and a restart keeps only a shorter import. Kang's Compare methods now says
  14 s after a restart (took 15) and 6 s once warm (took 7); both said 26 s. The merge check's "+70 s" is about 28 s,
  then none. WNN, velocity and the pipeline's neighbour step leave out a compile already done (neighbours once warm:
  0.6-2.5 s, was 16-64 s). Spot composition's time model is now a per-spot part plus fit genes x types squared:
  estimates had been 1.8-3.3x high and are within 20% on Visium HD 16 um bins (7-37 types), CytAssist spots and the
  Lymph Node. The ATAC likelihood-ratio estimate scales a fit's cost per cell with the cells (E18 mouse brain: says
  74 s, took 63; said 112).

### Fixed
- **The AI annotator matched numeric clusters by line order.** The cluster id was stripped from each line of the
  model's reply, so every numbered clustering (all Leiden runs) fell back to reading the labels in order, and one
  extra line in the reply shifted every label by one. Each line's id is now read, "Cluster 13" no longer matches
  cluster 3, and a cluster the reply leaves out stays Unknown. The reply's length limit grows with the number of
  clusters (it was 800 tokens, about 30 clusters).
- **A malformed Cell Ontology id in the AI annotator's reply stayed in the label** ("Microglia | CL:00002077",
  "Endothelial cells | CL", or "CL:none" as the whole label). An id of the wrong length is dropped and nothing of it
  is kept in the label text; a reply line with no name leaves the cluster Unknown.
- **Start scripts no longer suggest more memory than Docker or Podman has.** The 4 GB minimum was applied after the
  cap, so a fresh Windows install with a 3.8 GB Podman machine was offered 4 GB, above what the machine has. The
  suggestion is now at most the engine's memory minus 0.5 GB; when Docker or Podman is the limit, the script says
  so and how to give it more (macOS Podman: `podman machine set --memory`; Docker Desktop: Settings > Resources;
  Windows: the README's `.wslconfig` section). A saved limit above the engine's memory is warned about on every
  start. start.sh no longer stops silently on an older `.env` without the `SCVIEW_MEMORY_GB` line.
- **Spot composition's estimate when a run starts counted 20 types** whenever no label was left out or merged; it
  now counts the column's own, so the two agree (the Lymph Node's 33-type run said 46 s at the start, 79 s in the
  reference check).

## [0.63.0] - 2026-11-06

Runs queue for memory instead of stopping each other, backed datasets work across the analyses built on log-normalised expression, and History reads in words.

### Changed
- **A run whose next step fits only once another run or read has finished now waits for it instead of stopping.**
  Its progress shows what it is waiting for and for how long; Cancel still works, and it stops as before after 30
  minutes of waiting. A run that started later waits for an earlier one, never the reverse, so two runs cannot wait
  on each other. A step too big even with nothing else running still stops at once, nothing saved.
- **The ligand-receptor comparison and cytokine response name their two conditions as baseline and test**, on
  screen ("Baseline (A) / Test (B)" and "Test condition (A) / vs baseline (B)") and in the API (field descriptions
  and explicit `baseline` and `test` values in the responses). The meaning of `a` and `b` is unchanged: `a` is the
  baseline in the ligand-receptor comparison and the test in cytokine response.

### Fixed
- **TF and pathway activity, cytokine response, upstream TF candidates, ligand-receptor communication, genes
  along pseudotime, spatially variable genes, Mixscape, the marker panel and the shared marker choice failed on a
  dataset opened from its file** (a processed dataset over the backed-mode size, 2 GB by default, reopened after a
  restart). They now give the same results as in memory, and those that read the whole matrix first hold room for
  it or refuse with a 507 naming the dataset.
- **A DE test or another read of a backed dataset no longer fails when a run or an edit reads that dataset into
  memory at the same time**: the file stays open until the read has finished. Spot composition no longer closes
  the shared file when it reads its own copy of the spots.
- **Re-filtering by per-type % mito reserves the memory its re-processing needs**, like any pipeline run, so
  opening another dataset meanwhile is counted against it instead of risking the backend being stopped.
- History and the recipe give every recorded step a readable name, and the import step's details read as words
  instead of raw JSON.
- The Protein tab no longer asks for RNA expression of an antibody whose target gene is not in the data, so the
  browser no longer logs a 404.
- The QC filtering record holds up to 16 filtering criteria; a ninth would have been lost. Records written before
  still read.

### Added
- `GET /datasets/{id}/genes` takes an optional `layer`.

## [0.62.0] - 2026-11-05

edgeR QL joins the pseudobulk models, DESeq2 runs about 20x faster on scView's own engine, and CELLxGENE uploads get gene symbols.

### Added
- **edgeR quasi-likelihood is offered for pseudobulk DE** ("edgeR QL" in the model list, `pb_method: "edger_ql"`
  in the API and MCP tool): edgeR 4's QL F-test (glmQLFit with legacy=FALSE, then glmQLFTest), written from the
  papers and checked against R edgeR 4.6.3, whose p values it matches to a median of 3 in 100,000 on the log10
  scale. On the 18 datasets behind the voom default it held its false discovery rate as limma-voom did (1 in 7
  false at 2 donors per side, about 1 in 15 at 3 to 4) and ranked genes slightly better (AUC 0.90 vs 0.89).
  limma-voom stays the default. The first edgeR QL run after a restart compiles its kernels (a few seconds).

### Changed
- **Pseudobulk DESeq2 runs on scView's own vectorised DESeq2 instead of pydeseq2**, the same steps fitted for all
  genes at once: a cell type now takes 0.1-0.3 s instead of 2-6 s (Kang). Its dispersion trend is fitted on the
  genes above the dispersion floor, which is where R's DESeq2 lands, so p values sit 10x closer to R's (median
  0.0002 on the log10 scale) and a result can differ slightly from 0.61.0's (on Kang CD14 monocytes 2,199 genes
  at adjusted p < 0.05 vs 2,265 under pydeseq2). The FDR caution below 6 donors per side still applies. Milo
  neighbourhood DA keeps pydeseq2, matching pertpy.
- **A CELLxGENE .h5ad uploaded directly now gets gene symbols as its gene names**, as the Discover import always
  did: the symbols in var['feature_name'] become the gene names in X and .raw, the Ensembl ids are kept in
  var['gene_ids'], a gene with no symbol keeps its id, and a repeated symbol gets "-1". Before, these files kept
  Ensembl ids, so gene search ("XIST"), marker databases, gene sets and the sex and % mito checks found nothing.
  Applies to the quick upload and the import wizard; the uploaded file stays as sent, and the replay script
  repeats the rename. Datasets uploaded before keep their names.

## [0.61.0] - 2026-11-05

Pseudobulk DE defaults to limma-voom, chosen on 18 public datasets; sex-linked genes are flagged in every DE result.

### Changed
- **Pseudobulk DE now uses limma-voom by default instead of DESeq2.** On real donor splits of 18 public datasets
  (82 cell types, 166,000 fits) with known effects added, DESeq2 called about 1 false gene in 3 at a nominal FDR
  of 5% with 2 donors per side, 1 in 5 at 3 and 1 in 6 at 4; limma-voom called 1 in 7 at 2 donors and stayed near
  1 in 20 from 3 donors on, while ranking genes as well (AUC 0.89 for both). DESeq2 is still offered, and a DESeq2
  run with fewer than 6 donors on its smaller side now carries a caution that its adjusted p-values are optimistic.
  Results stored before this change keep their DESeq2 label. Method and numbers: `docs/validation/pseudobulk-de.md`.
- **The study planner's DE power now runs limma-voom on each simulated study**, the same code as pseudobulk DE,
  so a plan matches the test you will actually run. DESeq2 can still be chosen ("Test" in Compare > Plan a
  study > Pseudobulk DE). voom plans also run about 3x faster.

### Added
- **Sex-linked genes are marked in every DE result.** XIST, TSIX and Y-chromosome genes get a "sex" badge in
  the DE table (red when the two sides' samples differ in sex, grey otherwise), and the co-pilot is told which
  significant genes are sex-linked. Genes named by Ensembl id (a CELLxGENE file uploaded as it is) are matched
  through the symbols in var['feature_name']. In random splits of one condition's donors across 18 public datasets, these
  genes were among the most frequent false hits whenever the split happened to line up with donor sex.
- **DE between groups warns when the sides' sex mix differs**, not only when a side is all one sex: a gap of
  25 percentage points or more in the share of female samples (for example 3 female + 1 male against
  1 female + 3 male) now gives a caution.

## [0.60.1] - 2026-11-04

Memory estimates for large-dataset mode, measured on CSC Roihu (the 1,000-dataset run's batch nodes).

### Fixed
- **The memory estimate for doublet detection in large-dataset mode counts Scrublet's own variable genes**, as the
  in-memory estimate does, from sample rows of the counts on disk (about 14 s on 159k cells), and adds the channel's
  counts read in beside its arrays. It assumed 10% of all genes before: on a 159k-cell CELLxGENE dataset with
  35,469 genes it said 4.6 GiB for a step that rose 6.0 GiB; now 5.9. The check before each step of a large run uses
  the same estimate.
- **Large-dataset mode's per-step memory estimates are refitted on CSC Roihu** (8 datasets of 18k-185k cells, 4 and
  2 CPUs, measured every second). After doublet detection a process keeps about a tenth of that step's memory, and
  writing the matrices takes more with more CPUs; neither showed in the laptop run the model came from. Steps after
  doublet detection were up to 2.2x their estimate; now none is above 1.1x. The run's peak estimate is unchanged
  where doublet detection sets it.

## [0.60.0] - 2026-11-04

Pseudobulk DE on large datasets (part A of the pseudobulk study; checked on CSC Roihu on a 159k-cell placenta atlas:
all 7 per-cell-type comparisons that 0.59.0 refused now run; the whole analysis took 34 min and peaked at 7.3 GB).

### Added
- **Pseudobulk DE between groups runs on large datasets.** The counts are summed per donor and group straight from
  the counts file, a span of 5,000 cells at a time, so the matrix is never loaded; DESeq2 and voom give the same
  answer as in memory. Cell-level DE (Wilcoxon, t-test, logreg) is still refused in large-dataset mode.

## [0.59.0] - 2026-11-03

Batch 40: fixes from the 20-dataset pilot of the 1,000-dataset run (CSC Roihu, 1.1M cells).

### Added
- **Your file's own UMAP is kept** as `X_umap_original` ("UMAP (original)") before scView's replaces it, at reset to
  counts and before the UMAP step, whatever the case of its key (`X_UMAP` on Jäkel 2019).
- `PUBLIC_IMPORT_LOCAL_DIR`: a public import copies `<version>.h5ad` (and its listing record) from a folder instead of
  downloading, for batch nodes without internet.
- `SCVIEW_MEMORY_RESERVE` (0.05-0.5, default 0.25): the share of the memory limit kept free beside a load or run. A
  batch job with one client can use about 0.10; at 0.25 the pilot's two largest datasets (159k and 185k cells on 66
  and 69 GB jobs) were sent to large-dataset mode.

### Fixed
- **Marker genes no longer fail a whole column when a group has one cell** (author columns on two pilot datasets).
  Those groups are left out, named in the answer (`skipped_groups`), and their cells stay in every other group's rest.
- The sex check works on datasets opened from disk (it failed with "'_CSRDataset' object has no attribute 'sum'"),
  and takes sample columns of up to 1,000 groups (it refused 51 batches and 66 libraries at the violin cap of 50).
- A stored result asked for right after it was computed is found: enrichment of a DE result in the next MCP call
  answered "No such stored result" on 3 of 20 pilot datasets, because the result was still being written.
- The memory estimate for doublet detection costs the three largest capture channels, not only the one with the most
  cells: a smaller channel with more variable genes peaks higher (pilot: 9,400 cells with 4,219 variable genes against
  10,858 with 3,418; Scrublet measured 5.44 vs 5.03 GB, estimated 5.71 vs 5.34).

## [0.58.0] - 2026-11-03

Batch 39: fixes found by re-analysing five published datasets through the MCP server (Wilk 2020, Jäkel 2019,
Yang 2021, Otero-Garcia 2022, Liu 2021).

### Fixed
- **DE no longer copies `.raw`** for its transcriptome-size check. On Liu 2021 (125k PBMCs) that copy added 4.5 GB at
  the peak and the backend was killed; the same classical-monocyte Wilcoxon (38,334 cells) now takes +0.8 GB. In-memory
  DE (cell-level, pseudobulk, method comparison, audit) reserves its estimate and answers with a one-sentence 507 when
  it cannot fit, before allocating.
- **Neighbours, clustering and UMAP on a stored PCA read no expression matrix** and never convert a dataset to the
  large-dataset layout. On Otero-Garcia's 96k neurons the run took 48 s instead of 196 s plus a conversion, and DE
  stayed available.
- **Gene-set scores use log-normalised values**, never raw counts held in `.raw` (every CELLxGENE import and Kang).
  Scores on those datasets change: the old ones were wrong (Kang maximum 58.6, now 3.3).
- Embedding plots draw when some cells have no coordinates (Jäkel: 4,903 of 17,799), and say how many were left out.
- The first hand label on a dataset is saved in a fraction of a second (it rewrote the whole file: 11 s on 23k cells).
- Adding a label column no longer marks DE, Milo or composition results stale unless they used that column.
- The sex check no longer calls pooled-lane females "mixed" from ambient Y reads, and says when XIST is not in the
  matrix.

### Added
- **Pseudobulk covariates** (sex, batch, age, chemistry): up to 4 per-sample columns in the DESeq2 and voom designs,
  in the UI as "Adjust" chips. Voom matches limma 3.68.5 within 3e-11. A covariate that varies within a sample, or is
  confounded or collinear with the condition, is refused in one sentence.
- **Per-donor expression compares two chosen conditions** of a column with three or more values (a "Compare ... vs"
  choice in the UI), names the difference's direction and lists the donor-groups left out.
- **Warnings about replicates**: pseudobulk and composition warn when samples repeat a person (several samples per
  donor), when donors sit on both sides of an unpaired design, and when a side has fewer than 3 donors; pseudobulk then
  counts people, not samples.
- Cell-level DE warns when the two sides' cells differ in sex share by 15 points or more, even when the donors balance
  (Yang 2021: 7% vs 31%); with sex as a pseudobulk covariate the sex warning says it is adjusted for.
- **Composition**: shares within one compartment (e.g. T cells among immune cells), samples left out, and one stored
  result per comparison.
- Preflight says what converting to the large-dataset layout turns off; MCP `run_pipeline` needs
  `allow_large_mode=true` to convert.
- CELLxGENE imports check memory before downloading, from the file header read by HTTP ranges, and refuse a file that
  cannot open (MCP `check_public_dataset`).
- Milo reports k, the representation, and which samples fill each neighbourhood; the neighbours step takes a
  representation (`neighbors_use_rep`).
- GSEA runs are stored as results. Discover search matches words across collection descriptions.
- MCP: `describe_column` summarises a numeric column per group, `get_dataset` lists every column, `expression_by_group`
  pages, `score_gene_set` gives per-sample scores within groups with the donor-level test, cell filters refuse unknown
  keys, a user column takes up to 2,000 labels, and a missing column's error names the closest ones.

## [0.57.0] - 2026-11-03

### Added
- **MCP server for AI agents** at `/mcp` on the backend: `claude mcp add --transport http scview
  http://127.0.0.1:8080/mcp` (also `http://127.0.0.1:3000/mcp` through the frontend). 31 tools call the same routes
  as the UI, so validation, edit locks, memory checks and provenance are the UI's: find and import a CELLxGENE
  dataset, assess, preflight and run the pipeline, markers, CellTypist, labels from a cell filter (an agent cannot
  lasso), DE cell-level or pseudobulk, compare two DE results, per-donor expression, composition, Milo, enrichment,
  gene-set scores, a sex check, stored results, and PNG plots (embedding, violin, dot plot, optionally within one
  cell type). Long runs are jobs (`job_status`, `cancel_job`). Results are compact (top rows plus totals; each was
  under 8,000 characters on Kang), unknown arguments and pipeline parameters are refused with a hint, and nothing
  deletes. On by default, off in public mode (`MCP_ENABLED`); `ACCESS_TOKEN` applies; requests from other web
  origins or host names are refused (`MCP_ALLOWED_ORIGINS` to allow one). Tested end to end with the MCP SDK client
  and with a headless Claude Code run answering a question about the Wilk et al. 2020 COVID-19 PBMC data in 9 tool
  turns. Claude Desktop reaches it through `mcp-remote` (in the README, not tested by us).
- **Patients against controls, per donor:** per-donor expression now tests two unpaired conditions on the donor
  values (exact rank-sum and Welch t, at least 3 donors a side), shown in a table under the boxes with the
  difference's direction. With 7 against 6 donors the smallest possible rank-sum p is 0.0012 (2 / C(13, 7)); the
  result says so when a p sits at that floor.
- **Sex warning in DE between groups** (cell-level and pseudobulk): when the two sides' samples differ in sex make-up
  (from the metadata sex column, or a sex check already run), a red note says XIST, TSIX and Y genes will show up for
  that reason. On Wilk 2020 (every patient male, 2 of 6 controls female) cell-level DE put XIST at padj 4e-20 in
  CD14 monocytes with no warning before.

### Changed
- The composition warning for groups found almost only in one condition is worded for cell-type labels when the
  column is not a clustering: a real expansion or loss is possible, or a label called differently per condition.

## [0.56.0] - 2026-11-02

### Added
- **Peak-gene links** (Unified View > Accessibility): Signac's LinkPeaks defaults, in about a minute for the variable
  genes and about 3 minutes for all genes on PBMC 10k at 2 CPUs, under 1 GiB of memory (R took 573 s for 617 genes at
  1 CPU). Against Signac 1.17.1 on the same matrices, the tested pairs are the same and scores agree within 9e-15 on
  PBMC 10k and 2e-15 on mouse E18; z agrees at r 0.991 and 0.984 to 0.986, as close as two Signac seeds are to each
  other (0.992, 0.987), as do the kept links. BH is added and raw p kept. Clicking a row draws arcs under the coverage
  tracks, or on a peak track without a fragments file, with sign and strength visible, and in the SVG/PDF export. A
  link is labelled as a correlation and a candidate, never proof. Results go stale when the cells or RNA change. The
  time estimate is rough when you choose groups of cells (19 s against 33, 190 s against 140); whole-dataset estimates
  were within 11%.
- **Motifs in peaks** (Accessibility): the 879 JASPAR 2024 motifs in every peak, about 2 minutes and kept per peak set.
  Every hit is the same as motifmatchr 1.34.0 on PBMC 10k (143,836 peaks x 879 motifs, 0 differ).
- **Motif enrichment in a group's differential peaks** (Signac's FindMotifs: GC-matched background, FDR, fold), with
  the TF's own RNA in the group, sequence logos and a zip export. Ours against Signac: r 0.9986 on -log10 p and 19 of
  the top 20; two Signac seeds: 0.9981 and 19. The query defaults to the vignette's raw p < 0.005, open in over 20% of
  the group; FDR is one select away. A differential result keeps each group's 5,000 lowest p, so a query can stop
  there, and the caption says when it does.
- **chromVAR per-cell motif activity** (about 10 minutes on PBMC 10k): motifs by variability, a groups x motifs
  heatmap, group against the rest, and a motif on the plot. The same z as `computeDeviations` given the same background
  draws (within 9.5e-7, 147 motifs); with R's own draws r 0.975, as for two R seeds.
- The co-pilot can use all three. Ask about a gene's links or a TF's motif by name and it gets that gene's or TF's
  numbers, not only the top results.
- The replay script reproduces links, motif hits, enrichment and chromVAR z from the original upload. Peak-gene links
  and motifs need the genome sequence: the first run downloads the UCSC file once (hg38 835 MB, mm10 715 MB, checked
  against a pinned sha256) and says so first. `GENOME_AUTO_DOWNLOAD=false` turns that off.

### Fixed
- Differential peaks: a p or FDR between about 1e-15 and 5e-11 reached the table as 0, while smaller ones were right.
- Replay scripts: helpers that shared a name replaced each other when one script needed both (`barcode_core` in DSB
  and V(D)J, `MIN_GENES` in the doublet classifier and spot composition, `BLOCK_VALUES` in DSB and ATAC), so such a
  script could stop or use the wrong setting. A test now refuses any such clash.
- The TF and pathway activity help no longer says "Human networks only": mouse datasets have used them
  through orthologs since 0.43.0.
- The pipeline step "UMAP/tSNE" is now "UMAP": the step never computed a t-SNE. A t-SNE already in the file is
  still shown.

## [0.55.0] - 2026-11-02

### Added
- **Differential peaks with a depth covariate** (Unified View > Accessibility): logistic regression as in Seurat's
  LR test, group ~ peak + total counts against group ~ total counts, now the card's default. P-values match Signac
  1.17.1 + Seurat 5.6.0 on PBMC 10k to 5e-6 of a log10 unit (all 11,898 cells, 50,000 peaks, 2 clusters). A score
  test covers every peak and a true likelihood ratio the top 10,000 per group; about 5 minutes for 21 groups at
  2 CPUs against about 40 s for Wilcoxon, with progress, Cancel and an estimate before the run. Against a true
  likelihood ratio on every peak, FDR sets agree within 0.4% in three RNA clusters and under-count by up to 11% in
  low-depth clusters. Why it matters: in a low-depth cluster Wilcoxon passed 37,183 of 37,367 peaks, the covariate
  18,223; in normal-depth clusters the two are within a few percent. Separated and non-converged fits are named and
  counted per row and per group. Differential results now reach the replay script; older results read as Wilcoxon.
  The API's default stays Wilcoxon.
- **Spot composition: leave reference labels out or merge them** (Reference labels). Labels whose names look like
  catch-alls are marked and never applied; a column over 50 types can run after merging, with a preview by shared
  name (58 types to 39, or 17, on a mouse brain reference). The over-share note gains "Leave out <label> and run
  again". Left-out and merged labels are recorded and reach the replay script, the self-check and the co-pilot. A
  run with nothing ticked is bitwise the same as before. On Visium HD 16 um mouse brain against a snRNA reference,
  clusters following their markers went from 8 of 14 with catch-alls to 12 of 14 with Unk out, though per-bin
  correlation with marker panels did not move (0.41): this fixes what clusters get called, not per-bin signal.

### Changed
- % mito on segmented Visium HD cells now says one rule removes cell types unevenly and points to Per cell type.

### Fixed
- Gene matching across Ensembl ids and symbols rebuilt a set for every gene: 35 s on 31,000 genes, now a fraction of
  a second, in spot composition and reference mapping.
- The spot composition reference check said "0 genes shared by name" with a false warning for a reference indexed by
  Ensembl ids; it now counts the genes the fit matches (18,785 on a mouse brain reference).
- The memory estimate counted doublet detection on segmented Visium HD cells, where the step is skipped: 9.4 GB and
  large mode, now 1.9 GB in memory.
- The doublet memory estimate extrapolates Scrublet's variable genes on channels over 5,000 cells instead of using a
  5,000-cell sample, which ran up to 30% high (lymphoid atlas, 73,000 cells: within 11.5% now).

## [0.54.0] - 2026-11-01

### Added
- **Cell and nucleus outlines on the tissue plot** (Embedding: Tissue, Draw: Points, Cells or Nuclei). A whole
  section stays points; the segmented shapes are drawn once under 30,000 cells (or 400,000 vertices) are in view,
  with the same colours, hover and lasso. Xenium outlines come with the folder upload, or attach later from the
  Transcripts tab; Visium HD's come with its segmentation. Breast Rep1: 167,780 cells, a centroid inside its own
  outline for 99.992%, polygon area against the cell table r = 0.9997. Outlines stay beside the dataset, so an
  h5ad export carries none and a merged dataset has none.
- **Xenium transcripts** (Unified View > Transcripts): attach `transcripts.parquet` and draw up to 8 genes as
  coloured dots, thinned evenly per gene with "Showing N of M" when a view holds over 300,000. Indexing keeps
  q >= 20 and leaves out control features. Breast Rep1: 42.6M rows to 34.4M in about 11 s, 297 MB on disk. Checked
  against the count matrix when it runs: all 313 genes equal, 32,102,954 on both sides. A view on 3 genes
  answers in about 100 ms for the whole section. Dots show every molecule of the run, whichever cells are filtered.
- **Space Ranger 4 cell segmentation**: a `segmented_outputs` folder reads as cells, with outlines and Space
  Ranger's clusters as `spaceranger_graphclust`. A folder holding both cells and bins asks which, cells first;
  Space Ranger 3 folders are unaffected. Mouse brain: 40,222 cells, 160 MB sent, 12 s to import, default run 144 s
  and a 2.5 GB peak under an 8 GB limit. The cells hold 69% of the same section's bin counts, so counts run about
  31% lower than in bins (per gene r = 0.99). Against Space Ranger's clusters ARI 0.55 (14 clusters against 26);
  top 100 Moran's I genes 78 shared with 16 um bins. Moran's I uses each cell's 6 nearest and domains a Delaunay
  graph, not the bin grid. % mito stays on: the default removes 11.5% of these cells, unevenly (neurons 0-3%,
  glia and vessels 18-35%). Doublet detection does not run on them: a tissue section has no droplets, and
  Scrublet's random pairs do not model the signal cells share with neighbours. Spot composition stays
  for spots and bins.
- **Co-occurrence** (Unified View > Domains): score against distance per label, a labels x labels heatmap and
  CSV, in microns where the pixel size is known, with a replay step. Equal to squidpy 1.6.5 up to its size limit
  (largest difference 1.9e-6 on DLPFC, 3.8e-6 on a lymph node, 7.5e-5 on 20,000 Xenium cells). Each section's pairs
  are counted apart. Xenium 160,675 cells 17 s at 4 CPUs; 393,543 Visium HD bins 164 s at 2 CPUs.
- **Spot composition warns about a label that takes far more RNA than it has cells**: at least 20% of the RNA and
  at least 5 times its share of the reference. On mouse brain against a snRNA reference, a catch-all "Unk" (2% of
  the cells) held 21%; without the catch-all labels the clusters follow their markers (Oligo 0.61 in the Mbp
  cluster, CA1, DG and CA3 in the Hpca cluster, thalamic neurons in the Tcf7l2 cluster). The run has no option to leave
  a label out: use a reference without it.

### Changed
- **Above 22,897 observations co-occurrence counts all pairs, which differs from squidpy's default.** Squidpy
  cuts such data by row into splits and averages the split pairs' scores, and far-apart splits add zeros. Xenium
  rows are in spatial order: r = -0.15 against all pairs, same-label short-range scores 0.22 against 3.94; with the
  rows shuffled the same mean gives r = 0.999. A checkbox reproduces squidpy's result (largest difference 7.6e-5).
- Xenium folder uploads send about 17 MB more, for the boundary files. Without them nothing changes.

### Fixed
- **A file with pandas `string` columns could not be saved at import**: a current muon `.h5mu` or an `.h5ad` with
  nullable strings passed detection, then failed at Create dataset with an anndata error. Pandas 3 writes such
  columns by default. They now save as plain text, for every format.
- The import wizard called Xenium cells "spots".

### Checked
- The gaps listed for v0.53.0: real mouse multiome (E18 brain, mm10) through ATAC, WNN, fragment QC and coverage (Gad2
  coverage peaks in the clusters with its RNA); an `.h5mu` written by muon, through the wizard, with the same peaks as the
  .h5; a native velocyto .loom attached in the UI, all 812 cells matched (its barcodes also built the test dataset,
  so a separately made dataset is still untested); coverage as SVG and PDF.

### Known limits
- Doublet detection's memory estimate runs high (13.2 GiB against 9.0 measured on 40,222 cells).
- The first zoom or pan at Xenium scale was once seen to stall for up to 1 s on a loaded machine; not reproduced
  in the live check (longest task 133 ms).
- Outlines are not in SVG or PDF export.

## [0.53.0] - 2026-10-31

### Added
- **RNA velocity** (Unified View > Velocity): steady-state and stochastic models, written natively because
  scVelo's stochastic mode crashes on this image's numpy. Against scVelo 0.3.3 on scipy 1.12, on pancreas
  and dentate gyrus in both modes: the same 2,000 variable genes and velocity genes, gamma r = 1.000000,
  per-cell arrow cosine 1.0000, confidence r = 1.0. scVelo on scipy >= 1.13 gives different arrows (its
  `T.setdiag(0)` no longer clears self transitions; steady-state median cosine 0.57 against the same
  model): scView keeps scVelo's documented formula. Dynamical velocity and latent time are not offered.
- **Spliced and unspliced counts**: layers of an uploaded .h5ad or .loom move to their own store on disk;
  a velocyto .loom, an .h5ad or a STARsolo Velocyto folder can be attached to an open dataset, matched by
  barcode with the number matched said. The loom and STARsolo routes were tested on files converted from
  pancreas to those formats, because no native velocyto file could be downloaded.
- **Arrows on the plot** with density and scale, a per-group table (coherence, length, unspliced share),
  phase portraits with the fitted line, and a warning on low confidence or a low unspliced share. The text
  says what the arrows are: direction on a 2D plot, not a fate claim. Large datasets run from the store and
  never read X (tested to 29,568 cells). PNG export keeps the arrows; SVG, PDF and split view do not.
- **10x Multiome ATAC.** Peaks stay on disk in their own store, from a 10x .h5 or folder, an .h5ad with
  Peaks features or an .h5mu; `per_barcode_metrics.csv` adds FRiP, TSS and mitochondrial fractions. ATAC-only
  uploads are refused with a sentence, and merging samples with different peak sets leaves ATAC out with a
  warning. Cells keep their peak counts through filtering. PBMC 10k: import 19 s, store 0.54 GB.
- **Accessibility tab**, for datasets with peaks: per-cell QC, LSI as Signac runs it (the same 136,705 peaks,
  components 1-30 |r| = 1.000 against Signac 1.17.1) with the depth check that left out component 1 (r = 0.79),
  ATAC UMAP and clusters; about 2 min 20 s and 0.9 GB on PBMC 10k.
- **WNN on RNA + ATAC** (weights r = 0.976 against Seurat 5.6.0), with the RNA weight by group. Protein WNN
  is unchanged.
- **Gene activity** from peak counts in the gene body + 2 kb upstream (GENCODE v32 / vM23, fetched on first
  use), as an overlay and against RNA. Marker r with RNA matches Signac's fragment-based activity within .05
  (MS4A1 .56 against .57); the caption says it is peak-based.
- **Differential peaks** per group with nearest gene, distance to TSS and class. This is a Wilcoxon test on
  TF-IDF, the fast default, with no depth covariate: top-100 enriched overlap with Signac's logistic regression
  was 78, 72 and 52 on three clusters, and most peaks can pass in a large group, as the caption says. CSV and
  an .h5ad of the peak matrix for Signac and muon.
- **ATAC fragments.** Attach `atac_fragments.tsv.gz` + `.tbi` to a dataset or at import, with no new dependency:
  checked for BGZF, barcode share and genome, streamed and never held in memory (2.4 GB in 5 s through nginx).
  Per-cell TSS enrichment and nucleosome signal equal Signac 1.17.1 on all 11,898 PBMC cells (r = 1.0000, medians
  5.79 and 0.668; 236M lines in about 90 s, under 0.5 GB). Nucleosome signal reads the whole file, unlike Signac's
  default sample. Fragment length ladder and TSS profile.
- **Coverage tracks** per group for a gene or a region up to 2 Mb, with peak ticks and gene arrows (gene bodies, no
  exons): MS4A1 peaks in the B-cell cluster.
- All of it reaches the co-pilot, session files and the replay script. Replaying PBMC 10k from the upload: LSI equal,
  ATAC clusters ARI 1.0, TSS enrichment equal to 1e-6; 7 min and about 7 GB.

### Changed
- **Differential expression on a dataset opened backed reads only the compared cells** of the tested matrix (Kang
  every cell: 210 MiB, was 317), with the same results. The doublet threshold reads no matrix; joint embedding plans,
  reference mapping and learned gates read only the counts.
- **These reads say what they need first and hold that room until they end**, so two at once count each other, and are
  refused with a sentence (507) unless the whole estimate fits. A load keeps the low-end rule. The re-filter and spot
  composition also estimate and hold, though they still read their data whole. At 6 GB during a PBMC run: DE on another
  dataset is refused naming the run, then works when the run ends, with no restart.
- **A run's pre-step check counts what other runs and reads will still grow by.** A step that fits only without them
  says to run again once they finish. Two runs checking at the same moment can both stop; there is no queue.
- **Load estimates count `.raw` in files from anndata before 0.7**, so estimates for such files rise (Kang included).
- Compact co-pilot guide: three hints dropped to make room (Cell Ranger, donor shuffle, power).

### Fixed
- Test that compared wall-clock seconds (learned gates, large) no longer flakes on a loaded machine.

## [0.52.0] - 2026-10-31

### Changed
- **A load during a pipeline run counts the run's remaining estimated peak.** Opening another dataset that
  would not leave the run that room is refused with a sentence naming the run ("A run on X needs about N GB
  more at its peak ... wait for it to finish, or cancel it, before opening Y") instead of the backend being
  killed at the run's peak; a smaller open still goes ahead backed. The reservation shrinks as the run's big
  steps finish. Reading a second dataset into memory for markers, scoring or a PAGA during a run gets the
  same sentence. Before: on 8 GB, a Visium HD 8 um run plus opening the 16 um section was an OOM kill (exit
  137); now the 16 um section opens backed and the run finishes (peak 6.5 GB of 8).
- **Dataset library: an open the server refuses now shows the server's sentence** (it showed nothing before).
- **Run estimates on Visium HD bins.** Leiden on bins takes 2.5x (full optimisation) and 1.75x (fast Leiden)
  the time of the same number of cells, measured on a mouse brain (16 um bins: 811 s where 309 s was
  estimated; 8 um bins with fast Leiden 192 s where 116 was). The fast-Leiden offer now fires for a 16 um
  section, and the minute notes compare against the new estimate.
- **Spot composition: the count floor is a run setting**, shown beside Fit with how many bins fall below it.
  8 um bins default to 25 counts (the 100-count floor scaled by bin area) instead of 100, so 0.7% rather
  than 29% of a mouse brain's bins stay unfitted: bins of 20-100 counts agree with their 16 um parent bin
  as bins of 100 or more do (Pearson 0.86-0.91 over types, RMSE 0.11-0.12). Spots and 16 um bins keep 100.
  The floor is recorded in the summary, the provenance step and the replay script.
- **Compare methods** (batch correction): when a slower method was within 2 points of the pick, the sentence
  says so, names the best and uses the table's percentages. The button's time estimate follows measured
  times (about 10 s on 13,836 cells at 2 CPUs) and counts BBKNN's one-off compilation on the first run only.
- **Claims check:** a list of cluster ids for a label ("Monocyte clusters are 1, 2, 6 and 7") is checked by
  rule against the cited labels (supported, or left to the judge; never refuted); plural group names
  ("monocytes") are recognised.
- **Co-pilot:** a cell-type label on a cluster that is mostly predicted doublets carries a caution.

### Fixed
- Marker tables no longer ask for the ambient RNA estimate on a large dataset (it was a 422 in the console on
  every Unified View open).
- The preflight step list kept the bin factor through reset to counts and filtering.

## [0.51.0] - 2026-10-30

### Added
- **Xenium output folders import directly.** Choose the folder: only the files scView needs are sent (about
  30 MB plus the 655 MB DAPI image on a breast cancer section), and cells sit on the image in microns. A
  zipped run (10 GB) is refused, with what to do instead.
- **Targeted panels get their own settings** (Xenium, and any spatial cell dataset of up to 6,000 genes;
  anything you set wins). Target sum is the median cell's counts, every gene counts as variable up to 2,000
  genes, the QC floor is 10 counts per cell, and there is no doublet or ambient step. Cell-cycle and stress
  scores are skipped with a sentence when under half of their genes are on the panel. Species comes from the
  panel's organism. Enrichment, GSEA and activity use the panel as their background and say how to read that.
  Control-probe share joins the QC metrics. On 167,780 cells x 313 genes the default run took 8 min (4 CPUs)
  and gave 15 clusters that match the expected breast types.
- **Visium HD.** Choose `binned_outputs/` or a `square_NNNum/` folder and pick a bin size: 16 um by default,
  8 um offered. A dataset holds one bin size. 2 um, the `.tar.gz` and `segmented_outputs` are refused with a
  sentence. Bins are drawn at their real width, all of them at any "Max" setting, and read as solid tissue
  (393,491 bins: 60 fps idle, about 22 fps while zooming in a headless browser). On a mouse brain section the
  default run takes 10-17 min on 16 um bins and 3.5 GB; 8 um bins need Leiden with 2 iterations to finish in
  time (10 min in an 8 GB limit), and read noisier.
- **Moran's I on large datasets and bins.** Variable genes in 4 s at 393,491 bins, all 18,821 genes in 17 s,
  from the gene store. Bins use the 4-neighbour grid. Against squidpy 1.6.5 on 16 um bins and 2,000 genes:
  Pearson 1.0, same top 200.
- **Spot composition accepts bins**, with a note on how to read a bin's fractions at its size. Bins under
  100 counts are not fitted, which is 29% of the bins at 8 um.
- **Spatial domains** (Unified View > Domains): stretches of tissue that look alike and have alike
  surroundings, by CellCharter's method (Varrone et al. 2024) on every technology and on large datasets. Choose
  k, or let Suggest k show a stability curve (about 11 min on a large section, with Cancel). Colour the tissue,
  Find markers, a name of your choosing, replay script, co-pilot facts. On human cortex section 151673, ARI to
  the manual layers at k = 7 over 10 seeds is 0.305 +- 0.043, against 0.202 +- 0.010 for the same model without
  space and about 0.22 for Leiden at about 7 clusters. CellCharter 0.3.7 itself gave 0.341 +- 0.054 on one
  input, and the two tools agree as closely as seeds of either one. Absolute ARI moves 0.28-0.40 with small
  input changes; the gain over no space held in every variant (0.09-0.17). Suggest k does not recover the 7
  layers on this section.
- **Neighbourhood enrichment** of any label column, = squidpy's `nhood_enrichment` (Pearson of z 0.9997 on
  Xenium, 0.9993 on the cortex section, counts identical). **Composition niches** (Schurch et al. 2020) on
  cells and 8 um bins, with a note on bins; refused on Visium spots and 16 um bins, where each position mixes cells.

### Changed
- **% mito is off by default on spots and bins, standard Visium included: this changes results for existing
  Visium users.** A spot or bin is a position on a capture grid, and a high % mito marks mitochondria-rich
  tissue. The old default removed 40% of layer 3 and 37% of layer 1 on the cortex section (20.7% of all spots,
  0.6% of white matter), 19% of a lymph node section, and 13% of 16 um bins on a mouse brain. It did not change
  domain accuracy beyond noise. Tick "Apply % mito to spots and bins" to keep the old rule; the filter preview
  says which applies. Fixed mode on bins defaults to min counts 1, and the high-mito nudge is gone there.
- Spatial datasets open on the tissue plot, and the header, insight, preflight and filter preview say spots or
  bins instead of cells.

### Fixed
- The replay script says any skipped step was skipped. It called cell-cycle and stress skips "unknown".

### Known limits
- Opening a second large dataset while a run is clustering can run the backend out of memory (8 um clustering
  plus opening 16 um at 8 GB). The run alone fits. Let a large run finish first.
- Xenium in large mode: Leiden agrees with the in-memory result at ARI 0.946, not exactly.
- The Leiden time estimate uses the number of cells only: 16 um bins took 811 s against an estimate of 122-489 s.
- Not yet read: Xenium cell polygons and transcripts, Space Ranger 4 segmentation.

## [0.50.0] - 2026-10-29

### Added
- **Learned gates, WNN and spot composition work on large datasets.** These three refused large datasets
  until now. Each gives the in-memory answer on the same data, and none reads a whole matrix into memory.
  - Learned gates read only the training cells from the counts file to train (up to 30,000 cells, about
    10 s at any dataset size). Applying a gate reads the counts file once for its genes; the apply check
    says how long that takes (about 2 min at 1.3M cells). Train, check and apply give identical results.
  - Spot composition takes a large single-cell atlas as its reference. The atlas's counts are read from
    disk in blocks, in one pass (two for the self-check). On the lymphoid atlas (73,000 cells) against the
    lymph node section, peak memory was 1.1 GB, against 2.8 GB with the atlas in memory, with the same
    dominant type in every spot. A large spot dataset works too, and its result is written into it in place.
  - WNN saves its graph, UMAPs and clusters into a large dataset in place. The run estimate now scales with
    the number of cells. Memory is reserved before the run starts. When Leiden to convergence would make a
    run very long, the card suggests clustering with fast Leiden first.

### Changed
- WNN looks up each cell's 200 candidate neighbours a block at a time from one index, instead of holding
  them for every cell at once. On 300,000 cells this stage peaked at 1.4 GB, down from 4.4 GB, in the same
  time. Above 4,096 cells the candidates now come from index queries, as in Seurat, instead of the index's
  own neighbour graph. On pbmc10k CITE-seq the RNA weights correlate at r 0.99994 with v0.49.0.
- A learned gate apply that would be refused (too few of the gate's genes, a column the gate does not own)
  is refused before any values are read.

## [0.49.0] - 2026-10-29

### Changed
- **Learned gates separate neighbouring types better.** Candidate genes are now also ranked against the
  unselected cells most like the selection. Against all the other cells, fold change mostly picks genes that
  split lineages, so a gate for a subset (naive vs memory CD4 T) lost to its neighbours. Out-of-fold F1 rose
  on 20 of 23 cell types (Kang control cells, paul15) and fell by more than 0.01 on none. A gate trained on
  Kang CTRL CD4 Naive T and applied to the stimulated cells went from precision 0.73 to 0.81 (recall 0.92 to
  0.94), CD8 T from 0.77 to 0.85 (recall 0.77 to 0.76). Training time is unchanged.

### Fixed
- Violin, stacked violin and WNN box plots grouped by cluster numbers labelled only every other cluster:
  Plotly read the numbers as a numeric axis. Group axes are now always categories.
- Opening Deconvolution, Spot composition, Traits, Upstream TF candidates or Spatial genes on a dataset with
  no result yet no longer logs a 404 in the browser console.

## [0.48.0] - 2026-10-29

### Added
- **Joint RNA + protein (WNN)** (Unified View > Protein): Seurat's weighted nearest neighbours for CITE-seq
  data. Each cell gets an RNA and a protein weight, and the two modalities are joined in one graph that gives
  a WNN UMAP, a protein UMAP (both in the Embedding menu) and WNN clusters (`wnn_leiden`). The RNA, protein and
  WNN layouts are drawn side by side; Colour by RNA weight uses a fixed 0-1 scale, with a violin and a box plot
  by group. Isotype controls are left out; fewer than 5 proteins is refused. Given the same neighbours, the
  weights match Seurat 5.1 to within 1e-8. On a 10x PBMC CITE-seq set (7,281 cells, 14 proteins) it takes
  about 40 s; RNA weight is highest in pDC, platelets and dendritic cells, which the antibody panel has no
  markers for.
- **Spot composition** (Unified View > Spot composition, Visium only): split each spot into the cell types of
  another dataset in the library, with an RCTD-style Poisson-lognormal fit and a per-section platform-effect
  correction, or an NNLS baseline. Click a type to colour the tissue by its share (fixed 0-100%), colour by
  the dominant type, or read a type x cluster heatmap; shares are of each spot's RNA, or estimated cells. A
  self-check scores both methods on synthetic spots from held-out reference cells. On a 10x lymph node
  section against a lymphoid atlas (73k cells, 33 types) a run takes about 1.5 min; on the self-check the fit
  has RMSE 0.056 against NNLS's 0.086.
- **Learned gates** (Unified View > Gates > Learned gates): train a small model on a selection (L1 logistic
  regression on log-normalised genes; the other cells come from the selection's own samples and conditions,
  so a CTRL selection does not learn the interferon response), with out-of-fold AUROC, precision, recall and
  F1, the genes for and against, and Select disagreements to review where model and selection differ. Saved
  gates can be exported, imported, checked on any dataset (genes found, one-to-one orthologs between human
  and mouse, shift and interferon warnings) and applied as an in/out column and a gate score with per-sample
  counts. A Kang CTRL B-cell gate found STIM B cells with precision 0.95 and recall 0.87.
- **Convert an R object before creating the dataset** (Data): an .rds dropped with other samples can be
  converted with Convert now, so the merge plan shows its gene and cell overlap; creating the dataset then
  reuses that conversion (a file that changed since is converted again).

### Changed
- The bulk deconvolution over-share flag compares an estimate with the type's largest share in one donor
  (donors with at least 50 cells), not its share of all reference cells: on Kang, NNLS on a matched CTRL
  reference no longer flags CD8 T cells, which one donor has at 12%.

### Fixed
- Dark theme: the selection chip, Ask about this selection and the Spatial genes tab no longer show light text
  on light backgrounds.

## [0.47.1] - 2026-10-29

### Changed
- **Run time estimates no longer shrink with more CPUs.** They were measured at 2 CPUs and scaled down for
  more, but a benchmark of every step at 2, 4 and 8 CPUs (same files, 100k and 250k cells) found the times
  within about 15% of each other. On an 8-CPU machine the estimates were therefore about half the real time:
  a 1.3M-cell run came in at the top of its range, and its import (7.8 min) above it ("1 to 6 min"). Re-scored
  on 8 benchmark runs, no step over a minute now runs past its range (10 did before).
- **While a step runs, its minute note gives the estimate**: "Clustering: 12 min so far (estimate: roughly 4
  to 16 min)". A step past its estimate says so, and that it is still working; times over an hour read as
  "1 h 5 min".

## [0.47.0] - 2026-10-29

### Added
- **Bulk deconvolution** (Unified View > Deconvolution): estimate the cell-type fractions of your bulk RNA-seq
  samples with the open dataset as the reference: MuSiC-style weighted NNLS across donors, or NNLS on marker
  genes. Bulk files are TSV, CSV or GCT (and .gz), genes in rows or columns, symbols or Ensembl IDs; log values
  are refused and TPM is flagged as relative. **Reference cells** restricts the reference to one condition
  (e.g. stim = CTRL): a reference that mixes conditions gave untreated Kang samples 50-60% megakaryocytes
  (true 2%), and 2.4% once restricted. An estimate at least 20% and 5x its share of the reference cells is
  flagged. A self-check holds each donor out in turn (r, RMSE and bias per cell type; about 20 s on Kang), and
  Download reference writes the dataset for MuSiC, BayesPrism and CIBERSORTx.
- **GWAS trait links** (Unified View > Traits): GWAS Catalog traits linked to each group's marker genes
  (hypergeometric p, BH-adjusted, plus an empirical p from random gene sets matched on how many traits each gene
  has; overlap counted in loci), with the caveat that a hit is a lead, not a causal gene. One gene's traits with
  PubMed ids and GWAS Catalog / Open Targets / GTEx links, also in the gene report. Release 2026-10-22,
  downloaded once (about 93 MB, sha256-pinned) or placed by hand; `GWAS_AUTO_DOWNLOAD=false` turns fetching off.
  Mouse genes map to human through HCOP orthologs.
- **Reactome pathway diagrams** (Enrichment > Diagram): Reactome rows in ORA and GSEA results (Enrichment and
  DE tabs) open the pathway's diagram coloured by a DE result's or a group's log fold change; a complex or set
  shows the mean of its measured genes and "k of n changed". A pathway without its own diagram opens in its
  nearest ancestor's, outlined; an overview lists its sub-pathways. Diagrams are fetched on first use (release
  98, cached; only the pathway id is sent), `REACTOME_AUTO_DOWNLOAD=false` turns that off. SVG/PNG downloads
  carry the legend and the Reactome CC BY 4.0 attribution. KEGG and WikiPathways rows link out.
- **Upstream TF candidates** (Activity > Upstream TF candidates): JASPAR 2024 motifs enriched in each group's
  marker-gene promoters (UCSC 1 kb upstream; hypergeometric, BH per group), with whether the TF is expressed and
  whether CollecTRI activity agrees. The motif table is built once per genome (hg38, mm10, mm39) in about a
  minute; `TF_MOTIFS_AUTO_DOWNLOAD=false` turns fetching off. GC-rich motifs are flagged, and results say
  "candidate".

### Fixed
- History: spatial steps named in words; an import's tissue positions and R conversions shown as readable
  chips (a list of R conversions showed as "[object Object]").
- Folder uploads skip Space Ranger's alignment images and tables before sending, as the server does.
- The import wizard names Space Ranger spatial files by what they hold, not "raw matrix part".
- A step that only adds a result (Moran's I, gene programs, pseudotime, label columns and the new analyses) no
  longer marks other stored results as stale.

## [0.46.0] - 2026-10-28

### Added
- **Spatial transcriptomics (10x Visium).** Import Space Ranger output: the `outs/` folder, or the
  filtered matrix with its `spatial/` folder or `spatial.tar.gz` (Space Ranger 1.x and 2.x position files,
  and parquet). Each sample becomes a tissue section with its H&E image, spot positions and spot size;
  several samples merge into one dataset with one section each. Spatial `.h5ad` files from scanpy, squidpy
  or spatialdata-io (`obsm["spatial"]` or `X_spatial`) open with their sections and images. Visium HD and
  Xenium bundles are not supported yet and say so.
- **Tissue view.** Unified View > Embedding > Tissue draws the spots on the tissue image at their real size,
  with the same colouring, gene overlays, lasso and cluster highlight as the UMAP. A Section chooser shows
  one section at a time; the image can be hidden or faded, and a sharper image loads when you zoom in. The
  cluster map beside it keeps the UMAP, so a lasso on the tissue shows there and a cluster chip lights up
  its spots.
- **Spatially variable genes.** Unified View > Spatial genes ranks genes by Moran's I on the spots'
  neighbours (Visium grid or 6 nearest neighbours, never across sections), equal to squidpy's defaults
  (Moran's I within 4e-13 on the 10x lymph node). All 22,036 genes of the lymph node take about a second.
  Click a gene to colour the tissue by it; the table downloads as CSV, the run is in History and the
  replay script, and the co-pilot knows the top genes.
- **Seurat `.h5Seurat` files** (SeuratDisk) import directly, without R: counts, cell metadata, reductions,
  an antibody assay as proteins, and Visium images.
- **A Seurat or SingleCellExperiment `.rds` in the import wizard with other files**: it is converted when
  the dataset is created, and can be merged with a 10x sample. Seurat Visium objects (`.rds` or
  `.h5Seurat`) keep their tissue image, spot positions and scale factors and open in the tissue view; the
  spot grid that Seurat 5.1 objects no longer store is recovered from the spot positions.

### Changed
- **Doublet detection is skipped on Visium spots** (each spot holds several cells); the assessment says
  "Not applicable" and History and the replay script show the skip.
- Seurat objects whose genes are in a Spatial assay (Visium) now convert.

## [0.45.2] - 2026-10-28

### Fixed
- **Clustering, UMAP and marker genes failed on datasets over 8,192 cells in v0.45.1.** Its images were
  built with pynndescent 0.7.0 (released the day before), whose neighbour graphs scanpy 1.11.5 cannot read
  ("'csr_array' object has no attribute 'getnnz'"), so the neighbours step failed and the steps after it
  were skipped. pynndescent is now kept below 0.7, and a test builds a graph over 9,000 cells. v0.45.0
  and earlier are not affected.

## [0.45.1] - 2026-10-27

### Fixed
- **Dark theme: split and Compare panel titles** (CTRL / STIM, or the compared colourings) were
  near-black on the dark plot background. Titles and frames now take the plot background's colours.
- **Dragging the Unified View split was slow once results were open**: every pointer move resized and
  re-rendered the open plots (one drag took 79 s with DE results on screen). The drag now moves a guide
  line and the columns resize once, on release.

## [0.45.0] - 2026-10-27

### Added
- **Co-pilot plans**: a request with several steps ("re-cluster at resolution 1, then compute markers
  and run enrichment for cluster 3") becomes one numbered checklist with one Run plan button. Steps run
  in order; after each, a rule checks its result (the new clustering exists, markers exist, the group
  has enrichment) before the next starts. A failed check stops the plan and names the step and why;
  "Cancel after this step" stops it cleanly. Steps that use a re-cluster's result pick up the column
  it really wrote. On Kang the 3-step plan above finished in 40 s. No extra model call: the same
  single request that reads actions returns the steps.
- **Co-pilot actions in Compare and DE**: "open the composition test with condition stim, groups
  cluster and samples Donor" opens Compare > Composition with those pickers set (also Milo,
  miloDE, Joint embedding, Plan a study); "set up DE between STIM and CTRL in stim" fills DE >
  Between groups. Columns and groups are checked against the dataset; nothing runs until you click.

### Changed
- **Built-in marker sets** (offline marker-score annotator) gain plasmacytoid DC and CD16+ monocyte
  sets; JCHAIN left Plasma cell, CD68 left Macrophage, LILRA4 left Dendritic cell. On Kang, pDC were
  called "Plasma cell" and CD16 monocytes "Macrophage"; now 103 of 106 pDC and 940 of 974 CD16 Mono get
  their own label. Runs recorded before replay with the sets they used.
- **Author labels first**: pickers rank a dataset's own cell types before scView's annotations
  (CellTypist and the LLM next, marker scoring last), and a column called "cluster" whose labels are
  names counts as cell types. Kang now opens coloured by its author cell types, and Composition, Milo
  and the study planner default to them.
- **Compare two datasets**: a column per dataset, defaulting to the pair whose labels overlap most, so
  label transfer's `<prefix>_transfer` column compares with the reference's cell types.
- Co-pilot: an up-to-date feature guide with each feature's real path (a test checks every path
  against the interface's labels), sent in short form (about 350 tokens) with dataset answers too, so
  the co-pilot no longer denies features it has or invents menus.
- Co-pilot routing: a question naming a stored result (cytokine response, ligand-receptor) or pointing
  at the open dataset ("the monocyte cluster", "cluster 3", "the QC that was done") always uses the
  dataset's facts, even when the question classifier picks the literature.
- Co-pilot composition fact uses the newest stored run that still fits the data and was tested, and
  names its columns; one invalid run no longer hides earlier valid ones.

### Fixed
- QC driver facts gave a PC's share of the variance in the kept PCs (PC1 22% on Kang) while the PCA
  card gave the share of the total (2.9%); both now give the total.
- Claims check: the model judges a claim against the fact lines that name its clusters, groups or genes
  (a correct "Monocyte clusters are 1, 2, 6 and 7" was refuted because the judge saw clusters 0-5
  only); "13 labels, e.g. CD14 Mono" no longer reads as CD14 Mono having 13 cells.
- Layout-stability thumbnails follow the dark scatter background; tall cytokine heatmaps label their
  columns on top, where they are in view.

## [0.44.0] - 2026-10-26

### Added
- **DE audit** (DE tab, "Audit this result"): reruns the cell-level test with the condition shuffled
  across whole donors (each shuffled side keeps the same mix of real donors) and runs pseudobulk on the
  same cells (Squair et al. 2021; Zimmerman et al. 2021). Shows the shuffle counts against the observed
  count, a QQ plot, the overlap with pseudobulk (Jaccard, Spearman of fold changes) and the genes
  significant at cell level only with their top-donor share, and gives one plain sentence that names its
  rule. On Kang CD14 Mono STIM vs CTRL: 1,836 genes at cell level, 10-663 in 20 shuffles (median 310),
  1,565 also in pseudobulk. Streams with Cancel; kept in History > Results; the co-pilot cites it.
- **ssGSEA and GSVA** gene-set scoring beside score_genes and AUCell; both give R GSVA 2.6's scores on a
  dense matrix (Spearman 1.000 over 50 Hallmark sets on 2,000 Kang cells). GSVA is exact up to 20,000
  cells and takes the non-zero values from a fixed 20,000-cell reference above that; both refuse above
  300 million non-zero values. On sparse single-cell data (90% or more zeros) GSVA's note says its scores
  bunch together. "Check depth" compares all four methods.
- **DSB protein normalisation** (Mulè et al. 2022) for CITE-seq: antibodies against their background in
  the raw matrix's empty droplets, per-cell noise removed with the isotype controls; the same values as
  R dsb 2.0.1 to 1e-12 on 10x pbmc_10k_protein_v3. The Protein tab chooses CLR or DSB; "Compute DSB"
  shows the droplet window, the droplets it takes and the isotypes. Without a raw matrix it runs dsb's
  ModelNegativeADTnorm and says so. Recorded in provenance and the replay script. A raw matrix attached
  before this version needs attaching again for DSB.

## [0.43.0] - 2026-10-26

### Added
- **Mouse gene sets**: a mouse dataset uses MSigDB's mouse collections (2026.2.Mm: MH Hallmark, M2
  Reactome / WikiPathways, M5 GO, M8 cell types by default; M1, M2 CGP / BioCarta, M3, M5 MPO and M7 to
  pick), downloaded on first use (~27 MB, sha256-pinned; `MSIGDB_AUTO_DOWNLOAD=false` turns it off).
  The collection tree lists mouse sets first. Enrichment results, pipeline provenance, the replay script
  and the co-pilot name the gene sets' species ("mouse MSigDB MH"). Human installs download nothing new.
- **Human resources on mouse data through orthologs**: human gene sets, CollecTRI / PROGENy (TF and
  pathway activity, refused on mouse before) and CytoSig are mapped to mouse genes through the HGNC HCOP
  ortholog table with decoupler's rules (3+ supporting resources, at most 5 orthologs per gene). On paul15
  the mapped CollecTRI edges are identical to decoupler 2.2 (43,226 of 43,226) and per-cell activity
  agrees (Pearson 1.0000). The table (3.5 MB, CC0) is fetched once; `ORTHOLOGS_AUTO_DOWNLOAD=false` turns
  that off, and without it human gene sets fall back to matching symbols ignoring case.
- **Co-pilot knows what is on screen**: the open Unified tab, the DE result or enrichment shown and the
  open Compare tab go with each question, so "Explain this result", "What should I do next?" and "Write
  the methods paragraph" use the dataset's facts (the methods text comes from the recorded steps only).
- Co-pilot answers render Markdown tables, and code inside bold text.

### Changed
- Marker database check: a label whose database markers overlap its top markers no more than chance
  would give (p >= 0.05, often one shared gene) reads "weak support", not "supported" (Kang's B activated
  and T activated).

### Fixed
- Co-pilot routing: the question classifier's reply was cut off and failed to parse about half the
  time; questions naming a stored result (cytokines, miloDE, pseudobulk, CRISPR, programs, communication)
  keep that result in the answer.
- Co-pilot facts: median genes per cell after QC filtering, every cluster a label spans, why a Mixscape
  target with no KO cells has no DE genes, and the condition column are stated so the answer uses them.
- Claims ledger: a sentence sharing one gene name with a fact but saying more is "uncertain", not
  "supported"; a count before a bracketed list no longer misreads as a refutation.
- The first marker database download (several minutes) no longer ends in "backend did not answer": it
  runs in the background while the card shows progress.
- Error notes show the server's reason instead of `API error 400: {"detail": ...}`.
- Step names keep their abbreviations in History and Provenance ("QC metrics", "PCA").
- A scattered small cluster's label sits on one of its cells instead of in empty space.

## [0.42.0] - 2026-10-26

### Added
- **Download as Seurat or SingleCellExperiment (.rds)**, built by the R converter: counts and
  log-normalised layers kept sparse (scaled values never go into `data`), cell metadata with factor levels
  in order, PCA (with loadings) / UMAP / t-SNE, the active clustering as Idents, CITE-seq antibodies as an
  ADT assay or altExp, VariableFeatures, provenance in `Misc(obj, "scview")` / `metadata(sce)$scview`.
  Each export reports what went in and what was left out (neighbour graphs, stored results); Seurat's
  `_` -> `-` gene renames are counted. Checked with Seurat 5.6 (readRDS, DimPlot, FindMarkers,
  NormalizeData matches the data layer) and scater; counts round-trip exactly on Kang and paul15.
  Refused up front when the converter is not running or lacks the memory; the .rds is removed after
  download or after an hour.
- **Cell Ontology terms for labels**: the Label selection box searches the Cell Ontology (name, synonym,
  CL id) and attaches the term; free text still works. A "Cell Ontology…" card under the legend shows each
  label's term (linked to OLS), suggests terms from the label names to accept or reject (12 of Kang's 13
  labels get the right term first), marks labels unknown, and keeps the mapping through rename, merge and
  undo. The CELLxGENE export can write `cell_type_ontology_term_id` from a complete mapping you confirmed
  (opt-in); .rds exports carry it as `<column>_ontology_term_id`. The co-pilot states the term you chose.
- **Ask the co-pilot about a selection**: a lasso, gate or filter selection becomes chat context ("What
  are these cells?", "How do they differ from the rest?"), with its composition, QC and genes vs the rest
  cited as `[result:selection]`; a removable selection chip, "Ask" on the toolbar's selection, and "Run DE
  for this selection vs the rest" opening the DE tab set up (it does not run by itself). A stored DE on
  the same cells is cited.
- **Volcano gene names**: the DE volcano names the top 10 (or 20, or none) genes, split between up and
  down and stacked so they do not overlap.

### Fixed
- Plots sized by their own height (the DE volcano, gene-detection histogram and others) collapsed to
  nothing after switching between light and dark.
- Co-pilot routing: with the literature search off, a question is never sent to literature alone (it was
  answered with no sources about half the time); cell-type and gene questions keep the dataset's stored
  results; the same question gets the same route.
- p-values below what a float holds read "< 1e-300", not "0.00e+0".
- Mixscape's "Within" choice shows the setting of the saved run.
- Info tips no longer pop open, clipped, when a dialog opens.

## [0.41.0] - 2026-10-26

### Added
- **CRISPR screens**: "CRISPR Guide Capture" features are kept at import as `obsm['guide_counts']`
  (up to 2,000 guides) instead of being dropped, and Cell Ranger's
  `crispr_analysis/protospacer_calls_per_cell.csv` is read with the matrix.
- **Perturbation tab** (Unified View, shown when a dataset has guides): each cell's guide and target from
  Cell Ranger's calls, guide counts (minimum UMIs + top guide at least 2x the second) or an obs column,
  with an editable guide → target mapping and the control label (NT, non-targeting, ... detected). Writes
  `guide_call`, `perturbation`, `n_guides`.
- **Mixscape** (Papalexi et al. 2021): each target's cells split into KO / NP against the control cells,
  optionally within a replicate column; per-target KO fraction and DE genes, posterior histogram, LDA view
  and a one-click DE of KO vs control. On Papalexi 2021 (20,729 cells, 25 targets) it gives the same class
  as pertpy 1.0.3 for 99.3% of cells (99.65% with exact neighbours). Streams with Cancel.
- **Neighbourhood DE (miloDE-style)**, Compare > Composition: DE between two conditions inside each
  neighbourhood of the cell graph (second-order kNN neighbourhoods, miloDE's defaults), donors as
  replicates, no clustering needed; per gene the number of neighbourhoods where it is DE and its median
  log2FC, per-neighbourhood log2FC on the embedding, miloDE's overlap-weighted spatial FDR. Neighbourhoods
  where one condition holds under 10% of the cells are flagged. On Kang it runs in under a second.
- **Pseudobulk models**: pseudobulk DE can use limma-voom or voomByGroup (a mean-variance trend per group)
  besides DESeq2; TMM, voom and the moderated t match edgeR 4.10 / limma 3.68 to ~1e-9 on Kang.
- A variance check plots each group's voom trend and flags a group whose donors vary 2x or more,
  suggesting voomByGroup.
- **RUV** option (k = 0-3) adds unwanted-variation factors from empirical control genes (the 1,000 least
  associated with the condition in a first limma-voom pass) to any pseudobulk model, with a table of how
  each factor tracks condition, depth, cell count and donor, and a warning when it follows the condition.
  In a paired design the donor effect is removed from the control genes first, so a factor is not the
  donor effect again.
- Mixscape runs (`mixscape`) and neighbourhood DE (`milode`) are kept in History > Results and export as
  .tsv.gz; stored DE results name the pseudobulk model and RUV k. The co-pilot cites `[result:mixscape]`,
  `[result:milode]` and `[result:pseudobulk]`.

### Fixed
- The co-pilot did not see Mixscape, pseudobulk or neighbourhood-DE results for questions like "which
  knockouts worked?": its router now sends them to the facts that hold them.

## [0.40.0] - 2026-10-26

### Added
- **Results store** (phase 1): results that were kept in process memory only now also go to disk under
  `DATA_DIR/results/{dataset_id}/`, so they survive a backend restart and a dataset dropping out of the
  cache: DE results (GSEA, DE overlap and activity by result id), ligand-receptor runs and condition
  comparisons (CSV download by id), composition tests, Milo, merge and count-split checks, cytokine
  response, off-target (HPA), protein interactions and study plans. The co-pilot's facts about them come
  back with them. Tables are Parquet (zstd; floats narrowed to float32, p-values kept float64), arrays
  compressed NumPy, structure gzip JSON.
- Each stored result records the cells it was computed on and the recipe's step count: one computed on
  an earlier version of the dataset (cells filtered, pipeline re-run) is marked stale and not given to
  the co-pilot as a current fact.
- Retention: per kind and key the newest result plus the last 3 it superseded, at most 20 current results
  per kind, at most 200 MB (compressed) per dataset, oldest first. Deleting a dataset deletes its results.
- `GET /datasets/{id}/results` (filter `kind=`), `GET /datasets/{id}/results/{rid}`,
  `DELETE /datasets/{id}/results/{rid}`.
- **Earlier markers and enrichment kept** (results store phase 2): when a save replaces a column's markers
  or a group's enrichment with a different result, or drops them (markers re-run, groups merged,
  re-clustering), the saved ones go to the results store (last 3 per column / group). The h5ad keeps only
  the current one, so exports are unchanged. Each records the column's labels it was computed on: after a
  relabel, merge or filter it is stale.
- `POST /datasets/{id}/results/{rid}/restore` makes an earlier markers or enrichment result current again
  (the one it replaces is kept); refused (409) for a stale one, whose groups no longer exist. Restored
  markers drop the column's enrichment (computed from other markers; also kept).
- **Stored results list** (History, below "Reproduce this analysis"; results store phase 3): every stored
  result with its kind, when, size and badges (earlier = superseded, pinned, "earlier data" = stale), a
  kind filter, and per row: download as `.tsv.gz`, pin (kept until deleted) / unpin, delete (with a
  confirm), and for earlier markers and enrichment "compare with current" (top 50 genes shared per group;
  significant terms shared) and "make current again".
- **Exports**: one result as `.tsv.gz` (header + rows, 6 significant digits, p-values `%.3e`), all results
  as a zip with `manifest.json` (parameters, version, fingerprint, stale flag), and "Dataset + stored
  results (.zip)" in the Download menu (the h5ad beside a `results/` folder). The plain h5ad download is
  unchanged. `GET /datasets/{id}/results/{rid}/table.tsv.gz`, `GET /datasets/{id}/results.zip`,
  `GET /datasets/{id}/export/h5ad?with_results=true`, `POST /datasets/{id}/results/{rid}/pin`,
  `GET /datasets/{id}/results/{rid}/compare`.

### Fixed
- Enrichment overlap genes are listed sorted: their order came from gseapy's sets and changed between
  backend processes.

## [0.39.2] - 2026-10-26

### Added
- Elsewhere in the body (HPA): a tissue-level menu. Tissues with the nine brain regions as one brain
  (default, as before), every tissue with brain regions separate, or HPA's 15 organ groups (a gene shared by
  stomach and colon is one organ). Categories, verdicts and the heatmap follow the level; `level=` on
  `GET /datasets/{id}/markers/offtarget` and `POST /offtarget/genes`.

### Changed
- The HPA tissue table is stored with regions unmerged (`hpa_tissue.v2.npz`); an existing 0.39 store is
  rebuilt from the downloaded zip on first use, without a new download.

## [0.39.1] - 2026-10-25

### Fixed
- `docker-compose.yml` pins the 0.39 images: the 0.39.0 tag still pinned 0.38.0, so its images were not
  published. 0.39.1 carries everything listed under 0.39.0.

## [0.39.0] - 2026-10-25

### Added
- **Cytokine response** (Activity, "Cytokine response"): which of 43 cytokines a group is responding to, by
  CytoSig's signature matrix (Jiang et al. 2021): ridge regression of each group's profile on the signatures
  with a gene-shuffle z-score. Each group against the others or all cells, or condition A vs B within each
  group. Heatmap (z, diverging about 0) and top cytokines per group. Matches the reference CytoSig on Kang
  STIM vs CTRL (Spearman 0.9997; IFN1 tops every cell type). Works on large datasets; mouse genes are matched
  by upper-cased symbol. The signature (GPL-3.0 data) is fetched on first use and checked against a pinned
  sha256, not shipped; `CYTOSIG_AUTO_DOWNLOAD=false` turns fetching off. `POST /datasets/{id}/cytokines/activity`.
- **Elsewhere in the body** (Markers tab, "Look up in HPA"): where a group's markers are expressed across
  43 human tissues and 154 cell types (Human Protein Atlas v25.1, CC BY-SA 4.0), with HPA's specificity
  categories rebuilt from its published cut-offs, tau, top tissues and cell types, a tissue heatmap and a link
  per gene. Markers that are specific among your clusters but broadly expressed in the body are flagged.
  Tables fetched on first use (~21 MB; `HPA_AUTO_DOWNLOAD=false` turns this off).
  `GET /datasets/{id}/markers/offtarget`, `POST /offtarget/genes`.
- **Protein network** (DE results, Markers, Programs): STRING v12.0 physical interactions (CC BY 4.0) among a
  DE result's genes, a group's markers or a gene program: hubs, Leiden communities, colour by community or
  log2FC, a sortable node table with CSV, and a connectivity test (observed vs expected links from 1,000
  random protein sets matched on degree). Score cut-off 400/700/900, at most 500 genes. STRING is fetched once
  per species and queried locally: gene lists never leave the machine (`PPI_AUTO_DOWNLOAD=false` turns
  fetching off). `GET /datasets/{id}/ppi`, `POST /ppi/network`.
- **Copy number** (new Unified View tab): inferCNV-style inference of large-scale gains and losses from
  expression against normal reference cells you pick, to tell malignant from normal cells in tumour samples.
  Genome heatmap by group, CNV score per cell (`cnv_score`), CNV clusters (`cnv_leiden`) with gain/loss
  calls, both colourable on the embedding; Cancel, preflight estimate, replay step. Matches infercnvpy 0.6.1
  per-cell scores (Pearson 0.9999996 on maynard2020_3k). Gene positions for human GRCh38 and mouse GRCm39
  (Ensembl 113) ship with scView. A counts layer that holds log values gets a warning (logged twice flattens
  the profile). `POST /datasets/{id}/cnv/run`.
- The co-pilot can cite all four (`[result:cytokines]`, `[result:offtarget]`, `[result:ppi]`, `[result:cnv]`),
  and routes questions about them to the dataset's results.

### Changed
- A run stops before normalisation, with nothing saved, when the expression values are already
  log-normalised and there are no raw counts: normalising and logging them again flattened every difference
  between cells (a tumour's copy-number score fell from 5.5x to 1.8x the reference). The message says which
  steps to leave out.

### Fixed
- Uploads stored in half precision (float16) failed in marker genes ("float16") and highly variable genes
  ("infinity"); their matrices are now stored as float32 before a run.

## [0.38.0] - 2026-10-25

### Added
- **Keyboard and screen-reader access.** A skip link, page landmarks, Unified View and Compare tabs that switch
  with the arrow keys, named buttons and form fields, and sortable table headers that work by keyboard and say how
  the table is sorted. Dialogs and menus take focus, keep Tab inside, close on Escape and give focus back; the
  co-pilot drawer does the same.
- **Legend by keyboard:** one Tab stop; arrow keys move between groups, Enter selects one, Shift+Enter picks
  several; screen readers hear which groups are selected.
- **Scatter by keyboard:** focus the plot, then arrows pan (3D: turn), + and - zoom, 0 fits, and S draws a
  selection rectangle (Enter sets each corner). Screen readers hear what is plotted and every selection change
  ("2,481 cells selected"). **By group** in the Unified View toolbar selects the cells of chosen groups without
  the mouse. Plotly figures carry a description (kind, title, axes) and small ones a hidden data table; with the
  system's reduce-motion setting on, animations are skipped.
- **Accessibility mode.** A switch in the sidebar (under Theme), off by default and remembered per browser. On:
  text contrast to WCAG AA (4.5:1) in the light theme, no text under 12 px in the app or the plots, an always
  visible focus outline, underlined links, no transitions, and the colour-blind-safe group colours. Off, the
  settings stay tuned for clarity and looks.
- **Count-split check of markers** (Markers tab, "Check with count splitting"). Marker p-values test genes on the
  same counts that made the clusters, so they always look significant. The check splits every cell's counts in
  two (negative binomial count splitting, Neufeld et al. 2024), re-runs the clustering on one half and tests the
  markers on the other. Per group: markers kept and lost, ordinary vs split p-values, a "Split adj.P" column in
  the table, and a group's split result can be kept for GSEA and the DE overlap. Groups that differ only by
  sequencing depth can still pass (depth survives the split); the panel says so.
  `GET`/`POST /datasets/{id}/markers/count-split`, `POST .../count-split/keep`.
- **Compare communication between two conditions** (Communication, "Compare conditions"). Both conditions are
  scored on the same groups, ligand-receptor pairs and genes; each interaction is gained, lost, changed (means
  1.5x apart) or unchanged. A sender x receiver heatmap of the change in count or strength, a ranked table with
  filters and CSV, and one interaction per donor. P-values come from donors only (paired Wilcoxon, or Welch's
  t when donors differ; BH); without donors the differences are shown without p-values. Groups under 10 cells
  in either condition are dropped and listed; groups whose share changes 2x or more are flagged.
  `POST /datasets/{id}/communication/compare`.
- **Minimal marker panel** (Markers tab): the fewest genes that still tell the groups apart, by greedy forward
  selection on cross-validated balanced accuracy (logistic regression, 3-fold CV), compared with the top markers
  taken in turn. Required and excluded genes, an optional gene-list restriction (e.g. surface proteins), an
  accuracy curve, confusion matrix, dot plot, copy and save as gene list. Works in large-dataset mode.
  `POST /datasets/{id}/markers/panel`.
- **Object card** (Data Assessment): what the AnnData file holds (matrices with value range and a counts check,
  obs/var columns, embeddings and graphs, a uns tree), read cheaply at any size, with flags for missing counts,
  scaled X without .raw, NaNs, duplicate names and empty columns, and a copy-as-text summary.
  `GET /datasets/{id}/structure`.
- The co-pilot can cite the count-split check, the condition comparison and the marker panel.

## [0.37.0] - 2026-10-25

### Added
- **Dark mode.** Light / Dark / System (default System, follows your computer) in the sidebar, remembered
  per browser. Panels, tables, warnings, inputs, every plot and the cell scatters follow it. Downloaded
  figures stay on a white background in both themes.
- **SVG and PDF figures of the embeddings.** The scatter and the split / compare / gene grids download as
  PNG, SVG or PDF from one menu (Download figure / Download grid): vector points in the screen's order and
  colours, group labels, axis names, a title, the legend and the colour-scale caption, on white. Above
  50,000 points the SVG/PDF puts the points in as one high-resolution image with every text vector (what
  journals accept); all-vector is still offered. The PAGA graph drawn over the trajectory plot is in the PNG only.
- **PDF for every plot.** Each Plotly plot has a third download button, Download PDF (vector, text selectable).
- **Weighted density (Nebulosa-style).** A "Density" toggle in the Expression subtab colours the plot by an
  expression-weighted kernel density of the gene (plug-in bandwidth = `ks::hpi`), which shows where a gene is
  expressed even when few cells have reads; "Joint density" of several genes is in Several genes. Computed
  once over all cells, so split panels share one bandwidth. `GET /expression/density`.
- **Tracks plot** (scanpy's tracksplot): Marker Genes > Heatmap > Tracks draws each gene's per-cell value as a
  filled track in the groups' colours, each gene on its own scale.
- **Linked selection.** The current selection shows in the violin and stacked violin (a Selection group), the
  marker dot plot and heatmap (a Selection row), the composition bars (selected share per bar), the QC
  histograms and scatter, and gene vs gene; a "Selection: n cells [Clear]" chip marks each panel that reacts.
  Box or lasso select on the QC counts-vs-genes scatter and on gene vs gene sets the selection.
  `POST /expression/by-group` takes a cell set (large-dataset mode too).
- **Lasso in split view.** The lasso works in any split panel (that panel's cells, or every panel's); each
  panel's title says "n selected of N"; the compare and gene grids take the lasso too.

### Changed
- The scatter's Light/Dark background follows the app theme until you pick one in the plot toolbar.

## [0.36.0] - 2026-10-25

### Added
- **Import from CELLxGENE Discover.** Search Discover's public catalogue by text, organism, tissue,
  disease, assay and size, and import one dataset pinned to its version id. The download shows
  progress, can be cancelled, and is checked for disk space and memory first. The working copy uses
  gene symbols, and its source and citation are kept in the History, the methods text and the report.
- **Label from a reference dataset.** Label this dataset's cells from another library dataset's
  cell-type column: project onto the reference, integrate with it first (Harmony), or train CellTypist
  on it. Each cell gets a label, a score and a "not in reference" flag, and a cluster with 30% or more
  of its cells flagged is marked as possibly absent from the reference (Kang split by donor, B cells
  left out of the reference: the B cluster flagged, 36% vs at most 4% elsewhere). Reference and
  dataset of different species are refused. When the two datasets come from different studies, the
  flag's cut is scaled to the shift between them and the card says so. In the History and the replay
  script.
- **Scanorama and BBKNN** as batch-correction methods, beside Harmony, written natively (no new
  dependencies). **Compare methods** scores none, Harmony, Scanorama and BBKNN on the current PCA for
  batch mixing (iLISI) and cell-type separation, and recommends one, with the benchmark of Luecken et
  al. 2022 as background. Clonotype ids are prefixed by dataset in a joint embedding.
- **Export for CELLxGENE and Vitessce** from the download menu. CELLxGENE: an h5ad with a readiness
  check against Discover schema 7.1.0. Vitessce: a zip with an AnnData-Zarr store and a view config
  (one scatter per 2D layout, cell sets, gene list, heatmap), checked to open in vitessce.io.

## [0.35.0] - 2026-10-24

### Added
- **Ambient RNA.** A new Ambient RNA card on Data Assessment shows each sample's soup (the RNA in the
  empty droplets of an attached raw matrix) and its contamination fraction rho, estimated as SoupX's
  autoEstCont does. It matches R SoupX 1.6.2 on 10x pbmc, heart and neuron 1k, also with 5% and 20%
  soup added. Top markers whose signal outside their cluster is about what rho x soup predicts are
  flagged in the marker tables and counted in cluster QC. Without a raw matrix the card shows the
  cells' most abundant genes and says to attach one; it flags nothing.
- **Ambient RNA correction**, an opt-in step before normalisation: SoupX's subtraction for each cell
  with each sample's estimated or a set rho. The original counts are kept, and reset to counts brings
  them back. It is recorded in the History, replayed by the script from the raw matrices and estimated
  before the run. Not available in large-dataset mode yet.
- **A second doublet detector.** The doublet step's Detector setting: Scrublet (the default) | an
  scDblFinder-style classifier written from Germain et al. 2021 | both, per capture channel. On 10x's
  human + mouse mix it finds cross-species doublets as well as R scDblFinder (AUC 0.948 vs 0.951) in
  7 s instead of 232 s. It works in large-dataset mode, and the replay script reproduces it.
- The doublet card compares the two detectors, overall and per channel. Doublet sources gain
  "classifier", "consensus" (both detectors call it) and a wider "either"; Auto uses the classifier
  once it has run (hashing still comes first), because Scrublet's automatic threshold can call almost
  nothing (Kang's STIM lane: 9 of 7,263 cells).
- **Doublet-like clusters.** Cluster QC flags a cluster whose top markers are mostly two other
  clusters', with about two cells' RNA and fewer cells than either, and names the pair.
- **Regress out** in the Scaling step: total counts, % mito, S and G2M scores, S - G2M or any numeric
  column, fitted off the variable genes before their z-scores as `sc.pp.regress_out` does. Off by
  default; X keeps the log values. It works in large-dataset mode and is in the History, the replay
  script and the pre-run estimate. Cluster QC and the technical-driver check point to it when the
  cell cycle drives clusters.

### Fixed
- Large-dataset mode failed on files written by anndata older than 0.7 (Kang 2018 as published);
  they are now rewritten for the run.
- A log-normalised X with counts in .raw (Kang as published) was reported as not log-transformed, so
  Log Transform was offered and would have logged X twice.
- DE no longer copies the tested genes: about 1.4 GiB less at 100k cells.
- A PCA that cannot make a component says which setting to change.
- Markers of an earlier clustering are removed when a run clusters again on a new graph without the
  marker step.

## [0.34.1] - 2026-10-24

### Changed
- **3D UMAP is opt-in for large datasets.** The UMAP step's new `umap_3d` setting (Automatic | On
  | Off): Automatic makes the 3D UMAP in memory, as before, and leaves it out in large-dataset
  mode, where it is about half the step's time (about 25 min at 1.2 million cells). The pre-run
  note says when a run leaves it out, how long adding it would take, and adds it in one click;
  it also shows the cost when the 3D UMAP alone would take over 10 minutes. Recorded in the
  History; the replay script leaves out `n_components=3` for a run without it. Turning it off
  removes an older run's 3D UMAP, which came from another graph.

## [0.34.0] - 2026-10-24

### Added
- **Large-dataset mode.** A run that would not fit the memory budget now works with the matrices on
  disk, read in row blocks. It is chosen automatically (`chunked` auto/on/off), and its results equal
  the in-memory run (PCA to float precision). The 100k-cell mouse brain runs in 2.2 GB of a 6 GB
  backend; the full 1.3M-cell brain (imported from its 10x .h5) ran end to end in 1 h 45 min with
  fast Leiden, peaking at about 17 GB in clustering and UMAP, so give it 22 GB. Preflight shows the mode's memory, time and the disk it writes, and the run stops before
  it starts when the disk is short.
- **Large datasets open with their matrices on disk.** A processed large dataset keeps its counts and
  a gene-major copy beside it and is never read into memory: overlays, violins and dot plots answer in
  milliseconds, and the header says "Large dataset: matrices stay on disk". Labels, category order,
  renames, colours and other edits are saved without rewriting the matrix. Gene-set scores, markers
  for a new column, co-expression, per-donor values, pseudobulk QC, the sex check, PAGA and pseudotime
  work on them; DE, AUCell, gene programs, TF activity, ligand-receptor, sub-clustering, the HVG and
  merge checks, trend genes, re-filtering and joint embeddings say why they are not available yet.
  The download is one self-contained h5ad with its counts. A dataset too big for the memory left
  opens from disk instead of being refused.
- **Large imports are streamed.** 10x .h5 (both Cell Ranger layouts), MEX and .h5ad files are read
  and saved a block of cells at a time when reading them whole would not fit, giving the same dataset
  (the 10x 1.3M-cell brain: 6 minutes, 0.7 GB, in a 4 GB scView). Merging several large samples
  streams as well.
- **Doublets per capture channel.** Scrublet runs in each 10x lane / GEM well on its own, with its own
  expected rate and threshold. Channels come from the doublet step's "Capture channels" choice, else a
  lane / GEM well / library column, else the barcodes' prefix or `-N` suffix (saved as
  `capture_channel`), else donor / batch (a guess, said so), else all cells. The 100k mouse brain: 0.7%
  flagged instead of 8.6%. The doublet card lists the channels and warns when a channel's automatic
  threshold calls almost no doublets (under 0.2% of a channel of 2,000+ cells), with that channel's
  histogram for setting a threshold by hand. Channels under 100 cells are not scored, and the views
  say how many cells that leaves uncalled.
- **Fast Leiden.** When clustering would take over 10 minutes, the pre-run note offers Leiden stopped
  after two passes (opt-in; same algorithm and seed; ARI 0.93-0.97 against the full optimisation).
  It is recorded in the History and replay script, and the sweep, variable-gene and merge checks and
  sub-clustering follow it.

### Changed
- **A failed pipeline step skips only the steps that need it.** The others still run, results it would
  have replaced are removed, and the run is saved as "finished with errors" (run summary, provenance
  and replay).
- **Scaling no longer makes every gene dense.** PCA reads the z-scores (or Pearson residuals) of the
  variable genes only, with the same PCs. At 100k cells the scaling step needs 0.06 GB instead of
  14.5 GB. X keeps the log-normalised values (labelled as such) and saved files are about a third
  smaller.
- **Marker genes and cell-level DE (Wilcoxon, t-test) rank by blocks of genes**, identical to scanpy:
  100k cells' markers in 15 s instead of 2.5 min, +0.3 GB instead of +3.1 GB. DE no longer copies the
  cells to count detections (+3 GiB at 100k). On-demand markers use the dataset's log base for fold
  changes, as the pipeline's markers do.
- In private mode uploads have no size limit, and the proxy no longer cuts off a long import after
  5 minutes; `MAX_UPLOAD_SIZE_MB` applies in public mode, to the import wizard too.
- Preflight estimates for doublets, PCA and marker genes match measured runs within 10% (doublets were
  2x over on the mouse brain, markers up to 1.9x under). UMAP's memory and time are recalibrated on
  1.16M cells (they were 2.3x and 3x under there).

### Fixed
- Processing a dataset larger than 2 GB works: scView reads it into memory first, within
  `SCVIEW_MEMORY_GB`, or says it cannot fit. The threshold is `SCVIEW_BACKED_MIN_MB` in `.env`.
- 10x .h5 files in the Cell Ranger 2 layout (`/mm10`, `/GRCh38`) are recognised.
- Seurat v3 variable genes were wrong on dense integer counts above 46,340 (squared in int32).
- `predicted_doublet` is always true/false: a cell Scrublet dropped left empty calls, which stopped the
  dataset saving.
- The backend image no longer writes `.pyc` files into bind-mounted sources.

## [0.33.0] - 2026-10-22

### Added
- **Folder upload keeps each file's path.** A Cell Ranger `outs/` folder imports as one sample with its
  raw matrix and metrics (for sequencing QC) and both its TCR and BCR contig tables, which used to
  overwrite each other because they share file names. Multi-sample folders and `cellranger multi`
  `per_sample_outs` give one sample per folder. Cell Ranger's own analysis, BAMs, Loupe files and the
  HTML report are skipped with one line saying what and why; BAMs are never uploaded. A path that
  leaves the folder is refused, and two different files on one path are refused instead of one
  silently replacing the other.
- **Doublets: cell hashing first.** When cell hashing has been run, its calls are the primary doublet
  call everywhere (cluster QC, QC report, co-pilot, "Drop predicted doublets"), with Scrublet as the
  second opinion: the Doublet score card shows a hashing x Scrublet table, overall and per sample.
  Filtering can drop doublets by auto, cell hashing, Scrublet or either, and separately drop hashing
  negatives. Old recipes replay exactly as before.
- **Per-type % mito: "Never stricter than the global rule"** (off by default). A type's bound becomes
  the higher of its own and the global rule's, so high-baseline types keep more cells and low-baseline
  types lose no more than under the global rule (ovary oocytes: 5 removed instead of 23). The per-type
  table shows which bound each type used and the loss with the option off and on; the option is also
  in the re-filter, the Mito by cell type card, provenance and the replay script.

### Changed
- **The plain `.h5ad` upload splits feature types** like the import wizard: antibodies go to the
  Protein tab and CellPlex/hashtag tags to cell hashing instead of staying in the gene matrix. The
  uploaded file is kept as sent; the split copy is streamed (37k cells: +0.13 GB).
- **TCR/BCR contig tables are side files** of their sample, like the other Cell Ranger files, and say
  whether they hold TCR or BCR chains.
- **Filtering recomputes clonotypes,** clone sizes and expansion on the cells it keeps, so the
  Recompute prompt only shows for data filtered before this release.
- **Doublet detection scores a sparse copy of the counts alone** instead of the whole dataset, with
  identical results (bitwise on Kang and paul15). Peak memory of the step: Kang 3.0 to 2.7 GB, a 20k-cell
  ovary subset 11.9 to 8.1 GB, the 37k-cell ovary 20.9 to 13.9 GB. Its memory estimate is recalibrated
  (within 20% on every measured dataset) and the replay script runs the same counts-only scoring.
- Opening a dataset with no marker genes no longer logs a failed request; the marker views say how to
  get them.
- **Filtering cuts each matrix once** instead of twice (and the gene filter copies nothing when every
  gene stays), with identical results: a 20k-cell processed dataset re-filtered at 6 GB went from
  +4.3 GB (killed) to +1.3 GB, the 37k-cell ovary's filtering from +11.2 to +2.5 GB, and its whole
  reprocess now completes at 16 GB. The memory estimate now expects re-filtering processed data to
  remove cells (it assumed every cell passed, so the pre-step check let an out-of-memory run start).

### Fixed
- After a run, `/assessment/run`, `/rerun`, PAGA and pseudotime no longer hold the old copy of the
  dataset while it is reloaded from disk (two copies at once: a large dataset at 6 GB was killed
  reloading after a run the memory check had stopped).
- The co-pilot's "Should I run batch integration?" hint no longer treats columns scView writes itself
  (V(D)J chain status, hashing calls, QC states) as batches.
- A truncated contig table gives a sentence at import instead of a server error. Deleting a dataset
  also removes its imported copy. The assessor and co-pilot counted text "False" doublet calls as
  doublets after a file round trip. The replay script scored the processed matrix when the doublet
  step had used the counts layer.

## [0.32.0] - 2026-10-22

### Added
- **Sequencing QC from Cell Ranger.** Upload `metrics_summary.csv` and the raw (unfiltered) matrix with
  the filtered one, or attach them later on Data Assessment. Each sample's metrics (reads per cell,
  saturation, valid barcodes, mapping, reads in cells) are flagged against Cell Ranger 10's own alert
  lines, with the source. A barcode-rank plot marks the knee and inflection (computed as DropletUtils
  `barcodeRanks`, matching it to 10 decimals) and the dataset's cells. On pbmc_1k_v3 every value equals
  the web summary and the 1,222 cells' median UMIs equal Cell Ranger's. The QC report and the
  co-pilot cite the metrics.
- **Cell hashing (Data Assessment).** Demultiplex hashtag (TotalSeq) or CellPlex tags with HTODemux or
  hashedDrops, ported from Seurat 5.1.0 and DropletUtils 1.24: on GSE108313 (20,238 PBMCs, 8 tags)
  both give the same call as R for every cell. Calls are saved as `hash_call` / `hash_class`
  columns that the doublet views use, singlets can be named into a sample column, and the call is
  replayed exactly. CellPlex "Multiplexing Capture" features are now kept at load (they were
  dropped); hashtag antibodies are suggested and moved out of the protein modality when confirmed.
- **TCR/BCR repertoire (Unified View → Clonotypes).** Import Cell Ranger contig annotations or AIRR
  TSVs beside the matrix, or attach them later. Cells get chain status, clonotype (scirpy's CDR3
  amino-acid definition; matches scirpy 0.22 one-to-one on a 10k-cell 10x PBMC run), clone size and
  expansion. The tab shows the largest clonotypes (a click selects their cells), diversity per group
  (optionally rarefied), Morisita-Horn or Jaccard overlap, and V/J gene usage. The co-pilot knows the
  expanded clonotypes and the cell types they sit in.
- **SingleCellExperiment and SummarizedExperiment `.rds`** convert through the R converter, not only
  Seurat: assays, cell and gene tables, reduced dimensions, ADT and HTO altExps (to the protein and
  hashtag modalities) and simple metadata. Data Assessment lists what was kept and what was left out,
  with the reason. Seurat conversions are byte-identical to before. The converter image gains a
  Bioconductor 3.19 layer (+63 MB).

### Changed
- R files convert in the background: the upload returns at once and the Data panel names the object
  and its size while it converts, so a large object no longer fails at the web proxy's 5-minute
  limit; conversion peaks about a third lower in memory. A file scView cannot read is refused with
  its class named instead of a server error. (API: `POST /datasets/upload` of an `.rds` now returns
  `converting`; poll `GET /datasets/{id}`.)

### Fixed
- Plot cards with a set height (QC plots, PCA, mean-variance, barcode ranks) rendered at about 450 px
  regardless; they now keep their height.
- The Clonotypes tab no longer asks for per-group views on a dataset with no grouping column yet; it
  says what to do instead.

## [0.31.0] - 2026-10-22

### Added
- **AUCell gene-set scores.** Every place a gene set is scored (Gene Sets, Enrichment, DE enrichment,
  Gene lists) can use AUCell (Aibar et al. 2017) as well as scanpy's score_genes: a per-cell score
  from the ranks of the set's genes within each cell, so it does not move with sequencing depth. It
  matches decoupler's `aucell` exactly (9 gene sets on Kang and paul15). One remembered choice;
  AUCell overlays are labelled "(AUCell)" and kept in sessions and links. `POST
  /datasets/{id}/genesets/score` takes `method`.
- **Check depth (Gene Sets).** One sentence on which score follows sequencing depth on your data
  (Spearman with total counts and genes detected, within clusters): on Kang a B-cell marker set
  follows genes detected under score_genes (rho -0.43) and not under AUCell (0.06).
  `POST /datasets/{id}/genesets/score/compare`.
- **Gene programs (Unified View → Programs).** Consensus NMF finds programs of genes that vary
  together across clusters (an interferon response, heat shock, a lineage) with their stability over
  random starts, top genes, the share of cells each leads, a usage overlay, a program × group
  heatmap and Hallmark enrichment. On Kang k=10 finds an interferon program (CXCL10, ISG15, RSAD2;
  Hallmark IFN-γ adjusted p 4e-9) used far more in STIM than CTRL in every cell type. An optional
  sweep over k compares stability and error. Saved with the dataset, replayed by the script, known to
  the co-pilot.
- **Plan a study (Compare).** How many donors per group a follow-up needs, simulated from this
  dataset's own donor-to-donor variability, for a shift in a cell type's share (the app's own
  composition test) or pseudobulk DE in one cell type (DESeq2-style). Power curves by donors and
  cells per donor, a sentence answer, a pessimistic curve at the upper end of the pilot's
  uncertainty, and every assumption listed. On Kang a 1.5-fold change in CD14 monocytes needs 3 to
  4 donors per group (paired, 500 to 5,000 cells per donor). A dataset without replicate donors says so plainly. The co-pilot
  knows the last plan.
- **Layouts (Unified View toolbar).** Lays the same cells out again under three UMAP seeds and three
  neighbour counts, shows them side by side in the plot's colours, and says which groups touch in
  every, some or no layout; contacts in your plot that most layouts do not repeat are listed first,
  and hovering a pair outlines it everywhere. Keep any layout as an embedding (recorded and replayed
  exactly). The co-pilot knows which contacts are rare.

### Fixed
- Saving an edit (labels, kept resolution, merges...) to an uploaded h5ad whose cell index is named
  like one of its columns no longer fails.
- The raw-counts check no longer fails on files over 2 GB (opened from disk); TF activity,
  normalisation and gene programs use it.
- The study planner opens on the author's cell types, not the Leiden clustering, when the cell types
  sit in a column called "cluster".
- History shows long parameter lists (2,000 genes) as their first items and a count.

## [0.30.0] - 2026-10-22

### Added
- **Memory and time before a load or a run.** Before a file is loaded (Data → the ingest step) and
  next to Run in Data Assessment, scView says how much memory it will need at its peak and roughly
  how long it will take, against the memory it has free: "About 3.1 GB at peak (doublet detection);
  scView has 5.2 GB free of its 8 GB limit." Estimates come from the file header (h5ad, 10x .h5,
  MEX, loom, zarr, CSV; RDS is a guess and says so) and were within 20% of the measured peak on six
  datasets from 1,222 to 37,104 cells. When it will not fit, it says what to change: raise
  `SCVIEW_MEMORY_GB`, close other datasets, skip doublet detection, or subsample. It warns; Run
  stays enabled. `GET /ingest/session/{sid}/preflight`, `GET /datasets/{id}/preflight?steps=`.
- **A run stops before a step that cannot fit,** with the reason, instead of the backend being
  killed: "Stopped before doublet detection: it needs about 9.3 GB more at its peak, and scView has
  2.4 GB free of its 6.0 GB limit". Nothing is saved, so the dataset stays as it was. Only under a
  container limit, and only when even the low end of the estimate does not fit.
- **Cancel a run** (Data Assessment, and the mito re-filter): it stops before the next step (a
  clustering stops at once) and saves nothing. `POST /datasets/{id}/assessment/cancel`.
- **Run watch.** While a run works, scView warns when memory passes 80% and 90% of its limit and
  when one step has run for a minute or more. Each step's time and sampled peak memory are recorded
  with the run, and a finished run shows its peak memory.
- **Per-cell-type % mito bounds in filtering.** The filtering step can judge % mito per cell type:
  each type with at least 30 cells gets its own bound (its median + 3 MADs, at most 50%) instead of
  one ceiling; small, flat or damaged types keep the one rule. The cell-loss preview shows each
  type's loss under both rules. On the ovary atlas (37,104 cells) one 20% ceiling removed 19% of
  the granulosa cells, a fixed 10% removed 31%, per type 5%. A bound works both ways: a type with a
  low baseline is cut lower than the one rule would (said in the table).
- **Re-filter with per-cluster bounds** (Mito by cell type card), for a dataset that was filtered
  before it had cell types: starts again from the original upload, kept cells take their cluster,
  removed cells the nearest cluster by expression, then filters per type and re-runs the recorded
  steps. A preview shows per type what comes back first (pbmc_1k at a fixed 10%: 553 cells kept,
  1,062 after, 509 brought back); recorded in provenance and the replay script.
- **Plot reliability (Unified View toolbar).** Colours each cell by the share of its graph
  neighbours (the kNN graph the clustering used) that lie among its nearest 5% of cells on the 2D
  or 3D plot, and lists groups worst first; click a group to select it. Strays in another island
  and bridges between islands come out dark; on Kang's UMAP the median is 0.93 and 9% of cells
  are below 0.5. The strict measure is given as context: a cell keeps a median of 1 of its 15
  exact neighbours, so positions inside an island carry little. The co-pilot knows the result
  (`[result:embedding_reliability]`). `GET /datasets/{id}/embeddings/{name}/reliability`.

### Changed
- **Opening a dataset makes room in memory first.** scView kept up to three datasets open whatever
  their size; it now closes the least recently used ones (and frees them for real) until the new
  one fits with a working reserve, and refuses with the reason (HTTP 507) only when it cannot fit
  even alone. A run does the same for its estimated peak before it starts.
- **Cell-level DE no longer copies the whole dataset** (+2.15 GB → +0.44 GB per request on Kang,
  and faster).
- An unknown pipeline step name (e.g. "umap") is refused with the valid names and the one probably
  meant ("embeddings"), instead of a run that did nothing.
- Co-expression over more than 20,000 cells uses a fixed random sample of 20,000 and says so;
  Excel exports are limited to 2 million cells (CSV has no limit).
- The sidebar memory gauge, the estimates and the run watch all leave out the file cache the
  system can drop, so they agree.
- The Unified View tab bar wraps onto a second row instead of scrolling (at 1600 px "Markers"
  showed as "rs").

### Fixed
- **Opening a large dataset could get the backend killed.** The data assessment, the co-pilot's
  first note and the QC report each densified a 2,000-cell sample with three copies (1 GB on a
  20k-gene dataset) at the same moment; it is now read without densifying (+0.13 GB). A request
  that was reading a dataset when a run replaced it re-read the whole file outside the cache; it
  now keeps the copy it had.
- **Doublet detection on a processed dataset** scores the counts (it got the scaled matrix, filled
  the memory and failed with "Bin edges must be unique").
- **Ingesting a file whose cell index is named like a column with other values** (the ovary atlas:
  `cell_id`) failed with an error 500; it now loads, and a failed save is reported as such.
- Gene-set scoring and the violin endpoint no longer block other requests while they work.
- Cell-type columns named with a space or a dot ("Cell type", R's `cell.type`) are recognised as
  cell types.
- The run result card stayed empty after a run (the assessment refresh cleared it); a streamed
  event split across two network chunks lost its data.
- A run stopped mid-save no longer leaves a broken dataset file (it is written beside it and then
  swapped in).

## [0.29.0] - 2026-10-21

### Added
- **Cluster and gene reports (Unified View → Report).** One page per cluster: its size per donor and
  condition, what other label columns call it, marker genes with the marker-database verdict,
  HALLMARK enrichment, QC against the other cells (with the cells filtering removed), composition
  per donor and the recorded methods. One page per gene: expression per group, marker status, the
  database cell types listing it, per-donor values with paired tests, co-expressed genes, HVG status
  and MSigDB sets. Every section states its n and how it was computed, and says where to compute
  anything missing; nothing is downloaded on its behalf. "Download HTML" gives a self-contained file
  (no scripts, nothing fetched) that prints cleanly. Opened from a Markers row, the cluster map or
  the co-pilot. `GET /datasets/{id}/report/cluster`, `/report/gene` and their `.html` versions.
- **Copy link (header).** A link that reopens the current view: dataset, tab, embedding, colouring,
  gene overlay, highlighted group, zoom, DE settings, colours and selection. The view lives in the
  URL fragment, so it never reaches a server log. Opened in another scView, the link finds the
  dataset by its cells and says plainly when it is not loaded. Gene lists and gates are not shared.
  `GET /datasets` now includes `cells_fingerprint` for datasets already opened.
- **Compare DE results (Unified View → DE).** The significant genes of 2 to 8 DE comparisons side by
  side as an UpSet plot; click a bar for its genes with each comparison's log2FC, copy them or save
  them as a gene list. The sign is kept: a gene up in one comparison and down in another is listed
  as discordant, not shared. Logistic regression counts its top 50 genes per direction; "only genes
  tested in all" ignores genes the detection prefilter left out. Your gene lists can join as extra
  sets. On Kang, STIM vs CTRL within CD14 Mono, CD4 Naive T and B share 190 genes (ISG15, IFIT3,
  CXCL10...). `GET /datasets/{id}/de/results`, `POST /datasets/{id}/de/overlap`.
- **Gates (Unified View toolbar).** Save a lasso, legend pick or filter as a named gate. Gates nest
  (a gate drawn inside another keeps only its parent's cells), show n, % of parent and % of total,
  and combine with AND / OR / NOT. Click a gate to make it the selection for DE, labelling and the
  histogram. Gates survive reloads and go into session files; when a dataset's cells change they are
  marked stale and not applied. Write the leaf gates into one label column (undo in one step), or
  export the counts as CSV, per donor or sample.

### Fixed
- **Lasso on 3D embeddings.** The lasso did nothing on a 3D UMAP and a drag rotated the plot. It now
  selects every cell under the outline, at any depth.
- **Opening a session or link into an open DE tab** no longer sets the DE "Within" scope and the
  selection from the restored highlight; the session's own scope and selection are kept.
- A flaky Annotation test (waited for the marker-sets control).

## [0.28.0] - 2026-10-21

### Added
- **Who gets removed (Data Assessment → QC thresholds).** Under the filter preview, each threshold's
  loss per cell type × condition (or sample), with a chi-square test per cell type of whether the
  loss depends on the condition. A type is flagged when a filter removes more than 30% of it, or when
  its removal rate differs between conditions by 10+ points with p < 0.01. The filtering step now
  records the cells it removed (their QC values and labels, in the h5ad), so after filtering the card
  shows what was actually removed. On pbmc_1k_v3 a 10% mito ceiling removes 89% of NK cells.
- **Mito by cell type (Data Assessment).** Each cell type's % mito (median, MAD, percentiles), its own
  bound (median + 3 MADs) and what one global ceiling removes from it compared with that bound. Types
  the ceiling cuts hard are flagged, and a type with a median of 50% or more is called damaged.
  After filtering, the cells the ceiling removed are added back from the record. Advisory only:
  filtering is unchanged. `GET /qc/mito-by-type`.
- **QC states (Data Assessment).** Per cell type, the ribosomal-protein share of non-mito counts
  against % mito, with one or two Gaussian states per group (BIC, plus a separation rule so a
  skewed distribution stays one state) and the second state's share per condition. "Save as label"
  writes it as an obs column (`qc_state`) instead of a mito threshold. Without MT- genes it fits on
  the ribosomal share alone and says so. On Kang, CD14 monocytes split into two states: 12% of CTRL
  cells vs 65% of STIM cells are in the low-RP one. `GET /qc/states`, `POST /qc/states/label`.
- **Composition check before a DE between groups (Unified View → DE → Between groups).** Before
  the run, each side's share of every cell type (or cluster) is shown. A group is flagged when its
  shares differ by more than 10 points and Fisher's exact test gives p < 0.05. Each flagged group
  has a "Compare within" chip and a column picker. The result carries the caution beside the batch
  caution, and the co-pilot reads it. `GET /de/composition`; `/de/compare` returns `composition`.
- **MALAT1 share per cell.** A "% MALAT1" QC histogram and a "Low MALAT1" cluster flag: a median
  below a third of the median cell's share suggests cytoplasmic fragments or lost nuclei. The flag is
  off when MALAT1 is barely captured. An author intronic or unspliced share column is shown per cluster.
- **Co-pilot facts** for the removal audit, mito by cell type and QC states, each checked by the
  claims ledger.

### Changed
- Colour scale "Auto": signed cell columns such as S_score and G2M_score now use the diverging map
  centred on 0.

## [0.27.0] - 2026-10-21

### Added
- **PCA diagnostics (Data Assessment → PCA).** Each component's share of the variance with the
  cumulative curve, an elbow (Kneedle: the PC farthest below the chord of the scree curve), the
  number of PCs reaching 90% of the retained variance, the number of PCs the neighbour graph used,
  and the top loading genes of any PC; click a gene to colour the scatter by it. The scatter can
  show any pair of principal components (x/y chooser next to the embedding selector;
  `GET /embeddings/{name}?dims=i,j`). The co-pilot cites the result as `[result:pca]`.
- **Paired designs in per-donor plots.** When the same donors appear in several conditions, the
  violin's "Per donor" mode and the composition test's per-sample plot draw a slope chart: a line
  per donor, the mean per condition, and with two conditions a Wilcoxon signed-rank and a paired
  t-test per group (descriptive under 5 pairs). More conditions draw each donor's trajectory. The
  composition caption says whether the moderated test blocked on the donor. Boxes stay a toggle
  away. `GET /expression/per-donor` returns `design` and `paired_tests`; `POST /composition/test`
  returns `pairing` and `groups[].paired_test`.
- **Diverging colour maps for signed overlays.** Gene-set, TF/pathway activity, cell-cycle and
  stress scores and protein CLR values draw on a map centred on 0 (blue-white-red by default;
  purple-orange and brown-teal available), the legend marks 0, and the per-cell heatmap's
  z-scores follow the same map. "Auto / Sequential / Diverging" in the toolbar overrides the rule.
- **Colour-range clipping.** None, 1-99%, 2-98%, 5-95% or a manual min and max, as a toolbar
  default or per view from the legend. The legend states the range, the clipping and the share of
  cells beyond it. Saved with sessions.
- **Honest tables and captions.** Every table column header has a tooltip saying what the number
  is and how it was computed (markers, DE, enrichment, GSEA, activity, communication, Milo,
  sample / cluster / pseudobulk QC, joint summary, co-expression, doublets). Capped tables say so
  ("Top 200 of 2,311 by adjusted p; the CSV has all"), the Markers tab says it shows each group's
  top 100 by Wilcoxon score, and QC histograms, violins, the specificity plot and the stacked
  violin state their n. The batch-mixing score explains its formula and pairs its colour with a
  word.

## [0.26.0] - 2026-10-21

### Added
- **Cell-cell communication (Unified View → Communication).** Ligand-receptor interactions between
  groups, scored with LIANA's consensus rank aggregate (CellPhoneDB, Connectome, log2FC, NATMI,
  SingleCellSignalR) on all cells or on one condition. Table with CSV download, dot plot, circle
  plot or heatmap of interactions between groups, and sender/receiver roles. The human or mouse
  LIANA resource (2 MB) downloads on first use. Method scores, CellPhoneDB p-values and the
  specificity rank match liana-py 1.10 exactly on Kang and paul15. The magnitude rank counts the
  product shared by Connectome and NATMI once; liana-py 1.10 ranks it twice, which inverts it.
  No new dependency; Kang takes about 2 s.
- **Milo neighbourhood differential abundance (Compare → Composition → Neighbourhoods).** A
  cluster-free test of abundance shifts between two conditions with donors as replicates
  (paired when every donor is in both): neighbourhoods on the dataset's own graph, a negative
  binomial test per neighbourhood (DESeq2), spatial FDR, a beeswarm by cell type, the
  neighbourhood graph on any embedding, CSV, and selecting a neighbourhood's cells on the scatter.
  Matches pertpy's Milo (pydeseq2 solver) on Kang, agrees with edgeR (logFC r = 0.989), and finds
  nothing on shuffled labels. Warns when the graph was batch-corrected for the condition itself.
- **Clustering tree across the resolution sweep.** Cluster sizes at each resolution, cells moving
  between them, SC3 stability, and clusters that draw on two parents (clustree-style).
- **"Should any clusters merge?"** Neighbouring clusters are tested on held-out counts (negative
  binomial count splitting), so the test does not reuse the data that made the clusters. Pairs
  with fewer than 10 distinguishing genes (adjustable) are merge candidates; the result saves as
  `<column>_merged` and the replay script reproduces it. One split with a fixed seed: a guide, not
  a verdict. Needs raw counts; Kang takes about 1.5 min.
- **The clustering sensitivity check also varies the number of neighbours and the algorithm**
  (Leiden or Louvain), next to the variable-gene settings.
- The co-pilot knows the last communication, Milo and merge-check results and cites them.

## [0.25.0] - 2026-10-21

### Added
- **A memory limit for scView, and a gauge for it.** The start scripts ask how much memory scView
  may use on the first start, suggesting free memory minus 3 GB to the nearest GB (at most what
  Docker or Podman has, at least 4 GB), and store it as `SCVIEW_MEMORY_GB` in `.env`; the backend
  and converter get it as a container limit with no extra swap. A dataset that needs more stops
  scView instead of the computer: on Linux nothing else caps Docker or Podman, and a big run could
  make the whole machine unresponsive. Without the setting there is no limit, as before. The
  sidebar shows the backend's memory use against its limit (`GET /system/memory`), amber from 75%
  and red from 90%.
- **scView says when it ran out of memory.** When the backend was killed and restarted, a notice at
  the top of the page says so, with its memory limit and where to raise it, and that unsaved work
  such as a run in progress was lost. While the backend is down, error boxes say it is restarting
  or may have run out of memory, instead of showing nginx's "502 Bad Gateway" page; a run cut off
  midway says the same. `GET /system/memory` reports it as `unclean_restart`.

### Fixed
- **Windows + Podman: scView starts after Windows was shut down with Podman running.** That
  leaves a socket file behind, and the next Podman start then has no connection for Compose
  (`failed to connect to the docker API at npipe:////./pipe/podman-machine-default`), although
  Podman itself works. `start-windows.bat` now removes the stale file before starting the Podman
  machine, and restarts the machine once if the connection is still missing.

## [0.24.1] - 2026-10-19

### Fixed
- **Processing uses much less memory.** Scaling now keeps the expression matrix as float32:
  scanpy turned it into a dense float64 matrix of every gene, 1.5 GB for Kang's 13,836 cells, and
  processing Kang peaked at 4.9 GB. It now peaks at about 3.3 GB and runs faster (scaling 18 s →
  5 s, PCA 27 s → 14 s). The principal components are the same (correlation 1.0); a few cells at
  cluster edges can change cluster (Kang Leiden: adjusted Rand index 0.987 against before). The
  HVG check's control and the replay script scale the same way.
- **Memory is freed after a run.** The process held 4.6 GB after processing Kang, where opening the
  result takes 1.25 GB; it now holds 2.0 GB. On a Windows PC with 8 GB (about 4 GB for Docker or
  Podman) the old peak ran out of memory, which stopped the backend and showed "network error".
- **The backend and converter restart by themselves** if they are stopped by an error such as
  running out of memory, instead of leaving scView unreachable until it is started again.
- README: how much memory processing needs, and that Docker and Podman on Windows get half of it.
- **Annotation cross-check:** a Compare clicked right after the columns or reference changed could
  lose its answer (the result, or the server's reason for refusing), as if from an older selection.

## [0.24.0] - 2026-10-19

### Added
- **Podman support.** `start-windows.bat`, `stop-windows.bat` and `start.sh` use Podman when
  Docker isn't running: on Windows and macOS they start the Podman machine (creating it the first
  time) and check that Compose is set up. On Linux, `start.sh` picks a compose tool that can read
  scView's compose files (Docker Compose 2.24+ or podman-compose 1.2+), because `podman compose`
  would otherwise take an old `docker-compose` 1.x first. `SCVIEW_ENGINE=docker|podman` chooses
  when both are installed. See *Using Podman instead of Docker* in the README.

### Fixed
- **Long processing steps no longer fail with "network error".** The app's web server closed a
  response that had sent nothing for 5 minutes, and a step reports only when it ends: doublet
  detection on Kang in a Windows virtual machine under Podman took longer than that, the page
  showed "network error", and the run stopped without saving. Every streamed response (processing,
  re-clustering, integration, QC report, co-pilot) now sends a keep-alive every 15 seconds.
- **Without an API key, cell-type annotation starts on Marker score (offline)** instead of the AI,
  and the AI tab says it needs a key and would run CellTypist instead (`GET /assistant/llm-status`).

### Changed
- **Doublet detection is about 5 times faster.** Scrublet now finds neighbours exactly instead
  of with an approximate search, for datasets up to 50,000 cells. The scores are the same (Kang:
  Pearson correlation 1.0, the same top 1-5% of cells) and the step takes 18 s instead of 93 s
  on Kang, so processing the whole dataset takes 82 s instead of 142 s. It also skips a numba
  compilation that cost tens of seconds on the first run after every start. Larger datasets keep
  the approximate search, which is faster there. The replay script and provenance record which
  was used.

## [0.23.1] - 2026-10-19

### Changed
- **Marker choice needs adjusted p < 0.05.** The markers handed to the AI annotator, the co-pilot,
  the marker database check and the enrichment (ORA) query now also need an adjusted p-value under
  0.05, besides being detected in at least 20% of the group's cells. Only small groups change
  (under ~70 cells), where the fold-change leaders were genes seen in a handful of cells: on paul15
  the database lookup named 13 of 19 cell types correctly instead of 11 and the AI annotator 12
  instead of 11; Kang's large groups are unchanged, and its megakaryocyte enrichment query drops
  from 100 genes to 40 with 5 hallmark sets significant instead of 1. Results without p-values
  (logistic regression) are not filtered by them. Cached enrichment results are recomputed.

## [0.23.0] - 2026-10-19

### Added
- **Per-cell marker heatmap.** Marker Genes → Heatmap has a *Group means | Per cell* switch. Per
  cell shows the genes over a seeded sample of each group's cells (50-500 per group, at most
  5,000 cells), with a strip of group colours, boundaries and names, and each gene's z-score over
  the sampled cells clipped to ±2.5 (as Seurat's DoHeatmap) or the layer's fixed range. The
  caption says how many cells were sampled and which groups are shown whole (Kang, 13 cell types
  at 200: 2,368 of 13,836 cells).
- **Cluster genes** on the marker heatmap orders the genes by their profiles across groups
  (correlation distance, average linkage) with a gene dendrogram, next to the existing group
  dendrogram; top markers show which group each gene is a marker of.
- **Annotation cross-check** (Data Assessment): compare 2-6 label columns (CellTypist, the AI,
  marker scores, clusterings, hand labels) with pairwise ARI/AMI, and against a reference with a
  row-normalised confusion heatmap, a Sankey and per-label agreement. Each column's labels are
  mapped onto the reference label most of their cells carry, so different names for the same cells
  agree; same-name agreement is shown separately. The consensus label and per-cell agreement can be
  saved as columns to colour by (Kang `cluster` vs Leiden 0.5: 73.8% mapped agreement).
- **Gene lists** (Unified View): named gene lists that survive reloads, dataset switches and
  session files. Add genes from Markers, DE (up or down), Co-expressed genes and the gene search,
  or import / export them as txt, csv or gmt. A list can be overlaid gene by gene, shown as a
  feature grid, dot plot or heatmap, scored, tested for enrichment, or saved as a gene-set
  collection. Genes missing from the open dataset are greyed, with a one-click fix for case-only
  differences (CD3E vs Cd3e).
- **Marker database check** in the annotation cross-check: each label of the reference column is
  matched by name (with Cell Ontology synonyms and common abbreviations) to marker-database cell
  types, and its database markers are tested against the group's own top 25 markers
  (hypergeometric, measured genes as background): *supported*, *no database support*, *too few
  database markers* or *not in the databases*, with the shared genes. Verdicts use PanglaoDB +
  CellMarker 3.0 + ACT for human data and CellMarker 3.0's mouse rows for mouse; each group also
  gets the 3 PanglaoDB / curated CellMarker types its markers fit best (Kang: 13 of 13 labels
  supported; paul15: 16 of 19). The databases are downloaded from their own sites on first use
  (~73 MB, read straight from the zips; `MARKER_DB_AUTO_DOWNLOAD`), with pinned SHA-256 sums for
  CellMarker 3.0 and ACT; if CellMarker 3.0 is unavailable or has changed, CellMarker 2.0 is used
  instead and the card says so. None of the databases states a licence. The sources were chosen by
  measurement against HPA, Tabula Sapiens, Kang and paul15 (`PanglaoDB + CellMarker 3.0 + ACT`:
  HPA lookup 0.73 against 0.67 for PanglaoDB + CellMarker 2.0; MSigDB C8, Tabula Sapiens, Cell
  Taxonomy, scType, scCATCH, Azimuth and HCL/MCA added nothing).

### Changed
- **Marker-score annotation** names each cluster after the marker-database type its markers fit
  best (any tissue, human or mouse), or uses the 14 built-in immune sets when you choose them (they
  remain better on PBMC subtypes) or when the databases are unavailable (paul15: 11 of 19 labels
  right against 2 with the built-in sets).
- **Which genes describe a group.** AI annotation, the co-pilot's marker facts and per-group
  enrichment (over-representation) now use the genes expressed in at least 20% of the group, by
  fold change, without ribosomal or mitochondrial genes, instead of the top of the Wilcoxon score
  order, which led Kang's CD14 monocytes with FTL, CD63 and ANXA5 (CD14 at rank 70). Enrichment's
  *N genes* is now a maximum; cached enrichment results are recomputed when next opened.

### Fixed
- The AI co-pilot can open the Protein, Activity and Gene lists tabs.
- A continuous colour legend under a short label no longer runs its minimum and maximum together.
- The replay script explains the annotation cross-check step instead of calling it unknown, and
  writes marker-database labels in from the record.
- The co-pilot's context no longer claims to show only the first 16 groups' markers (it lists all).
- The smoke test of the published images retries its first requests, which could reach the
  frontend a moment before it accepted connections.

## [0.22.0] - 2026-10-18

### Added
- **More DE tests.** Pick the cell-level test: Wilcoxon (default), Welch t-test or logistic
  regression. Logistic regression gives coefficients and no p-values, so the volcano plots the
  coefficient, the table lists genes by it and over-representation is off (GSEA still works).
  MAST is listed as unavailable (R only). *Compare methods* runs the same comparison under every
  test and shows how far they agree: top-50 Jaccard per side, Spearman of the statistics, and the
  genes only one test ranks top (Kang CD14 Mono vs rest: Wilcoxon vs Welch 0.96 of the top 50 up).
  GSEA and TF/pathway activity from a DE result name the test and rank by its statistic.
- **Co-expressed genes** (Unified View → Expression): the genes most correlated with the overlaid
  gene (Pearson or Spearman) over all cells, the selection or the highlighted group, positive and
  negative, with click to overlay and *To grid* to send them to Several genes. Rare and constant
  genes are skipped and counted, never returned as NaN (Kang ISG15: IFIT3 0.85, IFIT1 0.85, IFI6 0.82).
- **Per-sample sex check.** Each sample is called female / male / mixed from XIST/TSIX against
  Y-chromosome genes (share of cells and pseudobulk CPM), shown as a Sex column and caption in the
  per-sample QC table. It flags disagreement with a sex metadata column (possible sample swap),
  mixed samples (pooled donors or cross-donor doublets) and sex confounded with a condition, and
  says so when the marker genes are missing; the QC report states the calls.
- **Sample-level (pseudobulk) QC** on the Data Assessment tab: a PCA of each sample's (x
  condition's) summed counts, coloured by any sample-level covariate, optionally within one cell
  type. Each PC is tested against the covariates (R² and p, driver marked) and outlier profiles
  are listed with the rule used; profiles under 10 cells are left out and listed (Kang: PC1 75% =
  stim, PC2 = donor).
- **Overlay histograms.** The Unified View distribution panel has a *Histogram* mode: a selection
  (lasso, legend picks, cell filter) or a group against the rest or another group, as overlaid
  normalised histograms or densities of the plotted gene or score or any numeric QC column, with
  n, medians, % expressing, median difference, KS distance and Mann-Whitney AUC as descriptive
  numbers, and a log / linear axis.
- **Reproduce this analysis** (History): download a standalone Python (scanpy) script that replays
  the recorded steps from the input file, and a RUN_MANIFEST.json with app and package versions,
  the input file's SHA-256 and every step. Each step is marked exact / approximate / from record /
  not replayable. On Kang and paul15 the script reproduced scView's cells, variable genes,
  clusters, PCA / Harmony / UMAP and markers exactly. Both files are also in the Download menu.

### Fixed
- Opening a dataset without UMAP no longer requests the UMAP of the dataset opened before (a 404
  on every open).
- The QC and pseudobulk scatter's plot buttons no longer cover the legend's first entry.

## [0.21.0] - 2026-10-18

### Added
- **Several genes at once** (Unified View → Expression): pick 2 to 9 genes to see the embedding
  coloured by each, side by side on one camera (each gene on its own scale or one shared scale,
  expressing cells drawn on top, PNG export), or as a stacked violin across the group-by. The
  genes load in one request.
- **Two genes compared.** *Blend* shows gene A, gene B and both blended on the embedding (A red,
  B green, both yellow, as Seurat does; or magenta / green with both dark, for red-green colour
  blindness) with a 2D colour key and a choice of where each gene's
  scale tops out (99th or 95th percentile of its expressing cells, or max). *Gene vs gene*
  plots each cell's A against its B by group, with Pearson and Spearman r (all cells and within
  the highlighted group) and the cells expressing both, one or neither; select the co-expressing
  cells or any quadrant (Kang CD3D vs GNLY: 350 both, r -0.03).
- **Per-donor plots.** The Unified View violin has a *Per donor* mode: one pseudobulk value per
  donor in each group, split by condition, as the mean log-normalised expression (default) or
  log2 CPM of summed counts; donors with under 10 cells in a group are left out. The composition
  test's per-sample plot is a box + strip with one point per donor, on one axis or a panel per
  cell type. Both captions give the donors per condition and warn under 3; without a sample
  column they say one is needed instead of plotting each condition as a sample.
- **Composition bars** (Observations): sort by stored order, name, size or the share of any
  stacked category; counts or %; split by a third column, with a one-click *Per donor by
  condition* view; click a bar for a 10 x 10 waffle; the CSV exports the view as shown.
- **Group dendrogram and correlation.** The cluster-distance heatmap draws its complete-linkage
  tree and toggles to the Pearson correlation of the group centroids (in the graph's PC space,
  where mean expression would put nearly every pair above 0.5). Marker dot plots and heatmaps
  can order their groups by the same tree and draw it alongside.
- **Comparable dot plots.** Colour by the mean over all cells or the mean in expressing cells,
  scaled per gene or on a fixed range for the layer (0 to its 99th percentile of non-zero
  values, editable; Kang 0-5) that stays put between plots. Dot area is the share of expressing
  cells on a fixed scale with a size legend, and groups under 5% expressing are drawn hollow.
- **Legend order and multi-select.** Drag or move categories, or sort them A-Z (numbers by
  value) or by size; the order is saved to the dataset, so legends, violins, dot plots and
  composition bars follow it, while each label keeps its colour (through renames and merges too).
  Ctrl/Cmd/Shift-click picks
  several groups; *Select these cells* makes them the selection.
- **CellTypist checks its fit.** The model list puts the dataset's species first, with a warning
  for a model of another species; mouse data starts with no model picked, since every mouse
  model is tissue-specific (there is no general mouse blood model). Each run records how many
  of the model's genes the dataset has (warns under 50%) and shows each label's cells, median
  confidence, % under 0.5 and agreement with the per-cell call, flags low labels, and cautions
  when the median is under 0.5. *Colour embedding by confidence* is one click (Kang with
  Immune_All_Low: 65% of model genes, median 0.69, DC2 and Late erythroid flagged).
- API: `/expression/by-group` returns `mean_expressing` and `scale_range`; `/clustering/distances`
  returns `tree` and `correlation`; `/metadata/crosstab` takes `facet`; new
  `GET /expression/per-donor`, `GET /annotation/celltypist` and
  `POST /metadata/{column}/order`. The CellTypist catalog lists each model's `species`.
  Dataset `obs_columns` carry `color_slots` (palette slot per category) once an order is saved.

## [0.20.0] - 2026-10-18

### Changed
- **QC filtering judges each sample's own outliers by default.** New filtering runs remove cells
  whose counts or genes per cell fall more than 5 MADs below their sample's median (on log1p),
  and cells whose % mito is more than 3 MADs above it, with the max % mito kept as a ceiling.
  Samples are detected as for the other checks (Kang: Donor); a sample under 100 cells is judged
  with the pooled cells. The upper bounds (median + 5 MADs) are opt-in: on Kang they took 12.3%
  of DCs, 10% of Eryth and 9.3% of pDCs, and doublet detection already covers the high tail.
  Fixed cut-offs stay one switch away, and a dataset filtered with them before keeps them on a
  re-run. A re-run with the same MAD settings reuses the recorded thresholds instead of trimming
  the tails again (Kang with upper bounds: 170 cells the first time, 0 the second). The Data
  Assessment tab shows each sample's thresholds and loss, and History and the methods text show
  them in one line.
- **DE tests only genes detected in at least 1% of either group's cells** (Seurat v5's default;
  5, 10 and 25% in the Genes menu). On Kang CD14 Mono vs CD4 Naive T this leaves out 6,016 of
  14,053 genes, and 1,586 of the 1,708 |log2FC| > 10 artefacts with them, while more genes pass
  BH (2,305 vs 2,161). The result says how many genes were tested and skipped and shows a
  detection histogram with the cut-off; GSEA and TF activity on a stored result say they rank
  only the tested genes. Pseudobulk keeps its own count filter.
- **Cells are drawn in a seeded random order**, the same for every colouring, with a selection
  drawn last, so a cluster stored late no longer paints over the others.

### Added
- **Per-sample QC table** on the Data Assessment tab: cells, median genes, UMIs and % mito per
  sample, the cells the current thresholds remove, and flags for a sample whose median sits over
  3 MADs from the others' or with few cells.
- **Cluster QC checks haemoglobin, the cell cycle and technical drivers.** A cluster is called
  erythroid by its red-cell markers other than haemoglobin (ALAS2, KLF1, CA1 ...) or a red-cell
  Hb level, and gets a note, never a flag; haemoglobin without them is flagged as likely ambient
  RNA from lysed red cells. Clusters of clearly cycling cells (S or G2M score above 0.1) are
  flagged. Total counts and % mito, ribosomal and haemoglobin reads are correlated with the
  clustering's first 10 PCs, over all cells and inside clusters: on paul15 log total counts tracks
  PC2 at r = 0.95 inside clusters, which the panel raises as a concern. These reach the co-pilot,
  the QC report and the insight banner.
- **Transcriptome-size caution on DE.** When one side's cells hold 1.5x or more the RNA of the
  other's (median raw total counts), the DE subtab says so with a violin of total counts: equal
  transcript numbers then read as a fold change. CD14 Mono vs CD4 Naive T on Kang is 1.79x; STIM
  vs CTRL within CD14 Mono is 1.12x and stays quiet.
- **UMAP and t-SNE plots note that gaps between clusters aren't distances**, with a link to a
  heatmap of distances between group centroids in the PCs the neighbour graph used (Harmony PCs
  when it did), ordered by complete linkage, under the Unified View's plot
  (`GET /clustering/distances`).
- **Log scale toggles** on violins (expression, gene-set scores, QC by group), with ticks and hover
  in the original units, and on the QC histograms' cell counts.

### Fixed
- **The lasso selects the cells under it.** It selected the cells mirrored about the plot's
  vertical centre, selected nothing before the first pan, and used a stale camera after Reset
  view.
- **The Data Assessment tab no longer overflows the window.** At 1600 px wide the QC plots pushed
  the page past the right edge and cut off the header's Session and Download buttons.

## [0.19.0] - 2026-10-17

### Added
- **Claims ledger under every co-pilot answer.** Each sentence that states a number, names a
  gene or cites a fact is checked against the text of the fact it cites, for free and without a
  model call: supported (its numbers and genes are in that fact, allowing rounding, "1.5k" and
  totals or shares recomputed from one line), refuted (the fact has a different number for the
  same group and statistic), or uncertain. Refuted claims stay visible; the rest fold into a
  one-line count. Negations, comparisons, ratios and hedged numbers ("about", "over") are never
  more than uncertain by rule. A **Check claims** button sends only the uncertain claims, each
  with its cited snippet, to one small model call (`POST /assistant/check-claims`, at most 12).
  On 23 real answers about Kang, pbmc_1k_v3 and ovary: 126 of 194 claims supported by rule,
  none refuted, and no wrong verdict found by hand.

### Changed
- **The co-pilot's context names its citation tags**, so answers cite facts that exist. Before,
  the model invented tags such as `[obs:stim]`, `[dataset]` or `[provenance]`: 10 of 24 tags in
  five Kang answers matched a real fact; with the tags shown, 26 of 26. Loose tags still resolve.

## [0.18.0] - 2026-10-17

### Changed
- **The composition test moderates variances robustly, as propeller does.** limma's robust
  empirical Bayes (`eBayes(robust=TRUE)`) is now the default: a group whose sample-to-sample
  variance is an outlier gets a smaller prior df of its own, so it neither borrows strength it
  shouldn't nor pulls the shared prior for the rest. Matches R limma to numerical precision. With
  fewer than ten groups limma's robust fit is the plain one, so results there are unchanged. On
  Kang by cluster, Eryth and pDC are flagged; CD14 Mono moves from FDR 0.053 to 0.056.

### Fixed
- **The legend no longer hides cells or labels in the Unified View.** When the automatic fit would
  put cells under the legend (Kang's T cells sat half under it), the cells are fitted beside it
  instead, as long as that keeps at least half the zoom; a camera you moved yourself is left
  alone. Group labels keep clear of the legend too.
- **Changing the colouring keeps your zoom.** Every colour change used to re-fit the camera to the
  whole embedding; the coordinates are now recognised as unchanged and the view stays put.

### Added
- **CITE-seq: antibodies as a protein modality.** A 10x feature-barcoding upload (`.h5` or MEX) or
  an h5ad with `feature_types` keeps its antibody counts apart from the genes (they used to be
  dropped from an `.h5`, or read as genes from a MEX), normalised by centred log-ratio per cell as
  Seurat does. A new Protein tab in the Unified View colours the cells by a protein (with its
  violin), shows every protein by group, keeps isotype controls apart as background, and sets a
  protein against the gene of its target (Spearman ρ and how often the RNA drops out). Protein
  colourings are saved in sessions, and the joint embedding and multi-sample uploads keep the
  antibodies every dataset measured. A Seurat object's ADT (or Protein) assay converts with it.
  Not yet: DSB normalisation, which needs the empty droplets.
- **Design and variance choices in the composition test.** A Design menu forces paired (by
  sample) or unpaired, with Auto showing what it picked; a Variances menu switches back to plain
  eBayes. The CSV export carries each group's prior df.

## [0.17.0] - 2026-10-17

### Added
- **Seurat v3 variable genes.** Processing offers the Seurat v3 HVG flavour, which ranks genes by
  the standardised variance of the raw counts (with a batch column, by how many batches pick them).
  The variable-gene check can now include it too. Needs raw counts in the dataset; the backend
  image adds scikit-misc for it.

### Changed
- **The variable-gene check says why its control row doesn't match.** When re-running the current
  setting doesn't reproduce the stored clusters, the note now names the cause it can see: the
  reference column came with the file, or the dataset was processed by an older scView whose PCs
  the check rebuilds with today's pipeline.

## [0.16.1] - 2026-10-17

### Fixed
- **The app no longer freezes while it clusters.** Leiden clustering (processing, re-clustering,
  the resolution sweep, sub-clustering and the variable-gene check) now runs in a separate process.
  Before, every other request waited for it, for up to about 5 s at a time on a 14k-cell dataset.
  The clusters are identical to before; each clustering takes about 1 s longer.

## [0.16.0] - 2026-10-17

### Added
- **Composition testing between conditions.** Compare → Composition → Between conditions tests
  whether cell-type or cluster proportions differ, with donors or samples as the replicates
  (propeller-style: per-sample shares, logit or arcsin-square-root, a moderated t or F test, paired
  by donor when every donor is in every condition, Benjamini-Hochberg). The statistics match R
  limma. A table, CSV and per-sample box plot, with the donor column found for you. It warns when a
  condition has fewer than 2 samples, when conditions are also batches, and when clusters were
  built on an embedding corrected for the condition.
- **PAGA.** The Trajectory panel computes a PAGA graph of any grouping and draws it over the
  embedding (node size by cells, edges by connectivity, a threshold and a spanning-tree switch).
- **Genes changing along pseudotime.** A heatmap of the genes whose expression changes most along
  pseudotime (spline F-test), on all cells or on one PAGA branch from the root, ordered by where
  they peak; click a gene to plot its trend and colour the embedding. CSV export.
- **Variable-gene check.** Unified View → Variable genes re-clusters under other variable-gene
  settings (half and 1.5x the gene count, Cell Ranger, Pearson residuals) at the dataset's own
  clustering settings, without changing the dataset, and compares each with the current clusters
  (ARI, AMI, and each cluster's stability). The current setting is re-run as a control. "Keep this
  one" saves a setting's clusters as a column.
- **Annotation confidence.** Data Assessment → Annotation confidence gives every cell of a label
  column a flag (confident, ambiguous, unsure, doesn't fit) from split conformal prediction sets at
  an error rate you choose, with held-out coverage, the worst-fitting labels and what they are
  confused with. Colour the embedding by the flags, or copy the column with rejected cells set to
  "Unknown". It measures how well a label agrees with the expression of the cells carrying it, not
  whether it is right. A CellTypist column can be scored with CellTypist's own probabilities, which
  the annotation step now keeps.
- **A note in Trajectory when a dataset has spliced/unspliced layers** (scView does not compute RNA
  velocity yet).

### Changed
- **The co-pilot warns against chi-square or Fisher tests on cell counts** for composition, and the
  two-dataset composition view and the joint embedding's per-cluster bars say their differences are
  descriptive. The co-pilot sees the last composition test.
- **The Trajectory panel draws 3-D embeddings in 3-D.**

### Fixed
- **An edit interrupted by closing the tab could overlap the next one.** Eleven actions that save
  a dataset (labels, renames, markers, keeping a clustering, sub-clusters, the doublet threshold,
  annotation checks, saving a joint embedding, pseudotime) released its lock as soon as the request
  ended, while their write could still be running. The lock is now held until the write finishes.
- **The variable-gene check** works on datasets processed outside scView, comparing with the
  clustering the rest of the app shows.
- **An empty pseudotime bin in a gene trend** is a gap, not 0.

## [0.15.0] - 2026-10-17

### Added
- **QC thresholds on the histograms.** Set min/max genes, min counts and max % mito (when it can
  be measured) by dragging a line or typing, with a live summary of the cells removed overall, per
  sample and per cluster, and a warning when a threshold removes a whole sample or cluster or more
  than half of one. The values are the filtering step's own, which gains max genes and min counts.
- **Doublet threshold.** A histogram of the doublet score on Data Assessment: drag or type a
  threshold to see the cells it calls, overall and per sample, beside the rate expected from each
  sample's size (a 10x rule of thumb, ~0.8% per 1,000 cells). Applying it writes
  `predicted_doublet`, keeps Scrublet's call in `predicted_doublet_scrublet`, and can put it back.
  The doublet step's expected rate now defaults to that suggestion instead of a fixed 6%.
- **GSEA.** Next to over-representation, GSEA on a DE result's whole ranked gene list, or a marker
  group's, against the chosen collections: NES, p, FDR and the leading-edge genes, with CSV. Click a
  term to score its leading edge on the plot.
- **TF and pathway activity.** CollecTRI transcription factors and PROGENy pathways scored per cell
  (decoupler's univariate linear model; scores match decoupler's to 5 decimals): colour the plot by
  one, see a group x source heatmap in Unified View → Activity, and score any DE result in the DE
  tab. The networks download on first use; human data only for now.
- **Marker specificity.** Marker tables show tau (how specific a gene is across groups) and its
  detection in the group against the next-best group, also in the CSV/Excel exports, with a plot of
  fold change against tau.
- **Pearson residuals.** Normalisation can use analytic Pearson residuals instead of the shifted log
  (the default). They choose the variable genes and feed PCA; markers, DE, scores and plots keep the
  log-normalised values. A mean-variance plot on Data Assessment compares them.
- **Filter cells.** Combine conditions on genes (expression above or below a value) and metadata
  (values of a column, a numeric range) with AND/OR, with a live count; "Select" makes the matching
  cells the selection for DE, labelling and highlighting.
- **Session files.** "Session" in the header saves the workspace (dataset, view, colouring, camera,
  overlays, selection, DE settings, colours) as a small JSON file, and opens one again, listing
  anything that no longer applies (a missing dataset, changed cells, a missing column or gene).
- **Cell-cycle caution.** Data Assessment and the QC report warn when most cells have S and G2M
  scores near 0, where phase calls are noise (Kang PBMCs: 95% of cells). Phase calls are unchanged.

### Changed
- **Group labels** stay inside the plot, no longer overlap (the bigger group keeps its place; the
  others move or wait for zoom), and sit where most of a group's cells are.
- **Hand-label columns** are picked as the default colouring and offered first in Compare, whatever
  they are called.
- **DE settings** survive switching tabs, and DE results carry each gene's test statistic.

### Fixed
- **DE on an unprocessed upload** tested the counts held in `.raw`, so fold changes were of counts
  (Kang: ISG15 log2FC 184 instead of 8.6). Cell-cycle scoring had the same problem. Both now use
  log-normalised expression.
- **Filtering** re-run on processed data counted genes in the scaled matrix instead of detected
  genes, and a % mito threshold on data without MT- genes removed every cell.
- **Marker Genes** could show "No marker genes available" when an earlier request answered last.

## [0.14.0] - 2026-10-16

### Added
- **Group labels on the plot.** Each group's name sits at its centre, readable on light and dark
  points, in split panels and in the PNG export. On by default up to 40 groups ("Labels" toggle).
- **Datasets open coloured by their cell types** (or hand labels, then clusterings) instead of the
  clustering. A colouring you picked yourself is kept when you switch datasets.
- **Compare colourings.** "Compare" shows the same cells coloured by 2-4 columns side by side, on
  one camera, with a legend per column and every value in the tooltip.
- **The hover tooltip leads with what the plot is coloured by:** the cell's group (with its
  colour), the number, or the gene's expression or gene-set score.
- **Split a cluster into sub-clusters.** The split icon on a legend entry re-clusters that group on
  its own genes, PCA and neighbour graph (with Harmony when the dataset was batch-corrected) and
  writes `<column>_sub`, leaving every other cluster as it was.
- **Enrich a DE result.** Under any DE result, "Enrich" runs over-representation on the significant
  up genes and, separately, the down genes, with the genes that were tested as the background and
  the table's cut-offs.
- **QC plots by sample, batch or cluster.** Data Assessment's QC plots get a "Group by" control:
  violins of genes, counts and % mito per group and the counts-vs-genes scatter in category
  colours. It starts on the donor column and stays light at 200k cells.
- **Batch/condition confounding check.** When each batch holds one condition (or batches sit
  inside conditions), a condition difference is also a batch difference. DE between groups, the
  joint-embedding plan and the QC report now say so.
- **Integration quality:** the joint embedding reports iLISI (how many datasets share each
  neighbourhood) and cell-type conservation before and after correction, and warns, naming the
  cell types, when correction looks like it merged real differences.
- **Dissociation-stress score.** A new pipeline step scores immediate-early and heat-shock genes
  per cell (`dissociation_stress_score`, after van den Brink et al. 2017), and Cluster QC flags a
  cluster whose score stands out and whose top markers are stress genes.
- **Species detection.** Human or mouse is called from Ensembl ids or gene-name case and shown on
  Data Assessment; enrichment on a mouse dataset notes that the gene sets are human.

### Changed
- **The server stays responsive during long work.** Pipeline runs, dataset loads, markers, DE,
  enrichment, ingest and Excel export run off the server's event loop. Before, a pipeline run
  froze every request (healthcheck included) for up to ~70 s on Kang. Requests for a dataset that
  is being processed wait up to 240 s and then get a "busy" answer; deleting a dataset mid-run is
  refused.
- **LLM cell-type labels use informative markers.** Ribosomal and mitochondrial genes are left out
  of the marker lists, and the model names the tissue first: on Kang, monocytes were called
  "macrophages", pDCs "plasma cells" and the main T-cell cluster "Ribosomal cells" before.
- **A legend click on any grouping opens that group's markers**, not group 0 of the clustering.

### Fixed
- **Dense-matrix data: markers, cell cycle and stress scores read scaled values.** Scaling a dense
  matrix (CSV uploads, dense `.h5ad`) also scaled the log-normalised copy in `.raw`, so most marker
  fold changes were missing and the Markers tab could crash. **Re-process such datasets from
  counts**; sparse data (10x, most `.h5ad`) was not affected. A missing fold change now shows a dash.
- **Cell-cycle scoring works on mouse data** (it failed with "No valid genes") and finds genes
  under their renamed symbols (CENPU, PIMREG, JPT1).
- **Data without MT- genes no longer reports 0% mito** as a measurement; QC plots, Cluster QC and
  the QC report say it can't be measured.
- **The saved neighbour graph is the one the clustering used** (UMAP was trimming it in place), so
  re-clustering and the resolution sweep agree with the pipeline.
- **DE by the column used as the batch-correction key** is no longer reported as a comparison
  between batches.

## [0.13.0] - 2026-10-16

### Added
- **Pseudobulk DE (donor-aware).** In DE, *Between groups*, a new test sums each donor's counts
  and tests donors rather than cells with DESeq2 (pydeseq2), so the p-values allow for
  donor-to-donor variation. It runs a paired design when every donor is in both groups, and the
  donor column is pre-selected. It needs raw counts and at least 2 donors per group.
- **Pseudotime inside the app.** The Trajectory panel can compute diffusion pseudotime from a start
  cluster or cell type you choose, and saves it with the dataset. You no longer have to re-upload
  a file that already has pseudotime.
- **Download figures.** Every chart's toolbar downloads a high-resolution PNG or an SVG, and the
  embedding scatter downloads a high-resolution PNG on white with its legend.
- **Download data.** From the header or the dataset library: the processed `.h5ad` (its size is
  shown first), the cell metadata as CSV, and the marker genes as CSV or Excel (one sheet per
  grouping).
- **Rename datasets** in the library, which now also shows when each dataset was added.
- **Joint embedding:** the tab shows how well the datasets mixed before and after batch
  correction (the share of each cell's 30 nearest neighbours from another dataset, against the
  ideal for their sizes), and which datasets each joint cluster is made of.
- **Marker dot plot and heatmap of your own genes:** type them or paste a list.
- **Ask the co-pilot about DE results**, for one gene or the whole result.
- **Colour choices:** a colour-blind-safe default palette (Okabe-Ito, then Paul Tol) and a Colours
  control in Unified View for categories and expression (viridis, cividis, magma, plasma).
- **HVG settings** in Data Assessment: the HVG flavour (`seurat`, `cell_ranger`) and a batch key.
- **"Open in Unified View"** button when a pipeline run finishes.
- **Split Unified View by a condition.** Choose a column with 2-6 values (a condition, sample or
  donor) to get one linked panel per value, with the other cells in faint grey. The panels share
  the camera, the category colours, one legend and one expression colour scale, so a gene can be
  compared across conditions; Download PNG saves all panels in one image.
- **Label cells by hand.** Select cells with the lasso or a legend click, choose "Label
  selection…", and save the label into an annotation column of your own. "Undo label" reverses
  the latest labelling.
- **Try clustering resolutions.** "Resolutions" compares several resolutions on the existing
  neighbour graph (cluster count, sizes, silhouette, and stability between neighbouring
  resolutions) without changing the dataset; "Keep this one" saves the chosen one as a clustering
  column.
- **Your own gene-set collections.** Upload a `.gmt` file (or a gene_set,gene CSV/TSV), then rename
  or delete it. Custom collections work alongside MSigDB, or without it, for enrichment, gene-set
  scoring and browsing, and show how many of their genes the current dataset measures. A file
  with problems is refused with a list of them by line.
- **Cluster QC on Data Assessment.** Each cluster's size, genes and counts per cell, % mito and
  ribo, and doublets, compared with the typical cluster, with plain-language flags for clusters
  that look low quality, doublet-rich or very small. The co-pilot answers "which clusters look low
  quality?" from these numbers, and the tip shown when a dataset opens names a cluster that stands
  out.
- **"Explain this data"** writes a short, cited QC report (Overview, Quality, Concerns, Suggested
  next steps) from the assessment numbers only. It is saved per dataset, so reopening the tab
  doesn't call the model again; without an API key it lays out the same facts.
- **The co-pilot shows what its literature search covers:** how many papers and tutorial documents
  it can search, and when the index last grew.

### Changed
- **A category has the same colour in every panel:** embedding, legend, cluster map, violins and
  the Observations charts.
- **Clicking a cluster** in the legend or the cluster map shows its marker genes in Unified View.
- **Marker genes for a grouping are computed once and kept**, instead of recomputed (up to ~30 s)
  on every visit; the loading message says when they are being computed.
- **Analysis guards:** marker tables say cluster-marker p-values are for ranking only;
  between-groups DE reports the donor column, donors per side and the largest donor's share, and
  warns when a side has fewer than 2 donors or one donor holds most of its cells; Data Assessment
  flags data normalised to the wrong scale (CPM ~1e6 instead of ~1e4).
- **The co-pilot can open the Compare panel and the DE subtab.**

### Fixed
- **Louvain clustering works.** The `louvain` package was never installed; it now uses
  python-igraph, seeded, and honours the resolution.
- **Enrichment results for labels containing "/"** (such as CellTypist's "Tcm/Naive helper T
  cells") are no longer lost when the dataset is saved.
- **DE picked the wrong donor column** on datasets whose per-sample column repeats the condition
  (Kang's `orig_ident`), and reported one donor per side.
- **Provenance no longer records default clustering settings** for runs that did not cluster.
- **Colouring by pseudotime** no longer turns every cell the same colour when some cells are
  unreachable.
- **Gene-set scoring ignores case**, so MSigDB sets score on mouse data, and enrichment no longer
  fails when a collection shares no genes with the marker list.
- **The gene overlay label** no longer hides under the legend, and the Joint embedding dataset
  picker shows a loading state instead of "No ready datasets".

### Development
- CI runs the backend and frontend checks on every pull request, and a smoke test pulls each
  release's published images and starts them on amd64 and arm64 (the images Apple Silicon Macs
  use).
- `npm run lint` works again; ten unused frontend packages were removed; the ruff rule set and
  `anndata<0.13` are pinned so fresh installs behave like the tested ones; the unused websocket
  stub (it echoed whatever a client sent) is gone.

## [0.12.0] - 2026-10-16

### Changed
- **The AI features use DeepSeek-V4.1-Flash by default** instead of Llama-3.1-8B. On scView's own
  requests it gave better co-pilot answers, proposed a complete pipeline for raw data (Llama
  skipped filtering, PCA and clustering), annotated about 12 of 13 cell types where Llama
  managed about 6, and answered fastest. Its "thinking" mode is switched off, and the pipeline
  advisor now follows the model setting instead of always using Llama. **Existing installs keep
  Llama**, because the old `.env.example` wrote `RAG_CHAT_MODEL` into `.env`: change that line to
  `RAG_CHAT_MODEL=deepseek-ai/DeepSeek-V4.1-Flash`, or delete it.
- **Windows install steps** in the README now cover the WSL prompts on a PC that hasn't used WSL,
  where Docker Desktop shows "Engine running", and a download that looks stuck near the end, with
  screenshots.

### Fixed
- **Security: deleting a dataset with a path-like id removed the whole data folder.** Dataset and
  session ids must now be plain ids, and uploads keep only the file's own name.
- **A pipeline run stopped partway (tab closed, proxy timeout) no longer leaves half-processed
  data behind.** Running the steps again could log-transform the data twice.
- **Clustering keeps a dataset's own `leiden` column**, and re-running PCA no longer reuses the
  previous run's Harmony embedding.
- **Differential expression returns down-regulated genes** when the data holds only the highly
  variable genes. Before, the volcano plot showed only up-regulated ones.
- **QC counts per cell come from counts.** Data Assessment summed scaled data, so Kang showed a
  median of -286 counts per cell. Files over 2 GB also work.
- **Enrichment and markers stay right after re-annotating, renaming or merging labels.** Cached
  enrichment for a surviving label was served after re-annotation, a renamed label lost its
  markers and enrichment, merging labels created a "nan" group, and markers saved under scanpy's
  default name were not found ("No marker genes found").
- **Trajectory** no longer fails on cells the trajectory can't reach; they are drawn light grey.
  Gene trends use the expression layer instead of scaled data.
- **Errors:** metadata with missing values and expression export for genes kept only in `.raw` no
  longer fail with a server error.
- **Yes/no columns** such as `predicted_doublet` get a proper True/False legend.
- **Cell annotation no longer replaces the Marker Genes panel's default table** with the new
  column's markers.
- **Switching datasets resets the cell selection and grouping**, which carried over (a
  1,113-cell dataset showed "355 selected").
- **Enrichment results stay with the group they were computed for** when you change the group.
- **Unified View markers always match the grouping column.**
- **The co-pilot answers an "ask about this" question once.** It was answered twice, doubling
  the AI cost.
- **The Expression tab's layer menu works**, and matches the toolbar's Layer menu.
- **Highlighting a group selects its cells** in Unified View, from an Observations row or the
  co-pilot.

## [0.11.2] - 2026-10-15

### Fixed
- **Enrichment works for annotated cell types.** Cell-type annotation now also computes marker
  genes for its new column, so enrichment and the marker plots can use it straight away. Before,
  choosing the cell-type column in Unified View's Enrichment tab listed the cluster numbers
  (0-9) instead of the cell types, and Compute failed.
- **A column without marker genes is explained.** The Enrichment tab says so and offers a
  "Compute markers for this column" button, instead of showing another column's groups. This
  also fixes datasets annotated with an earlier version.

## [0.11.1] - 2026-10-15

### Fixed
- **The co-pilot describes the marker plots accurately.** It said filtering the marker table changes
  the heatmap (it doesn't) and suggested follow-up questions about plot settings scView doesn't have.
  It now knows the plots' limits and that the camera icon in their toolbar downloads a PNG, and its
  suggested follow-ups stick to the data and the biology.
- **"Can I see a heatmap of these markers?" opens the heatmap**, like "Can you show…" already did.

## [0.11.0] - 2026-10-15

### Added
- **Marker gene dot plot and heatmap.** The Marker Genes panel can switch between the table, a dot
  plot and a heatmap of the top up-regulated markers of every group (3, 5 or 10 per group). Colour
  is mean expression scaled per gene; in the dot plot, dot size is the share of cells in the group
  that express the gene.
- **The co-pilot opens them.** "Can you show a heatmap of these markers?" opens the heatmap, and
  asking for a dot plot opens the dot plot.

### Fixed
- **The co-pilot no longer describes features scView doesn't have.** Asked for a heatmap, it used
  to point to a heatmap option that didn't exist. Its list of scView's features is now complete,
  questions about scView's screens always include it, and it is told not to go beyond it.
- **Code blocks in co-pilot answers render as code** instead of running together into one paragraph.

## [0.10.3] - 2026-10-15

### Fixed
- **scView no longer fails to start on slower computers with "container scview-backend-1 is
  unhealthy".** Docker gave the backend only about 30 seconds to answer its health check, and a
  slow machine (seen on a Windows 11 virtual machine) can take longer than that to load. It now
  has two minutes. Starting scView a second time was a workaround.

## [0.10.2] - 2026-10-15

### Fixed
- **Unified View no longer says "No embeddings available" after the pipeline has run.** The
  released backend image ran two server processes, each with its own copy of the dataset in
  memory. After a pipeline run only one of them had the results, so pages showed the processed or
  the raw data depending on which process answered, and Data Assessment flickered between
  "13 completed" and "0 completed". The backend now runs a single process, as the development
  setup always has. Restarting scView was a workaround on 0.10.0 and 0.10.1.

## [0.10.1] - 2026-10-14

### Fixed
- **A Seurat `.rds` added together with other files no longer fails with "Internal Server Error".**
  The upload wizard now explains that `.rds` files are converted separately and asks for the file on
  its own. A single `.rds` was already converted correctly.
- **Unreadable files get a clear error.** A file that can't be read is reported with the reason
  instead of a server error, and a damaged file that parses to no cells or no genes is rejected
  instead of being saved as an empty dataset.

## [0.10.0] - 2026-10-14

### Added
- **Prebuilt images.** Each release now publishes the backend, frontend and converter images to
  GitHub's container registry (`ghcr.io/thirtysix/scview-*`) for amd64 and arm64, so installing
  downloads about 1 GB instead of building everything locally. `docker-compose.yml` pins them to the
  release and still builds any image that isn't available; `./start.sh --prod --build` or
  `docker compose up -d --build` builds from source.
- **Windows install without a command line.** `start-windows.bat` starts Docker Desktop if needed,
  creates `.env` (asking for an optional API key), starts scView and opens the browser;
  `stop-windows.bat` stops it. The README has a Windows step-by-step guide and troubleshooting table.
- **MSigDB gene sets download on first start.** The collections were never in the repository, so a
  fresh install had no gene sets for enrichment or scoring. The backend now fetches the 22
  collections (~45 MB) from the Broad Institute into a Docker volume, in the background.
  `MSIGDB_AUTO_DOWNLOAD=false` turns this off.
- **README table of contents.**

### Changed
- **Fixed Compose project name (`scview`)**, so datasets stay available when an update is unpacked
  into a differently named folder (a GitHub ZIP unpacks to `scView-main`). Upgrading from a folder
  with another name: earlier datasets are in the Docker volume `<folder name>_dataset_data`, which
  this version no longer uses.
- **`.env` is optional.** Requires Docker Compose 2.24 or newer, which `start.sh` now checks.
- **`./start.sh --clean` asks before deleting**, since it removes every uploaded dataset.
- The backend's own log messages now appear in `docker logs`.

### Fixed
- **The converter image build fails when an R package is missing**, instead of producing an image
  that cannot convert anything.
- README: the sample-data link now points at a downloadable file (the old path was not in the
  repository), and the disk estimate is corrected.

## [0.9.0] - 2026-10-14

### Added
- **Joint embedding across datasets.** A *Joint embedding* tab in the Compare panel merges 2–8
  library datasets on their shared genes, labels each cell by its dataset, and runs batch-aware HVG
  selection, PCA and Harmony, then joint clustering, UMAP and markers. Raw counts are recovered from
  the original upload when processing had replaced them. The uncorrected UMAP is kept beside the
  corrected one, so the effect of correction is visible. On Kang 2018 (IFN-β, CTRL vs STIM) the
  share of each cell's neighbours from the other condition rises from 0.11 to 0.48 (0.50 is perfect
  mixing) while cell types stay separate.
- **Between-groups differential expression.** A *Between groups* mode in the DE subtab tests group A
  against group B, optionally within one cluster or cell type (for example STIM vs CTRL monocytes).
  The UI notes that cell-level p-values are optimistic when cells from one donor are not independent.
- **Enrichment Programs view.** Collapses a wall of near-duplicate enrichment terms into a few
  biological programs (clustered on shared genes, EnrichmentMap method), with a similarity graph;
  click a program to score its genes on the embedding.
- **"Exclude ribosomal/mito genes" toggle** on both enrichment surfaces (on by default).

### Fixed
- **Raw counts kept through processing.** An upload carrying counts only in `.raw` lost them at the
  scaling step, which overwrote `.raw` with log-normalised values. They are now kept in
  `layers['counts']`. CellTypist annotation now builds its input from those counts instead of from
  `.raw`, which after scaling held log-normalised values that were then normalised a second time.
- **Pathway enrichment (ORA) correctness.** The background is now the genes the experiment measured
  (previously the union of the selected collections), the gene-count control works above 100 genes,
  cached results are reused only when their parameters match, and ribosomal/mitochondrial genes are
  filtered from the query, which fixes wrong-tissue cell-type-signature (C8) labels.

## [0.8.0] - 2026-06-11

### Added
- **Compare datasets.** A new Compare panel puts two datasets side by side — pick a shared
  categorical column (cell type / cluster) to see their composition as grouped proportions plus a
  Δ table, with one click to ask the co-pilot what the differences mean.
- **Batch-integration quality score.** The Observations panel reports how evenly each cluster mixes
  across a batch/condition (normalized-entropy score, 0–1), with the least-mixed clusters called out
  and a co-pilot tie-in.
- **Workspace restored across reloads.** The active panel, scatter color/embedding, and last-open
  dataset persist, so a refresh drops you back where you were.
- **"Ask about this" on QC plots and Observations rows** — the contextual co-pilot affordance now
  covers the QC distributions (with the metric values embedded) and each composition group.

### Changed
- **Smaller, faster Plotly.** A custom partial Plotly bundle (only the trace types scView uses)
  cuts the charting payload ~68% (4.9 MB → 1.5 MB), loaded lazily.

## [0.7.0] - 2026-06-10

### Added
- **Deployment-mode safety flag.** A new `DEPLOYMENT_MODE` setting (`private` by default, `public`
  when you expose scView beyond localhost) makes the safe path the default: in `public` mode the
  backend runs a startup self-check that loudly logs what's still missing. An optional `ACCESS_TOKEN`
  adds a coarse shared-secret gate on every API route (a no-op when unset, so localhost is unchanged).
- **Always-on input hardening.** Caps on co-pilot query length/word count and conversation-history
  size, plus a JSON body-size limit, reject oversized requests before they reach the model. The
  co-pilot system prompt gained a "treat user input as data, not instructions" clause.
- **`SECURITY.md`** documenting the deployment posture and exactly what a public/multi-user
  deployment still needs (per-user auth, rate limiting, quotas).
- Expanded the frontend test suite to 35 tests (co-pilot citation/markdown rendering, color mapping,
  unified-view store).

## [0.6.0] - 2026-06-10

### Changed
- **Much faster first load via code-splitting.** Panels are lazy-loaded and heavy libraries
  (Plotly, deck.gl, Arrow, React) are split into their own cacheable chunks. The initial JavaScript
  payload drops from ~6.2 MB to ~73 kB (app shell) plus React; Plotly and deck.gl now download only
  when you open a panel that uses them, and an app-code change no longer invalidates the vendor cache.

### Added
- **Frontend test suite (Vitest).** `npm test` runs jsdom + Testing Library tests; initial coverage
  spans CSV escaping, formatting helpers, the co-pilot ask queue, and the color-legend component.

## [0.5.0] - 2026-06-10

### Added
- **Differential expression with a volcano plot.** A new "DE" tab in the Unified View: lasso (or
  click) a population, then compute one-vs-rest Wilcoxon differential expression over all genes,
  shown as an interactive volcano plus a significant-gene table. Click a point or row to overlay
  that gene's expression; export the full table to CSV. Backed by `POST /datasets/{id}/de`.

### Changed
- README refreshed for the v0.2–v0.5 feature set (natural-language actions, ask-about-this,
  write-methods, reranking, proactive insight, three-way annotation, DE/volcano), with new
  screenshots (DE volcano, insight banner, write-methods, literature-grounded answers).

## [0.4.0] - 2026-06-10

### Added
- **"Write methods" export.** The co-pilot turns your recorded provenance recipe (steps, tools,
  parameters) into a methods-section paragraph, grounded strictly in what was run — it never invents
  methods. A "Write methods" button appends it to the conversation; works as a deterministic digest
  with no LLM key.
- **Cross-encoder reranking** of literature/tutorial retrieval. When a reranker model is configured,
  retrieval casts a wider net and a DeepInfra reranker reorders candidates by direct query relevance
  for sharper citations; it degrades to the hybrid order if unavailable.
- **Proactive insight gets QC-anomaly detection and optional polish.** The "I notice…" banner now
  flags high mitochondrial content (stressed/dying cells), and an actionable nudge can be reworded
  into a friendlier sentence by the LLM (facts preserved; the deterministic text is the fallback).
- **Transparency affordances.** Each answer shows which model produced it with an "AI-generated —
  verify" note, plus a per-answer 👍/👎 rating.

## [0.3.0] - 2026-06-10

### Added
- **Co-pilot "ask about this".** A ✦ button surfaces a contextual question on each categorical-legend
  group, the gene expression overlay, every marker-gene row, and every enrichment-term row — it opens
  the co-pilot pre-loaded with that question (and highlights the group so the answer resolves).
- **Proactive insight on load.** A deterministic one-line "I notice…" banner when a dataset opens,
  picking the most useful next step by walking the preprocessing state in order (raw counts → doublet
  load → condition/batch split → cluster → annotate → done), with a click-to-ask follow-up the co-pilot
  answers or executes. No LLM call, so it's reproducible.
- **Conversation persistence.** Per-dataset co-pilot threads survive reloads and dataset switches
  (localStorage), with a Clear control.
- **Offline `marker_score` annotation.** A third cell-type method that scores curated marker sets per
  cluster — deterministic, no network, no model download.
- **Edit/override cell-type labels.** Inline-rename any categorical label from the color legend;
  persists to the derived layer (never the original upload), merging if the target label exists.
- **CSV export on every results table.** Markers, enrichment, gene sets, and the Observations
  summary + composition now export through a shared RFC-4180 helper.

### Fixed
- CSV export no longer corrupts rows whose cells contain commas (e.g. enrichment term names).
- Citation chips render reliably even when the model reformats a citation tag.

## [0.2.0] - 2026-06-10

### Added
- **Natural-language actions in the AI co-pilot.** Type a command and it drives the UI:
  view/navigation commands ("color by cluster", "show CD8A expression", "switch to 3D", "go to
  marker genes", "group by stim") apply immediately and reversibly; mutating commands ("annotate
  cell types", "cluster at resolution 1.0", "compute markers", "run enrichment", "detect doublets")
  are **confirm-gated**, showing an overwrite advisory and a rough time estimate before anything
  runs. Every action is re-validated server-side against a strict allow-list, the LLM handles
  parametrized commands while a deterministic matcher catches unambiguous ones (working even with no
  LLM key configured).

## [0.1.0] - 2026-06-10

Initial public release.

### Added
- **AI co-pilot** — grounded, cited chat over your dataset's results and provenance plus a
  dual-corpus scRNA-seq literature/tutorials RAG. Intent routing keeps it cheap, it's context-aware
  ("what is this cluster?" resolves to the highlighted cluster and its cell type), and it works even
  before a dataset is loaded.
- **AI-assisted data assessment** — a deterministic assessor reports the state of ~15 preprocessing
  steps; an LLM advisor recommends the next steps with reasons and sized parameters (you approve).
- **Provenance — "git for the h5ad"** — every analysis step is recorded into the data with
  dependency-aware *edit & re-run from here* and an exportable, replayable recipe.
- **Cell-type annotation** — tissue-agnostic LLM-from-markers (default, no reference model to pick)
  and reference-based CellTypist; runs on any chosen clustering, writing an explicit obs column.
- **Unified View** — one linked screen: scatter (2D/3D) + tabbed Markers/Expression/Gene Sets/
  Enrichment + violin, scaling to ~200k cells via a FastAPI backend, Apache Arrow, and deck.gl.
- **Multi-format ingestion** — h5ad, 10x MEX/HDF5, loom, zarr, dense CSV, Seurat `.rds`, and
  nf-core/scrnaseq outputs, via a guided import flow.
- **On-demand analysis pipeline** — QC, doublet detection, normalization, HVG, PCA, Harmony batch
  correction, Leiden/Louvain clustering, UMAP/t-SNE, marker genes, MSigDB enrichment, cell cycle.
- **Docker self-host** — `./start.sh` or `make` on Linux, macOS, and Windows.

[Unreleased]: https://github.com/thirtysix/scView/compare/v0.11.2...HEAD
[0.11.2]: https://github.com/thirtysix/scView/releases/tag/v0.11.2
[0.11.1]: https://github.com/thirtysix/scView/releases/tag/v0.11.1
[0.11.0]: https://github.com/thirtysix/scView/releases/tag/v0.11.0
[0.10.3]: https://github.com/thirtysix/scView/releases/tag/v0.10.3
[0.10.2]: https://github.com/thirtysix/scView/releases/tag/v0.10.2
[0.10.1]: https://github.com/thirtysix/scView/releases/tag/v0.10.1
[0.10.0]: https://github.com/thirtysix/scView/releases/tag/v0.10.0
[0.9.0]: https://github.com/thirtysix/scView/releases/tag/v0.9.0
[0.8.0]: https://github.com/thirtysix/scView/releases/tag/v0.8.0
[0.7.0]: https://github.com/thirtysix/scView/releases/tag/v0.7.0
[0.6.0]: https://github.com/thirtysix/scView/releases/tag/v0.6.0
[0.5.0]: https://github.com/thirtysix/scView/releases/tag/v0.5.0
[0.4.0]: https://github.com/thirtysix/scView/releases/tag/v0.4.0
[0.3.0]: https://github.com/thirtysix/scView/releases/tag/v0.3.0
[0.2.0]: https://github.com/thirtysix/scView/releases/tag/v0.2.0
[0.1.0]: https://github.com/thirtysix/scView/releases/tag/v0.1.0
