# Changelog

All notable changes to scView are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **A memory limit for scView, and a gauge for it.** The start scripts ask how much memory scView
  may use on the first start, suggesting free memory minus 3 GB to the nearest GB (at most what
  Docker or Podman has, at least 4 GB), and store it as `SCVIEW_MEMORY_GB` in `.env`; the backend
  and converter get it as a container limit with no extra swap. A dataset that needs more stops
  scView instead of the computer: on Linux nothing else caps Docker or Podman, and a big run could
  make the whole machine unresponsive. Without the setting there is no limit, as before. The
  sidebar shows the backend's memory use against its limit (`GET /system/memory`), amber from 75%
  and red from 90%.

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
