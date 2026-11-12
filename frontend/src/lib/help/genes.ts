/** Help for the gene-centred Unified View tabs. Keys: `tab.<id>` for the tab itself, `<tab>.<thing>` inside it. */
export const GENES_HELP: Record<string, string> = {
  "tab.markers":
    "Genes that set each group apart from the rest. Use it to see what makes a cluster distinct and to name cell types.",
  "tab.expression":
    "Colour the cells by a gene's expression and compare groups in the violin, or look at several genes at once. Use it to check where a known marker is on.",
  "tab.genesets":
    "Browse gene sets (MSigDB and your own) and score each cell for one, to see where a pathway or programme is active.",
  "tab.enrichment":
    "Pathways over-represented among a group's marker genes, or ranked by GSEA, so a cluster reads as biology rather than a gene list.",
  "tab.de":
    "Differential expression: a selection against the rest, or one group against another, with cell-level or per-donor (pseudobulk) tests and a volcano plot.",
  "tab.activity":
    "Transcription factor and pathway activity per cell, inferred from each one's known target genes; also cytokine response and TF candidates from promoter motifs.",
  "tab.genelists": "Save and name your own gene lists, and reuse them in the other tabs and in later sessions.",
  "tab.report":
    "One page of evidence about a cluster or a gene, gathered from results scView already has, with an HTML download.",
  "tab.programs":
    "Gene programmes (NMF): sets of genes that rise and fall together across cells, and how much each cell uses each one.",

  // Markers
  "markers.groups":
    "Pick a grouping column (its number of groups in brackets) and one group (its number of kept markers in brackets). Each group's cells are compared with all other cells, gene by gene, with a Wilcoxon test.",
  "markers.table":
    "LogFC: log2 fold change against all other cells (above 0: higher in the group). Adj.P: the p-value corrected for testing many genes; use it to rank, not as proof, since the groups came from these same cells. Tau, In / next and Ratio say how specific a marker is. Click a row to colour the plot by the gene.",
  "markers.min-fc":
    "Hide markers whose log2 fold change is smaller than this, in either direction. 1 means twice the mean expression of the other cells, 2 means four times.",
  "markers.specificity-plot":
    "Each point is a marker: its fold change (x) against tau (y), the part of its expression that belongs to this group alone. Top right: strong and specific. Strong but low: shared with other groups.",
  "markers.minimal-panel":
    "The fewest genes that still tell the groups apart, added one at a time and checked by a classifier on held-out cells. Useful when designing a small panel for sorting, imaging or qPCR. Balanced accuracy: the mean share of each group's cells called correctly.",
  "markers.panel-curve":
    "Balanced accuracy (y) as genes join the panel (x). The solid line is the selected panel; the dashed line takes each group's top marker in turn. Where the curve flattens, more genes add little.",
  "markers.panel-confusion":
    "Rows are the true groups, columns the group the panel's model calls each cell. A strong diagonal means the panel separates the groups; colour off the diagonal shows which groups it confuses.",

  // Expression
  "expression.search":
    "Type a gene symbol to colour every cell on the plot by its expression and to show it in the violin under the plot, group by group.",
  "expression.layer":
    "Which matrix the colours and violins read. Log-normalised values suit most uses; a counts layer holds raw molecule numbers; scaled values are centred per gene, so they go below zero. The same setting as the toolbar's Layer menu.",
  "expression.group-by": "The column whose groups the violin under the plot compares, for example clusters or cell types.",
  "expression.several-genes":
    "Pick 2 to 24 genes to see them together: a feature grid (one plot per gene), a stacked violin (genes by groups) or their joint density. With two genes, Blend and Gene vs gene show where they overlap.",
  "expression.stacked-violin":
    "One row per gene, one column per group. Each violin is wide where many of the group's cells sit at that expression level. Every gene has its own scale, so compare groups within a row, not rows with each other.",
  "expression.two-genes":
    "Blend colours the plot by both genes at once, so cells expressing A, B or both stand out. Gene vs gene plots each cell's A against its B, with correlations and how many cells express both, one or neither.",
  "expression.gene-vs-gene":
    "Each point is a cell: gene A on x, gene B on y, coloured by the group-by column. Cells along an axis express only one of the genes. Click a count below, or draw a box on the plot, to select those cells.",
  "expression.feature-grid":
    "One small plot per gene, all on the same view, to see where each gene is expressed. Shared scale gives every gene one colour range, so levels compare; off, each gene uses the full range.",
  "expression.blend":
    "Each cell's colour mixes gene A and gene B: in the red / green scheme, red cells express A, green cells B and yellow cells both. Top at caps each gene's colour at a high quantile, so a few extreme cells do not wash out the rest.",

  // Per donor (the violin's Per donor mode)
  "perdonor.plot":
    "One point per donor or sample in each group: the gene's value over that donor's cells, with a box for the spread. A change seen in every donor is more trustworthy than one seen only in pooled cells.",
  "perdonor.split-by":
    "A condition column (for example treated and control) that splits each group's donors, to see whether the gene moves with it donor by donor.",
  "perdonor.donor":
    "The column naming the donor or sample each cell came from. Auto picks one that looks like it; choose another if the guess is wrong.",
  "perdonor.value":
    "Mean log-norm: the average log-normalised expression over the donor's cells, on the violin's scale. log2 CPM: the donor's raw counts summed and scaled per million, as pseudobulk DE uses them; deeper cells weigh more.",

  // Gene sets
  "genesets.collection":
    "A library of gene sets to browse, such as MSigDB Hallmark (well-defined biological processes), Reactome or GO, or a collection you uploaded or saved from a gene list.",
  "genesets.list": "The sets in the chosen collection with their number of genes. Pick one, then Score it.",
  "genesets.score":
    "Gives each cell one number for how strongly it expresses the set's genes together, and colours the plot by it. Found: set genes this dataset has; missing: not measured here. The scale depends on the method, so compare cells and groups, not values between methods.",
  "genesets.group-by": "The column whose groups the violin under the plot compares for the score.",

  // Enrichment
  "enrichment.group":
    "The column and group whose marker genes are tested. Enrichment starts from the column's own markers, so a column without them needs markers computed first.",
  "enrichment.top-n":
    "How many of the group's top markers (highest fold change first, after the marker filters) Compute tests as one list. GSEA ignores it: it ranks every gene.",
  "enrichment.gsea":
    "NES: positive when the set's genes sit near the top of the ranking (higher in the group, or in the first group of a DE), negative near the bottom. FDR below 0.05 is the usual cut-off. Lead: the set's genes that drive the score, out of its genes in the ranking. Click a row to score the leading edge.",

  // Pathway diagram
  "pathway.diagram":
    "A Reactome pathway: boxes are proteins, genes, complexes and small molecules, arrows are reactions. Boxes take each gene's log fold change from the result chosen under Colour by (a group's markers or a kept DE result), so you can see which steps change together. Click a sub-pathway box to open it.",

  // DE
  "de.mode":
    "Selection vs rest: cells you lasso, or a group you click in the legend, against all other cells. Between groups: two values of one column, such as treated and control, optionally within one cell type.",
  "de.compare": "The column holding the two groups to compare, such as condition, treatment or sample.",
  "de.groups":
    "Group A (left) against group B (right). Up genes are higher in A, down genes higher in B. The arrow button swaps them.",
  "de.within":
    "Restrict both groups to the cells of one value of another column, for example treated against control within one cell type. Without it, changes in which cell types each side holds mix into the result.",
  "de.test":
    "Cell-level tests treat every cell as a replicate: fast, but p-values are too optimistic when the groups come from different donors. Pseudobulk sums each donor's counts and tests donors, so it allows for donor-to-donor variation; prefer it for conditions with several donors per side.",
  "de.method":
    "Wilcoxon rank-sum (the default) and Welch t-test test each gene on its own; logistic regression fits all genes at once and gives coefficients without p-values. Compare methods runs all three and shows how far they agree.",
  "de.min-pct":
    "Test only genes detected in at least this share of the cells of either group. Fewer tests make the multiple-testing correction fairer and keep near-absent genes off the edges of the volcano.",
  "de.donor":
    "The column naming the donor or sample each cell came from. Pseudobulk sums the counts per donor in each group, so pick the column for independent biological replicates.",
  "de.covariates":
    "Per-sample columns, such as sex or batch, to adjust for, so their effect is not counted as part of the comparison. Up to 4; a column that matches the comparison is refused.",
  "de.min-fc":
    "A gene counts as up or down when its adjusted p is at most 0.05 and its |log2 fold change| at least this; 1 means a twofold difference. It sets the table, the volcano colours and the Up and Down lists.",
  "de.detection":
    "How many genes are detected in what share of cells, taking the larger share of the two groups. The dashed line is the Genes cut-off: genes left of it (grey) were not tested.",
  "de.volcano":
    "Each point is a gene: log2 fold change on x (right: higher in the first group), significance on y (higher: smaller adjusted p). Red: up, blue: down, grey: short of a cut-off (dashed lines). Logistic regression plots its coefficient on y. Click a point to colour the cells by the gene.",
  "de.table":
    "The genes past both cut-offs, most significant first. log2FC above 0: higher in the first group. Adj.P: the p-value corrected for testing thousands of genes. Click a row to colour the cells by the gene; the CSV has every tested gene.",
  "de.transcriptome-size":
    "Shown when the two sides differ a lot in total RNA per cell. Normalising each cell to the same total then shifts fold changes, so a gene can look up or down partly because of cell size. The violins show each side's total counts per cell.",
  "de.overlap":
    "Compare the significant genes of several DE results the server has kept, for example one condition within several cell types, to see which genes are shared and which belong to one result.",
  "de.upset":
    "Each column is one combination of results: the bar on top counts the genes significant in exactly those results (filled dots). The bars on the left count each result's genes. Click a column to list its genes.",
  "de.activity-network":
    "TFs (CollecTRI): transcription factors, scored from their target genes. Pathways (PROGENy): signalling pathways, scored from genes that respond to them. Positive: more active in the first group.",

  // Activity
  "activity.group-by":
    "The column whose groups the heatmap averages over. None: the list only, without the heatmap.",
  "activity.heatmap":
    "Mean activity of each TF or pathway (rows) in each group (columns). Red: its activated targets are high and repressed ones low, so it looks active; blue: the reverse; white: no signal. Click a row to colour the cells by it.",
  "cytokine.mode":
    "Each group vs the others: which signals each group looks like it received. Two conditions: how a treatment changes that response, in all cells or within each group.",
  "cytokine.reference": "What each group's expression is compared with: the rest of the cells, or all cells including the group.",
  "cytokine.heatmap":
    "CytoSig z-score of each cytokine (rows) in each group or comparison (columns). Red: the expression change looks like the response to that cytokine; blue: the opposite. The list below names those with z of 2 or more.",

  // Gene lists
  "genelists.lists":
    "Your gene lists, kept in this browser and saved with a session file. Import a .txt, .csv or .gmt file, or add genes from the Markers, DE and Expression tabs.",
  "genelists.actions":
    "Feature grid: one plot per gene. Dot plot and Heatmap: the genes by group, on the Marker Genes page. Score: one score per cell for the whole list. Enrichment: pathways over-represented in the list. Save as collection: use the list as a gene set in other tabs.",

  // Report
  "report.mode":
    "Cluster: a report on one group of a column, from its size and donors to markers, QC and composition. Gene: a report on one gene across the groups of a column.",

  // Programs
  "programs.k":
    "How many programmes NMF looks for. Too few merges distinct processes; too many splits one into fragments. Tick compare k to see how stable other values are.",
  "programs.seeds":
    "How many times NMF runs, each from a different random start. A programme found in every run is marked stable; more seeds take longer but make that call firmer.",
  "programs.sweep":
    "For each k: the silhouette (left axis; higher means the runs agree on the programmes) and the relative error (right axis; it always falls as k grows). Prefer a k where the silhouette is still high.",
  "programs.table":
    "One row per programme: its top genes, the share of cells where it is the largest part (Top in), its mean usage share, and its best Hallmark set. Stable: found in every run. Click a row to colour the cells by its usage.",
  "programs.heatmap":
    "Mean usage share of each programme (rows) in each group (columns): the darker the blue, the more the group's cells use it. Click a row to colour the cells by it.",
};
