/** Help for the spatial and ATAC Unified View tabs. Keys: `tab.<id>`, `<tab>.<thing>`. */
export const SPATIAL_HELP: Record<string, string> = {
  "tab.spatial": "Genes whose expression forms a pattern across the tissue rather than being scattered at random.",
  "tab.transcripts": "Single transcripts drawn as dots on the tissue, gene by gene (Xenium).",
  "tab.domains":
    "Regions of the tissue with a similar make-up, which cell types sit next to each other, and recurring niches.",
  "tab.spotcomp": "Estimate which cell types make up each spot or bin, with a single-cell dataset as the reference.",
  "tab.accessibility":
    "Chromatin accessibility (10x Multiome ATAC): peaks, motifs, gene activity and links between peaks and genes.",

  // Tissue plot controls (toolbar, when the plot shows the tissue)
  "tissue.section":
    "This dataset holds several tissue sections and the tissue plot shows one at a time. Choose the section to draw; the others are hidden from the plot, not removed from the data.",
  "tissue.image":
    "Shows the microscope image of the section (an H&E stain on Visium) under the dots, so expression can be related to the tissue's structure. The slider sets how strongly the image shows through.",
  "tissue.draw":
    "Points draws each cell as a dot. Cells or Nuclei draws the outlines the imaging run segmented instead, once you zoom in far enough that few cells are in view; zoomed out, the section stays as points.",

  // Spatial genes
  "spatial.genes":
    "Which genes to test. Variable genes uses the genes the pipeline marked as highly variable (every gene when none are marked) and is quicker. All genes can also find patterns in genes that vary little between cells, and takes longer.",
  "spatial.section":
    "Test one tissue section, or all of them together. With all sections, neighbours are found within each section and each section is centred on its own mean, so a gene that is only higher on one slide does not count as a pattern. Each section's own I is shown beside the pooled one.",
  "spatial.by-section":
    "Moran's I of the gene within each section alone, the same value a run on that section gives. A pooled I carried by one or two sections shows here.",
  "spatial.table":
    "A high Moran's I with a small adjusted p means the gene is high in some areas of the tissue and low in others. Click a gene to colour the tissue by it and see where.",

  // Transcripts
  "transcripts.genes":
    "Choose genes to draw their single molecules as dots on the tissue, one colour per gene. It shows where each transcript lies within and between cells, which per-cell counts cannot show. The number beside a gene is how many of its transcripts are in view.",
  "transcripts.dot-size":
    "The size of each transcript dot on screen. Larger dots are easier to see zoomed out; smaller ones keep molecules that lie close together apart.",
  "transcripts.check":
    "Checks that the transcripts belong to this dataset: per gene, the transcripts that fall in a cell are counted and compared with that gene's total in the count matrix. r near 1 and small differences mean the two agree.",
  "transcripts.outlines":
    "The cell and nucleus shapes the Xenium run segmented. With them the tissue plot can draw cells as outlines instead of dots when you zoom in, which shows which transcripts fall inside which cell.",

  // Domains
  "domains.k":
    "How many domains to split the tissue into. Too few merges distinct regions; too many splits one region into pieces. Suggest k tries a range of values and picks the most stable one.",
  "domains.hops":
    "How far around each spot, bin or cell the method looks: 1 hop is its direct neighbours, 2 adds their neighbours, and so on. More hops tend to give larger, smoother domains; 0 ignores position and clusters on expression alone.",
  "domains.components":
    "How many principal components (summaries of the expression profile) describe each spot, bin or cell. More components keep finer differences, but each domain then has many more parameters to fit, so the fit is slower.",
  "domains.save-as":
    "The name of the new metadata column that holds each spot's, bin's or cell's domain. It appears in the colour-by and group-by menus, for example to find marker genes per domain.",
  "domains.stability":
    "Each point is one number of domains (k). Higher stability means repeated fits at that k agree with fits at the neighbouring values, so the split is not an accident of the starting point. Peaks are good candidates; click a point to use that k.",
  "domains.table":
    "How many spots, bins or cells fall in each domain, numbered from largest to smallest, with a column per section when there are several. A domain found in only one section deserves a closer look.",
  "domains.enrichment-labels":
    "The metadata column whose labels are tested, usually cell types or domains. Every pair of labels is scored for how often they sit side by side on the tissue.",
  "domains.enrichment-heatmap":
    "Rows and columns are the labels. Red: the two sit side by side more often than chance; blue: less often; grey: about as expected. The diagonal shows whether a label keeps to itself in patches.",
  "domains.niche-labels":
    "The labels whose local mix defines a niche, usually cell types. Places with the same mix of these labels around them form one niche, whatever the label of the cell itself.",
  "domains.niche-window":
    "How many nearest spots, bins or cells (itself included) make up each window whose mix of labels is compared. A small window describes the immediate surroundings; a large one the wider region.",
  "domains.niche-shares":
    "Which shares from Spot composition describe each spot. Cell shares estimate how many cells of each type the spot holds; RNA shares how much of its RNA each type made. Cell shares are closest to cell2location's abundances.",
  "domains.niche-spot-window":
    "1 groups each spot by its own shares. A larger window first averages the shares over that many nearest spots (itself included), which smooths noisy spots but blurs the edges between niches.",
  "domains.niche-count":
    "How many niches to find. Compare a few values and keep the niches that make sense in the tissue.",
  "domains.niche-heatmap":
    "Rows are niches (with their size), columns are labels. Each value is the mean share of that label in the niche's windows, darker blue for a larger share: read a row as the niche's typical neighbourhood.",

  // Co-occurrence (Domains tab)
  "domains.cooccur-labels":
    "The metadata column whose labels are scored, usually cell types. It starts on the column the plot is coloured by.",
  "domains.cooccur-distances": "How many distances to score, evenly spaced across the range. More give a finer curve.",
  "domains.cooccur-reach":
    "The longest distance to score, in the unit shown; the distances then run evenly up to it. Left empty, squidpy's default range is used, which on a large section starts far above a cell's size: set a value to look at close neighbours.",
  "domains.cooccur-near":
    "The label the curves are measured from. Each line is a label: above 1 at a distance, it is found within that distance of this label more often than among all pairs at that range. The thicker line is the label with itself.",
  "domains.cooccur-section":
    "All sections combined, or one section's own curves. Combined, each section is scored against its own label shares first, so labels that are simply common on the same slides do not look close together.",
  "domains.cooccur-heatmap":
    "The distance at which the labels-by-labels heatmap below is drawn. Short distances show which labels sit close together; longer ones show which share broader regions.",

  // Spot composition
  "spotcomp.reference":
    "An annotated single-cell dataset from your library, whose cell types are mixed to explain each spot. Use one from the same species and tissue: a cell type missing from the reference is given to its nearest type.",
  "spotcomp.celltypes":
    "The reference's metadata column with the cell-type labels to estimate. Labels with too few cells are left out; finer labels give more detail but are harder to tell apart within a spot.",
  "spotcomp.table":
    "One row per reference type. Mean: its average share over the fitted spots; Max: its largest share in one spot; Top in: the spots where it has the largest share. Click a type to colour the tissue by its share on a fixed 0 to 100% scale.",
  "spotcomp.cluster-heatmap":
    "Rows are this dataset's clusters, columns are reference types. Each number is the mean share (%) of that type in the cluster's spots, darker blue for a larger share: it shows which cell types each cluster of spots is made of.",
  "spotcomp.merge-names":
    "Merges labels that share the start of their name, such as T_1 and T_2 into T, to get fewer, broader types. Choose where the name is cut and the separator, then Preview to tick which merges to apply.",

  // Accessibility (ATAC)
  "atac.group-by":
    "Splits this tab's QC plots, WNN weights and chromVAR comparisons by a grouping such as clusters or cell types, and is the first choice for differential peaks. Fragment QC and the coverage tracks have their own Group by.",
  "atac.qc":
    "Per-cell quality from the peak matrix, one histogram per measure. Few peak counts or peaks detected suggest an empty droplet or a broken nucleus. Good cells have a higher fraction of fragments in peaks (FRiP) and at TSSs; a high mitochondrial fraction suggests a damaged cell.",
  "atac.lsi-depth":
    "Each bar is one LSI component, a summary of the peak data like a principal component; its height is how closely it follows each cell's total counts. Blue components build the graph, UMAP and clusters; grey ones are left out. A first component above the dotted line mostly reflects sequencing depth.",
  "atac.activity-vs-rna":
    "Each dot is a cell: the gene's RNA across, its gene activity (open chromatin around the gene) up. Dots along a rising diagonal mean open chromatin and expression go together. Many cells at zero on one axis are normal, as both measures are sparse per cell.",
  "atac.da-groups":
    "The metadata column whose groups are compared: each group's cells against all other cells, giving one table per group, or one group against another.",
  "atac.da-fold-change":
    "How log2FC is computed. Signac: as Signac's FindMarkers on peaks, the log2 ratio of the mean log1p(TF-IDF) of each side, so numbers match its vignettes. Mean TF-IDF + 1: scView's formula before v0.79, about 4 times smaller. Only log2FC changes; the peaks and p values stay the same.",
  "atac.da-compare":
    "Each group vs the rest gives one table per group. One group vs another compares two groups directly and leaves every other cell out, as Signac's FindMarkers with ident.1 and ident.2 (the vignette's CD4 naive vs CD14 monocytes). Fold changes are larger than against the rest.",
  "atac.links-genes":
    "Which genes to link to nearby peaks: the variable genes (quicker), every gene, or a list you type. Each gene is tested against the peaks within 500 kb of its TSS.",
  "atac.links-cells":
    "All cells, or only the groups you tick from a column. Across all cells a link mostly reflects which cell types open the peak and express the gene; within related groups it asks whether the two vary together there.",
  "atac.links-runs": "Earlier link runs kept for this dataset, for example with other genes or cells. Choose one to see its table again.",
  "atac.motif-query":
    "These cut-offs choose the query: peaks more open in the group than in the rest, below this p (or FDR) and open in more than this share of the group's cells. 5% by default, as FindMarkers' min.pct; the Signac vignette's 20% leaves mostly promoters on sparse nuclei, where GC-rich motifs win.",
  "atac.chromvar-heatmap":
    "Rows are the most variable motifs, columns the groups. Red: that motif's peaks are more accessible than expected in the group's cells; blue: less; white: as expected. Click a motif to colour the plot by its activity.",
  "atac.chromvar-vs-rest":
    "Ranks motifs by how much higher their activity is in the chosen group than in all other cells (Difference), with a Wilcoxon test for each. Click a motif to colour the plot by it.",
  "atac.chromvar-variable":
    "Motifs whose activity varies most from cell to cell across the whole dataset, whatever the groups. They point to the transcription factors that may set cell types apart. Click one to colour the plot by it.",
  "atac.fragments-per-cell":
    "The median number of fragments per cell in the fragments file, a measure of sequencing depth. Low values mean little signal per cell.",
  "atac.fragment-group":
    "Adds the TSS enrichment and nucleosome signal per group, and splits the fragment length ladder by group, to see whether one group, such as a cluster, has worse quality than the rest.",
  "atac.coverage-group": "Draws one track per group of cells; with many groups, the largest are drawn.",
  "atac.coverage-tracks":
    "Each filled track is one group's fragment depth along the region, all on the same scale, so heights compare between groups. Grey bars under the tracks mark peaks; lines with arrows mark genes and their strand.",
};
