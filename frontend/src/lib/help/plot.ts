/** Help for the Unified View shell: plot, toolbar, violin, filters, gates, layouts, clustering checks. */
export const PLOT_HELP: Record<string, string> = {
  // Toolbar
  "plot.embedding":
    "The map of cells: each dot is a cell, and cells with similar expression sit close together. On UMAP and t-SNE, gaps between islands and island sizes say little about how different groups are; use Cluster distances under the plot for that. PCA axes are the main directions of variation.",
  "plot.pc-pair":
    "Which two principal components the plot shows. PC1 and PC2 carry the most variation; a later pair can separate groups the first two overlap. Data Assessment > PCA shows how much each one carries.",
  "plot.colour-by":
    "Colour the cells by a metadata column (cluster, cell type, sample) or a number per cell (counts, % mito). The bracket gives a column's number of groups. In the legend, click a group to select its cells; Ctrl/Cmd-click picks several.",
  "plot.split":
    "One plot per value of a column with 2 to 24 values, such as condition or sample, all sharing the camera and colours. Use it to see whether a group or a gene's pattern appears in every sample or only in some.",
  "plot.grid-pages":
    "How many panels the grid shows at once. Prev and Next page through the rest; every page keeps the same camera and colour scale. Download offers this page, or all panels on one sheet.",
  "plot.layer":
    "Which values the gene colouring and the violins show. Log-normalised values are the usual choice. Raw counts depend on how deeply each cell was sequenced; scaled values are centred on each gene's mean, so below-average cells go negative.",
  "plot.opacity":
    "How see-through each dot is. Lower it where many cells overlap: dense areas then build up colour, and cells hidden underneath show through.",
  "plot.colours":
    "Colour schemes for the plot. Categories: groups such as clusters. Expression: genes and other values from low to high. Signed: scores with a meaningful zero, white at 0 and one colour on each side; the next box sets when it is used.",
  "plot.clip":
    "Cells beyond these percentiles take the end colours, so a few extreme cells do not wash out the rest of the scale. The legend can change it for one view.",
  "plot.max-cells":
    "The most cells drawn at once. A larger dataset shows a fixed random sample of its cells so the plot stays quick; the analyses still use all cells. Raise it on a fast computer.",
  "plot.select":
    "Draw around cells (Lasso), tick groups (By group), set conditions (Filter) or click a saved gate to select cells. The selection is what DE selection vs rest, Label selection, gates and the histogram work on.",

  // Shell: summary, cluster map, distribution
  "plot.summary":
    "What the side panel works on. Grouping is the column the cluster map, the violin and the analysis tabs group cells by; Overlay is the gene, score or protein colouring the main plot, if any.",
  "plot.cluster-map":
    "A small copy of the plot coloured by the grouping, which stays put while the main plot shows a gene or a score. Click a cell or a group name to highlight that group. On a UMAP, zooming or panning either plot moves both.",
  "plot.distribution":
    "The gene or score on the plot, per group. Cells: one violin per group. Per donor: one value per donor, so each person counts once. Histogram: a selection or group against the rest.",
  "plot.violin":
    "Each violin is one group's cells: wider where more of its cells have that value. The small box inside spans the middle half of the cells, with lines at the median and the mean. Tick log scale when a few high values squash the rest.",
  "plot.cluster-distances":
    "How different the groups' average expression is, measured in PCA space, which a UMAP's gaps do not show. Similar groups sit next to each other, with their tree on the right. Read each square's value on the colour bar; click a row to highlight its group.",
  "plot.histogram":
    "Two overlaid distributions of one value per cell: the selection or a group (blue) against the rest or another group (orange). Each side is scaled to its own number of cells, so groups of different sizes compare fairly. The numbers below describe the difference; they are not a test.",

  // Toolbar dialogs
  "plot.compare":
    "The same plot coloured by 2 to 12 columns side by side, one camera for all. Use it to set two clusterings, or a clustering and cell-type labels, against each other: where the same cells form one group in every panel, the columns agree.",
  "plot.select-by-group":
    "Tick groups, then Select to make their cells the selection, as a lasso would. Useful for groups that are scattered on the plot or hard to draw around.",
  "plot.label-selection":
    "Give the selected cells a label in an annotation column of your own, for example to name a cluster. A new column starts with every other cell Unassigned; in an existing one, the other cells keep their labels. Undo label takes back the latest one.",
  "filter.about":
    "Select cells by conditions on gene expression and metadata, such as CD3E above 1 and sample is ctrl. A group nests conditions under the other rule, for (A or B) and C. The count updates as you build it; Select makes the matching cells the selection.",
  "filter.match":
    "ALL: a cell must meet every condition. ANY: meeting one is enough. Gene values use the Layer chosen in the toolbar.",

  // Gates
  "gates.about":
    "A gate is a named set of cells: a saved selection, a Filter (asked again each time), or an AND, OR or NOT of other gates. A gate inside another keeps only cells also in its parent. Click a gate to make it the selection; new gates then go inside it.",
  "gates.learned":
    "A learned gate is a small model (L1 logistic regression, up to 50 genes) of what sets the selected cells apart from other cells of the same samples, so a lasso on control cells does not learn the treatment. Unlike a saved gate, it can be applied to other datasets from Learned gates.",
  "gates.learned-result":
    "How well the gate tells the selected cells from the others, scored on cells each model did not train on. AUROC near 1 and a high F1 mean a clean separation. Genes for and against are what the model relies on: check they make biological sense.",

  // Clustering checks
  "resolutions.about":
    "Clusters the cells at several resolutions without changing the dataset: a higher resolution gives more, smaller clusters. A higher silhouette means better-separated clusters; ARI near 1 means little changed from the row above. Keep the resolution that separates the cell types you expect.",
  "resolutions.tree":
    "How clusters split as the resolution rises: each row is a resolution, and edges follow cells into the clusters of the next row. A tidy tree only splits. A cluster drawing cells from two parents (red outline) means cells move between branches, so that resolution is past what the data support.",
  "merge.about":
    "Checks whether neighbouring clusters really differ, using half of each cell's counts that the clustering did not see. A pair with few distinguishing genes is a merge candidate: possibly one cell type cut in two. Tick the pairs to join and save a merged clustering as a new column.",
  "merge.threshold":
    "A pair of clusters with fewer distinguishing genes than this on the held-out half is a merge candidate. A lower number flags fewer pairs.",
  "hvg.about":
    "Re-clusters the cells under other variable genes, neighbour counts or algorithm, and compares each result with your clustering. A cluster found again under most settings is robust; one that splits, merges or dissolves depends on analysis choices, so name it with care.",
  "hvg.settings":
    "Each setting picks its own variable genes (method and number), or changes the neighbour count or the clustering algorithm. The current setting is run again as a control: its ARI should be near 1.",
  "reliability.about":
    "Colours each cell by the share of its real neighbours (from the graph the clustering used) that the plot puts near it, from 0 to 1. Below 0.5, a cell sits away from most of its neighbours, so do not read its position closely. Compare the scores with Chance and Random placement.",
  "layouts.about":
    "A UMAP is one random layout of many possible. This lays the cells out again with other seeds and neighbour counts and checks which groups touch in each. Trust contacts seen in every layout; a contact seen in only some comes from the layout, not from the cells.",
  "layouts.placement":
    "Placement, as in Plot reliability: the share of a cell's graph neighbours among its nearest 5% of cells on the plot, as a median over the group. A wide range means the group sits near its real neighbours in some layouts and away from them in others.",

  // Markers tab checks (owned by this group)
  "countsplit.show-p":
    "Adds each marker's adjusted p on the held-out half to the marker table, so you can see which markers held up on counts the clustering did not use.",
};
