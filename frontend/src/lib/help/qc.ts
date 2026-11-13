/** Help for the QC cards outside Unified View (Data Assessment). Keys: `<card>.<thing>`. */
export const QC_HELP: Record<string, string> = {
  "evidence.section":
    "Why each cluster was labelled or flagged, from the numbers scView's own rules read. Plain evidence, never a probability: a probability model was tested on benchmark clusters and did worse than these rules.",
  "evidence.markers":
    "How well the cluster's top markers match the best cell type in the marker databases, against random genes. 1 in 10 random lists of the same size reach the p shown. Markers that do no better are labelled Unidentified by AI annotation.",
  "evidence.qc":
    "Median genes and mitochondrial share per cell against the typical cluster (the median across clusters), with the cluster QC flags that fired. Few genes with high mito, or very high mito, marks damaged cells, which AI annotation labels Low-quality cells.",
  "evidence.doublet":
    "Most of the cluster's top markers belong to two other clusters, and its cells carry more RNA than either, as doublets of the two would. It can also be a real state between them: check both clusters' markers in it.",
  "evidence.content":
    "Most of the cluster's top markers belong to one other cluster, yet its cells carry far less RNA: the same cells damaged or losing RNA, or droplets of ambient RNA, split off by sequencing depth. A caution, not a label: check its genes per cell and its markers against that cluster.",
  "evidence.makeup":
    "What leads the cluster's marker list. Sex genes (XIST, Y genes) first: cells of one sex or donor split from their type. Mostly small RNAs, pseudogenes or unnamed genes: a technical cluster more likely than a cell type. A caution, not a label.",
  "clusterqc.mito_cells":
    "Share of the cluster's cells with any mitochondrial read. In nuclei the median cell has none, so a median of 0% hides contamination; a cluster where far more nuclei carry mito reads has cytoplasm or whole cells in it. Hover a value for its mean share of reads.",
  "evidence.types":
    "Marker database types that match the cluster's markers, as shares of the evidence: how the database scores split between types, summed up the Cell Ontology. Not the chance that the cells are that type. Shown for named groupings only.",
  "evidence.decision":
    "The label scView gave the cluster itself, without asking the AI, or would give it in an AI annotation. Fixed rules on the numbers above, not a model.",
};
