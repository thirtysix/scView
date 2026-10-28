/** Abbreviations kept in capitals when a step name is shown ("qc_metrics" -> "QC metrics"). */
const UPPER: Record<string, string> = {
  qc: "QC", pca: "PCA", umap: "UMAP", tsne: "t-SNE", hvg: "HVG", de: "DE", dpt: "DPT", paga: "PAGA",
  nmf: "NMF", ora: "ORA", gsea: "GSEA", rna: "RNA", adt: "ADT", tcr: "TCR", bcr: "BCR", hto: "HTO",
};

/** Step ids whose spaced-out form reads badly. */
const NAMED: Record<string, string> = {
  spatial_autocorr: "Spatially variable genes (Moran's I)",
  doublet_detection_skipped: "Doublet detection skipped",
  gwas_traits: "GWAS trait links",
};

/** A pipeline step id as a label: words spaced, first letter capital, abbreviations in capitals. */
export function stepLabel(step: string): string {
  if (NAMED[step]) return NAMED[step];
  const words = step.split("_").filter(Boolean);
  return words
    .map((w, i) => UPPER[w.toLowerCase()] ?? (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}
