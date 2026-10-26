/** Abbreviations kept in capitals when a step name is shown ("qc_metrics" -> "QC metrics"). */
const UPPER: Record<string, string> = {
  qc: "QC", pca: "PCA", umap: "UMAP", tsne: "t-SNE", hvg: "HVG", de: "DE", dpt: "DPT", paga: "PAGA",
  nmf: "NMF", ora: "ORA", gsea: "GSEA", rna: "RNA", adt: "ADT", tcr: "TCR", bcr: "BCR", hto: "HTO",
};

/** A pipeline step id as a label: words spaced, first letter capital, abbreviations in capitals. */
export function stepLabel(step: string): string {
  const words = step.split("_").filter(Boolean);
  return words
    .map((w, i) => UPPER[w.toLowerCase()] ?? (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}
