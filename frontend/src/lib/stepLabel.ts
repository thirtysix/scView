/** Abbreviations kept in capitals when an unknown step name is spelt out ("qc_metrics" -> "QC metrics"). */
const UPPER: Record<string, string> = {
  qc: "QC", pca: "PCA", umap: "UMAP", tsne: "t-SNE", hvg: "HVG", de: "DE", dpt: "DPT", paga: "PAGA",
  nmf: "NMF", ora: "ORA", gsea: "GSEA", rna: "RNA", adt: "ADT", tcr: "TCR", bcr: "BCR", hto: "HTO",
  atac: "ATAC", wnn: "WNN", lsi: "LSI", cnv: "CNV", dsb: "DSB", vdj: "V(D)J", tf: "TF", gwas: "GWAS",
};

/**
 * Every step kind the backend records (backend core/step_kinds.py RECORDED_STEPS; the
 * test reads that file, so a new kind needs a label here), in words.
 */
export const STEP_LABELS: Record<string, string> = {
  // Bringing data in
  ingest: "Imported",
  integrate: "Datasets merged (joint embedding)",
  // The pipeline
  reset_to_counts: "Reset to counts",
  qc_metrics: "QC metrics",
  doublet_detection: "Doublet detection",
  filtering: "Cell/gene filtering",
  ambient_correction: "Ambient RNA correction",
  normalization: "Normalization",
  log_transform: "Log transform",
  highly_variable_genes: "Highly variable genes",
  scaling: "Scaling",
  pca: "PCA",
  uncorrected_embedding: "Uncorrected UMAP",
  batch_correction: "Batch correction",
  neighbors: "Neighbor graph",
  clustering: "Clustering",
  embeddings: "UMAP",
  marker_genes: "Marker genes",
  enrichment: "Pathway enrichment",
  cell_cycle: "Cell cycle",
  dissociation_stress: "Dissociation stress",
  cell_type_annotation: "Cell-type annotation",
  run_errors: "Run finished with errors",
  // Edits and choices kept
  subcluster: "Subclustered",
  merge_clusters: "Clusters merged (merge check)",
  keep_resolution: "Clustering resolution kept",
  keep_hvg_setting: "Variable-gene setting kept",
  keep_layout: "UMAP layout kept",
  label_cells: "Cells labelled",
  undo_label_cells: "Cell labels undone",
  doublet_threshold: "Doublet threshold set",
  mito_refilter: "Re-filtered by % mito per cell type",
  qc_state_label: "QC states labelled",
  // Annotation
  label_transfer: "Reference mapping",
  annotation_confidence: "Annotation confidence",
  annotation_confidence_unknown: "Annotation confidence (unsure cells marked)",
  annotation_crosscheck: "Annotation cross-check",
  // Other modalities
  protein_dsb: "Protein DSB normalisation",
  wnn: "Joint RNA + protein (WNN)",
  hashing_demux: "Cell hashing demultiplexed",
  hashing_samples: "Samples from hashing",
  vdj_attach: "TCR/BCR contigs attached",
  vdj_clonotypes: "Clonotypes computed",
  vdj_remove: "TCR/BCR data removed",
  crispr_assign: "CRISPR guides assigned",
  mixscape: "Mixscape perturbation classes",
  atac_lsi: "ATAC LSI",
  atac_wnn: "Joint RNA + ATAC (WNN)",
  atac_differential: "Differential peaks",
  atac_fragment_qc: "ATAC fragment QC",
  atac_links: "Peak-gene links",
  atac_motif_enrichment: "Motif enrichment in peaks",
  atac_chromvar: "chromVAR motif deviations",
  velocity: "RNA velocity",
  // Analyses kept with the data
  pseudotime: "Pseudotime",
  paga: "PAGA",
  cnv: "Copy-number inference",
  gene_programs: "Gene programs (NMF)",
  tf_motif_enrichment: "Upstream TF candidates (promoter motifs)",
  gwas_traits: "GWAS trait links",
  bulk_deconvolution: "Bulk deconvolution",
  learned_gate_train: "Learned gate trained",
  learned_gate_apply: "Learned gate applied",
  // Spatial
  spatial_autocorr: "Spatially variable genes (Moran's I)",
  spot_deconvolution: "Spot composition (cell types per spot)",
  spatial_domains: "Spatial domains",
  spatial_niches: "Composition niches",
  nhood_enrichment: "Neighbourhood enrichment",
  co_occurrence: "Co-occurrence",
};

const SKIPPED = "_skipped";

/** A recorded step kind as a label: its own words, "<step> skipped" for a step that did not
 *  apply, else (a kind from a newer version) words spaced with abbreviations in capitals. */
export function stepLabel(step: string): string {
  if (STEP_LABELS[step]) return STEP_LABELS[step];
  if (step.endsWith(SKIPPED) && step.length > SKIPPED.length) {
    return `${stepLabel(step.slice(0, -SKIPPED.length))} skipped`;
  }
  const words = step.split("_").filter(Boolean);
  return words
    .map((w, i) => UPPER[w.toLowerCase()] ?? (i === 0 ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(" ");
}
