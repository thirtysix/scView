/**
 * A pipeline run that finished with errors (backend core/step_failures.py): the results of
 * an earlier run it removed as stale, in words, for the run summary and the History.
 */

/** Plain names of what a run removed (keys of obsm, obsp, uns, var, obs, raw). */
const CLEARED_NAMES: Record<string, string> = {
  X_pca: "PCA", PCs: "PCA", pca: "PCA", X_pca_harmony: "Harmony PCs", X_umap: "UMAP",
  X_umap_3d: "UMAP", umap: "UMAP", X_umap_uncorrected: "uncorrected UMAP",
  neighbors: "neighbour graph", connectivities: "neighbour graph", distances: "neighbour graph",
  scview_active_clustering: "active clustering", highly_variable: "variable genes", hvg: "variable genes",
  scview_scaling: "scaling", rank_genes_groups: "marker genes", S_score: "cell-cycle scores",
  G2M_score: "cell-cycle scores", phase: "cell-cycle scores",
  dissociation_stress_score: "dissociation-stress score", raw: "log-normalised .raw",
};

/** "PCA, UMAP, neighbour graph and marker genes", or null when nothing was removed. */
export function clearedText(cleared: Record<string, string[]> | undefined | null): string | null {
  const keys = Object.values(cleared ?? {}).flat();
  const names = [
    ...new Set(
      keys.map((k) => CLEARED_NAMES[k] ?? (k.startsWith("rank_genes_groups__") ? "marker genes"
        : k.startsWith("enrichment__") || k.startsWith("ora_meta__") ? "enrichment" : null)),
    ),
  ].filter((n): n is string => !!n);
  if (!names.length) return null;
  return names.length === 1 ? names[0]! : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
