/**
 * The cluster-distance view (C074): what UMAP and t-SNE plots don't show.
 * Their gaps and blob sizes say little about how different clusters are, so
 * those plots carry a caveat linking to a heatmap of the distances between
 * group centroids in the PCs the neighbour graph was built on
 * (GET /clustering/distances).
 */
import type { GroupDistances } from "@/api/clustering";

/** "UMAP" or "t-SNE" for an embedding whose cluster gaps and sizes are not
 *  distances; null for PCA and any other. */
export function nonlinearEmbedding(key: string | null | undefined): "UMAP" | "t-SNE" | null {
  const k = (key ?? "").toLowerCase();
  if (k.includes("umap")) return "UMAP";
  if (/t-?sne/.test(k)) return "t-SNE";
  return null;
}

/** The heatmap of `matrix` (the distances, or another group x group matrix
 *  such as the correlation) in the clustering's order, rows top to bottom. The
 *  diagonal is null: a group's 0 (or 1) to itself would stretch the colour scale. */
export function orderedHeatmap(
  d: GroupDistances,
  matrix: number[][] = d.distances,
): { labels: string[]; sizes: number[]; z: (number | null)[][] } {
  const order = d.order.length === d.groups.length ? d.order : d.groups.map((_, i) => i);
  return {
    labels: order.map((i) => d.groups[i]!),
    sizes: order.map((i) => d.sizes[i]!),
    z: order.map((i) => order.map((j) => (i === j ? null : matrix[i]![j]!))),
  };
}

/** The `k` groups nearest to `group`, nearest first (empty for a group not in `d`). */
export function nearestGroups(d: GroupDistances, group: string, k = 3): { group: string; distance: number }[] {
  const i = d.groups.indexOf(group);
  if (i < 0) return [];
  return d.groups
    .map((g, j) => ({ group: g, distance: d.distances[i]![j]! }))
    .filter((_, j) => j !== i)
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
}

/** The space the distances are measured in, for the caption. */
export function spaceLabel(d: GroupDistances): string {
  const what =
    d.representation === "X_pca_harmony"
      ? "Harmony-corrected PCs"
      : d.representation === "X_pca"
        ? "PCs"
        : `${d.representation} components`;
  const space = `${d.n_pcs} ${what}`;
  if (d.rep_source === "graph") return `${space}, the space the neighbour graph was built in`;
  if (d.rep_source === "missing") return `${space} (the neighbour graph's own space is not in the file)`;
  return `${space} (no neighbour graph recorded)`;
}
