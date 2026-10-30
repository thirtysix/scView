/**
 * Spatial domains and niches (backend core/spatial_domains.py): the tab's words and the
 * matrices its heatmaps draw.
 */
import type { DomainPlan, RunProgress, RunStage } from "@/api/spatialDomains";
import { formatDuration } from "@/lib/hvgSensitivity";

/** "Harmony-corrected PCs" / "PCA": the embedding a run uses, as the tab says it. */
export function repName(rep: string): string {
  return rep === "X_pca_harmony" ? "the Harmony-corrected PCs" : rep === "X_pca" ? "the PCA" : rep;
}

/** " and 1.2 GB": what a run holds, said once it is worth saying. */
function memory(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? ` and ${gb.toFixed(1)} GB` : "";
}

/** "First 10 components of the PCA, 3 hops; about 20 s at k = 7." */
export function planText(plan: DomainPlan): string {
  return (
    `First ${plan.n_pcs} components of ${repName(plan.rep)}, ${plan.n_hops} ${plan.n_hops === 1 ? "hop" : "hops"}; ` +
    `about ${formatDuration(plan.estimate_seconds)}${memory(plan.estimate_bytes)} at k = ${plan.k ?? "?"}.`
  );
}

/** "Suggest k tries k = 2 to 15, 5 fits each; about 3 min." */
export function scanText(plan: DomainPlan): string {
  return (
    `Suggest k tries k = ${plan.k_min} to ${plan.k_max}, ${plan.repeats} fits each` +
    `${plan.n_fit < plan.n_obs ? ` on a sample of ${plan.n_fit.toLocaleString()}` : ""}; ` +
    `about ${formatDuration(plan.estimate_scan_seconds)}.`
  );
}

/** "Step 2 of 4: Neighbour features", with the fits or permutations done when there are any. */
export function runText(stage: RunStage | null, progress: RunProgress | null): string {
  if (!stage) return "Starting…";
  const count = progress ? `, ${progress.done.toLocaleString()} of ${progress.total.toLocaleString()}` : "";
  const elapsed = progress?.elapsed_s ?? stage.elapsed_s;
  const so = elapsed >= 5 ? ` (${formatDuration(elapsed)} so far)` : "";
  return `Step ${stage.index} of ${stage.total}: ${stage.label}${count}${so}`;
}

/** The share of a run done, for its bar. */
export function runShare(stage: RunStage | null, progress: RunProgress | null): number {
  if (!stage) return 0;
  const inside = progress && progress.total > 0 ? progress.done / progress.total : 0;
  return Math.min(1, (stage.index - 1 + inside) / stage.total);
}

/** Largest |z| of an enrichment matrix, for a colour scale symmetric about 0. */
export function zRange(z: (number | null)[][]): number {
  let top = 0;
  for (const row of z) {
    for (const v of row) if (v != null && Math.abs(v) > top) top = Math.abs(v);
  }
  return top > 0 ? top : 1;
}

/**
 * The pair of different labels most often and least often side by side: the caption's
 * reading of the heatmap. Null beside either when no pair is beyond |z| 2.
 */
export function strongestPairs(
  labels: string[],
  z: (number | null)[][],
): { high: [string, string, number] | null; low: [string, string, number] | null } {
  let high: [string, string, number] | null = null;
  let low: [string, string, number] | null = null;
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const v = z[i]?.[j];
      if (v == null) continue;
      if (v > 2 && (!high || v > high[2])) high = [labels[i]!, labels[j]!, v];
      if (v < -2 && (!low || v < low[2])) low = [labels[i]!, labels[j]!, v];
    }
  }
  return { high, low };
}

/** Label columns a run can use: categorical, 2 to 100 labels. */
export function labelColumns(
  columns: readonly { name: string; dtype: string; n_unique: number }[],
): { name: string; n_unique: number }[] {
  return columns.filter(
    (c) => (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") && c.n_unique >= 2 && c.n_unique <= 100,
  );
}
