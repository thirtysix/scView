/**
 * WNN joint RNA + protein neighbours (backend core/wnn.py): the card's text and the
 * cells its three small layouts draw.
 */
import type { WnnPlan, WnnStage } from "@/api/wnn";
import { formatDuration } from "@/lib/hvgSensitivity";

/** The three layouts side by side: obsm key and the title above it. */
export const WNN_LAYOUTS: { key: string; title: string }[] = [
  { key: "X_umap", title: "RNA" },
  { key: "X_umap_protein", title: "Protein" },
  { key: "X_umap_wnn", title: "WNN" },
];

export const WNN_CLUSTERS = "wnn_leiden";
export const RNA_WEIGHT_LABEL = "RNA weight (WNN)";
/** Cells drawn per small layout (the same cells in all three). */
export const THUMB_CELLS = 5000;

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/** "14 proteins (3 isotype controls left out), 13 protein PCs, 50 RNA PCs (PCA); about 1 min." */
export function planText(plan: Pick<WnnPlan, "n_proteins" | "controls" | "d_protein" | "d_rna" | "rep" | "batch_key"> & {
  estimate_seconds?: number;
}): string {
  const controls = plan.controls.length
    ? ` (${plural(plan.controls.length, "isotype control")} left out)`
    : "";
  const rep = plan.rep === "X_pca_harmony" ? "Harmony PCA" : plan.rep === "X_pca" ? "PCA" : plan.rep;
  const batch = plan.batch_key ? `, protein PCs Harmony-corrected on ${plan.batch_key}` : "";
  const time = plan.estimate_seconds ? `; about ${formatDuration(plan.estimate_seconds)}` : "";
  return `${plural(plan.n_proteins, "protein")}${controls}, ${plural(plan.d_protein, "protein PC")}, ` +
    `${plural(plan.d_rna, "RNA PC")} (${rep})${batch}${time}.`;
}

/** "Step 3 of 9: Protein neighbours (45 s so far)". */
export function stageText(stage: WnnStage | null): string {
  if (!stage) return "Starting…";
  const so = stage.elapsed_s >= 5 ? ` (${formatDuration(stage.elapsed_s)} so far)` : "";
  return `Step ${stage.index} of ${stage.total}: ${stage.label}${so}`;
}

/** Up to `max` cell indices spread evenly over `n` cells (the same on every call). */
export function sampleCells(n: number, max = THUMB_CELLS): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = n / max;
  return Array.from({ length: max }, (_, i) => Math.floor(i * step));
}

/** The sampled cells' x, y from a decoded embedding (null when it has no x / y). */
export function thumbOf(
  decoded: Record<string, ArrayLike<number>>,
  cells: number[],
): { cells: number[]; x: number[]; y: number[] } | null {
  const x = decoded["x"];
  const y = decoded["y"];
  if (!x || !y) return null;
  return { cells, x: cells.map((i) => Number(x[i])), y: cells.map((i) => Number(y[i])) };
}

/** Group with the highest and the lowest mean RNA weight, for the caption. */
export function extremes(rows: { group: string; mean_rna_weight: number; n_cells: number }[]):
  { rna: string; protein: string } | null {
  const kept = rows.filter((r) => r.n_cells >= 10);
  if (kept.length < 2) return null;
  const sorted = [...kept].sort((a, b) => b.mean_rna_weight - a.mean_rna_weight);
  return { rna: sorted[0]!.group, protein: sorted[sorted.length - 1]!.group };
}
