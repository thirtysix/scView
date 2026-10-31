/**
 * 10x Multiome ATAC (backend core/atac.py): the Accessibility tab's text and the small
 * calculations behind its plots.
 */
import type { AtacLsi, AtacPlan, AtacStage, AtacWnnPlan } from "@/api/atac";
import { formatDuration } from "@/lib/hvgSensitivity";

export const ATAC_CLUSTERS = "atac_leiden";
export const WNN_ATAC_CLUSTERS = "wnn_atac_leiden";
export const WNN_ATAC_WEIGHT = "wnn_atac_rna_weight";
export const ATAC_UMAP = "X_umap_atac";
export const WNN_ATAC_UMAP = "X_umap_wnn_atac";
/** Cells drawn per QC histogram and violin (the plots slow down past this). */
export const QC_MAX_CELLS = 20_000;

export const activityLabel = (gene: string) => `${gene} gene activity`;

const gb = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(bytes >= 10 * 1024 ** 3 ? 0 : 1)} GB`;

/** "11,898 cells, 143,887 peaks, 104M values; about 4 min and 1.2 GB beyond the open dataset." */
export function planText(p: AtacPlan): string {
  return `${p.n_cells.toLocaleString()} cells, ${p.n_peaks.toLocaleString()} peaks, ` +
    `${Math.round(p.nnz / 1e6).toLocaleString()}M values read from disk a block at a time; ` +
    `about ${formatDuration(p.estimate_seconds)}, ${gb(p.estimate_bytes)} beyond the open dataset and ` +
    `${gb(p.estimate_disk_bytes)} of disk.`;
}

export function wnnPlanText(p: AtacWnnPlan): string {
  const rep = p.rep === "X_pca_harmony" ? "Harmony PCA" : p.rep === "X_pca" ? "PCA" : p.rep;
  const first = p.lsi_dims[0];
  const last = p.lsi_dims[p.lsi_dims.length - 1];
  return `${p.d_rna} RNA PCs (${rep}) with LSI ${first}:${last}; about ${formatDuration(p.estimate_seconds)} and ` +
    `${gb(p.estimate_bytes)} beyond the open dataset.`;
}

/** "Step 3 of 6: ATAC UMAP (45 s so far)". */
export function stageText(stage: AtacStage | null): string {
  if (!stage) return "Starting…";
  const so = stage.elapsed_s >= 5 ? ` (${formatDuration(stage.elapsed_s)} so far)` : "";
  return `Step ${stage.index} of ${stage.total}: ${stage.label}${so}`;
}

/** The depth-correlation choice in one sentence. */
export function depthChoiceText(lsi: Pick<AtacLsi, "depth_r" | "depth_limit" | "dropped_first" | "dims_used">): string {
  const r = (lsi.depth_r[0] ?? 0).toFixed(2);
  const first = lsi.dims_used[0];
  const last = lsi.dims_used[lsi.dims_used.length - 1];
  const limit = lsi.depth_limit.toFixed(2);
  return lsi.dropped_first
    ? `The first LSI component follows sequencing depth (r = ${r}, above the ${limit} limit), so it is left out: the graph uses components ${first}:${last}.`
    : `The first LSI component does not follow sequencing depth closely (r = ${r}, limit ${limit}), so it is kept: the graph uses components ${first}:${last}.`;
}

/** Every n-th cell's value, so a plot draws at most `max`; nulls (no metric) dropped. */
export function sampled<T>(values: readonly T[], max: number): T[] {
  if (values.length <= max) return values.slice();
  const step = values.length / max;
  const out: T[] = [];
  for (let i = 0; i < max; i++) {
    const v = values[Math.floor(i * step)];
    if (v !== undefined) out.push(v);
  }
  return out;
}

/** A column's values split by group, at most `max` cells overall (the same cells in each column). */
export function splitByGroup(
  values: readonly (number | null)[],
  groups: readonly (string | null)[],
  order: readonly string[],
  max: number,
): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  for (const g of order) out[g] = [];
  const step = Math.max(1, values.length / max);
  for (let f = 0; f < values.length; f += step) {
    const i = Math.floor(f);
    const v = values[i];
    const g = groups[i];
    if (v !== null && v !== undefined && g !== null && g !== undefined) out[g]?.push(v);
  }
  return out;
}

/** A p-value for a table cell: "2e-21", "0.034", "< 1e-300". */
export function formatP(p: number | null | undefined): string {
  if (p === null || p === undefined || !isFinite(p)) return "–";
  if (p === 0 || p < 1e-300) return "< 1e-300";
  return p < 0.001 ? p.toExponential(0) : p.toPrecision(2);
}

/** Distance to the nearest TSS: "at TSS", "+1.2 kb" (downstream), "-340 b" (upstream). */
export function formatDistance(d: number | null | undefined): string {
  if (d === null || d === undefined || !isFinite(d)) return "–";
  if (d === 0) return "at TSS";
  const sign = d > 0 ? "+" : "-";
  const a = Math.abs(d);
  return a >= 1000 ? `${sign}${(a / 1000).toFixed(1)} kb` : `${sign}${a} b`;
}
