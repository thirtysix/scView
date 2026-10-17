/** Helpers for the annotation confidence card (backend core/annotation_confidence.py). */
import type { ConfidenceLabelRow, ConfidenceResult } from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";
import { rankGroupingColumns } from "@/lib/obsColumns";
import { pct } from "@/lib/qcThresholds";

/** Every column the check writes starts with this. */
export const CONFIDENCE_PREFIX = "scview_conf_";

/** What the check measures (the server sends the same sentence as `note`). */
export const CONFIDENCE_NOTE =
  "This measures how well each cell's label agrees with the expression neighbourhood " +
  "of the cells carrying that label, not whether the label is right.";

export function isConfidenceOutput(name: string): boolean {
  return name.startsWith(CONFIDENCE_PREFIX);
}

type DatasetColumns = Pick<DatasetInfo, "obs_columns" | "n_cells" | "active_clustering" | "user_annotation_columns">;

/**
 * Label columns the check can take, best first (hand labels and cell types,
 * then clusterings): categorical groupings with 2–100 values that are not the
 * check's own outputs. Boolean flags are left out.
 */
export function confidenceColumns(dataset: DatasetColumns | null): { name: string; kind: string }[] {
  if (!dataset) return [];
  const bools = new Set((dataset.obs_columns ?? []).filter((c) => c.dtype === "bool").map((c) => c.name));
  return rankGroupingColumns(dataset.obs_columns ?? [], {
    nCells: dataset.n_cells,
    activeClustering: dataset.active_clustering,
    userColumns: dataset.user_annotation_columns,
  }).filter((c) => !isConfidenceOutput(c.name) && !bools.has(c.name));
}

/** The column the card opens on: the best cell-type or hand-label column, else the best grouping. */
export function defaultConfidenceColumn(dataset: DatasetColumns | null): string {
  const cols = confidenceColumns(dataset);
  return (cols.find((c) => c.kind === "cell type" || c.kind === "user labels") ?? cols[0])?.name ?? "";
}

/** Share of a label's cells whose own label was rejected (unsure or doesn't fit). */
export function rejectedShare(row: ConfidenceLabelRow): number {
  return row.unsure + row.not_fit;
}

/** Labels that fit worst first: by rejected share, then ambiguous share. */
export function sortLabelRows(rows: readonly ConfidenceLabelRow[]): ConfidenceLabelRow[] {
  return [...rows].sort(
    (a, b) => rejectedShare(b) - rejectedShare(a) || b.ambiguous - a.ambiguous || b.n_cells - a.n_cells,
  );
}

/** The held-out coverage sentence: the number the guarantee is about. */
export function coverageText(r: Pick<ConfidenceResult, "coverage">): string {
  const c = r.coverage;
  return (
    `${pct(c.held_out)} of ${c.n_held_out.toLocaleString()} held-out cells kept their label in their set ` +
    `(target ${pct(c.target)}; sampling spread ±${pct(2 * c.sd)}).`
  );
}

/** A smaller alpha that would let sets grow, from the classifier's error rate. */
export function smallerAlpha(r: Pick<ConfidenceResult, "accuracy">): number {
  return Math.max(0.01, Math.floor((1 - r.accuracy) * 70) / 100);
}

/**
 * Why nothing is ambiguous, when that is so: the classifier is right more
 * often than 1 − alpha, so LAC sets shrink (in-between cells come out unsure)
 * and a smaller alpha shows the labels they are torn between. Null otherwise.
 */
export function shrinkHint(r: Pick<ConfidenceResult, "alpha" | "accuracy" | "counts">): string | null {
  if (r.counts.ambiguous > 0 || r.counts.unsure === 0) return null;
  const smaller = smallerAlpha(r);
  const tail =
    smaller < r.alpha
      ? ` Try α = ${smaller} to see which labels the unsure cells are torn between.`
      : "";
  return (
    `No cell is ambiguous at α = ${r.alpha}: the classifier picks the cell's own label for ` +
    `${pct(r.accuracy)} of cells, more than 1 − α, so sets shrink instead of growing and ` +
    `in-between cells come out unsure.${tail}`
  );
}

/** "CD4 Naive T 13.6%, CD8 T 4.0%" */
export function confusedWithText(row: ConfidenceLabelRow): string {
  return row.confused_with.map((c) => `${c.label} ${pct(c.share)}`).join(", ");
}
