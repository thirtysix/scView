/**
 * C035: what each Plotly panel adds for the current selection, as plain data
 * (tested without Plotly). The panels: the violin and the stacked violin get a
 * "Selection" group; the marker dot plot / heatmap a "Selection" row; the
 * composition bars a marker per bar for the selected share; the QC histograms
 * the selected cells' counts in the same bins; the per-cell scatters the
 * selected cells drawn on top.
 */
import { SELECTION_LABEL, sampleSelected } from "@/lib/linkedSelection";

/** The selected cells' (x, y), an even sample of at most `max`, for a trace drawn over the rest. */
export function selectedPoints(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  selected: ReadonlySet<number>,
  max: number,
): { x: number[]; y: number[]; n: number } {
  const n = Math.min(x.length, y.length);
  const cells = sampleSelected(selected, n, max);
  const px = new Array<number>(cells.length);
  const py = new Array<number>(cells.length);
  for (let k = 0; k < cells.length; k++) {
    px[k] = x[cells[k]!]!;
    py[k] = y[cells[k]!]!;
  }
  let total = 0;
  for (const i of selected) if (i >= 0 && i < n) total++;
  return { x: px, y: py, n: total };
}

/** "Selection (1,234 cells)": the name of the extra group / row (never a real group's name). */
export function selectionName(n: number): string {
  return `${SELECTION_LABEL} (${n.toLocaleString()} cells)`;
}

/** The selection's row of a marker plot in the plot's gene order, when the
 *  server's genes are the plot's (same genes, same order); else null. */
export function selectionRow(
  sel: { genes: string[]; mean: number[]; mean_expressing: number[]; fraction: number[] } | null,
  dataGenes: readonly string[],
  colIdx: readonly number[],
): { mean: number[]; meanExpressing: number[]; fraction: number[] } | null {
  if (!sel || sel.genes.length !== dataGenes.length || sel.genes.some((g, i) => g !== dataGenes[i])) return null;
  return {
    mean: colIdx.map((c) => sel.mean[c] ?? 0),
    meanExpressing: colIdx.map((c) => sel.mean_expressing[c] ?? 0),
    fraction: colIdx.map((c) => sel.fraction[c] ?? 0),
  };
}

/** One marker per composition bar: the share (0-100) or number of its cells that are selected. */
export function compositionSelection(
  bars: readonly { row: string; facet: string | null; total: number }[],
  selectedOf: (row: string, facet: string | null) => number,
  percent: boolean,
): { y: number[]; counts: number[]; shares: number[] } {
  const counts = bars.map((b) => selectedOf(b.row, b.facet));
  const shares = bars.map((b, i) => (b.total > 0 ? (100 * counts[i]!) / b.total : 0));
  return { y: percent ? shares : counts, counts, shares };
}
