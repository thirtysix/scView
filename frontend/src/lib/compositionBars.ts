// The Observations composition bars: bar order, counts vs shares, one bar per
// (facet, row) when split by a third column, and the 10 x 10 waffle of one bar.
import { sortByCategoryOrder } from "@/lib/categoryColors";

/** `/metadata/crosstab` response: counts[row][col], and per facet with `facet=`. */
export interface Crosstab {
  row_values: string[];
  col_values: string[];
  counts: number[][];
  facet_values?: string[];
  facet_counts?: number[][][];
}

/** How the bars are ordered: the column's stored category order, by name, by
 *  cell count, or by the share of one stacked category (e.g. % CD14 Mono). */
export type BarSort = "category" | "name" | "size" | "share";

/** One bar: its label (facet value + row value when split) and the counts of each stacked column. */
export interface Bar {
  row: string;
  facet: string | null;
  counts: number[];
  total: number;
}

const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

/** The bars of a crosstab: one per row, or one per (facet, row) with cells when `split`. */
export function crosstabBars(ct: Crosstab, split: boolean): Bar[] {
  const bar = (row: string, facet: string | null, counts: number[]): Bar => ({
    row,
    facet,
    counts,
    total: counts.reduce((a, b) => a + b, 0),
  });
  if (split && ct.facet_values && ct.facet_counts) {
    const out: Bar[] = [];
    ct.facet_values.forEach((f, fi) => {
      ct.row_values.forEach((r, ri) => {
        const counts = ct.col_values.map((_, ci) => ct.facet_counts![fi]?.[ri]?.[ci] ?? 0);
        const b = bar(r, f, counts);
        if (b.total > 0) out.push(b); // a donor with no cells in this condition has no bar
      });
    });
    return out;
  }
  return ct.row_values.map((r, ri) => bar(r, null, ct.col_values.map((_, ci) => ct.counts[ri]?.[ci] ?? 0)));
}

/**
 * The bars in display order. `share` sorts by the share of the stacked column
 * `shareOf` (largest first); `size` by cell count (largest first). Split bars
 * stay grouped by facet, in the facets' stored order, and every facet lists
 * the rows in one order, sorted on the rows' totals over all facets: a donor
 * keeps its place across conditions (Plotly's grouped axis also orders the
 * inner labels once, by first appearance, so per-facet orders would not hold).
 */
export function sortBars(
  bars: readonly Bar[],
  sort: BarSort,
  opts: { rowOrder: readonly string[]; colValues: readonly string[]; shareOf?: string; facetOrder?: readonly string[] },
): Bar[] {
  const rowPos = new Map(sortByCategoryOrder(Array.from(new Set(bars.map((b) => b.row))), opts.rowOrder).map((r, i) => [r, i]));
  const si = opts.shareOf ? opts.colValues.indexOf(opts.shareOf) : -1;
  const share = (b: Bar) => (si >= 0 && b.total > 0 ? (b.counts[si] ?? 0) / b.total : 0);
  const compare = (a: Bar, b: Bar): number => {
    switch (sort) {
      case "name":
        return collator.compare(a.row, b.row);
      case "size":
        return b.total - a.total;
      case "share":
        return share(b) - share(a) || b.total - a.total;
      default:
        return 0;
    }
  };
  const byRow = (a: Bar, b: Bar) => compare(a, b) || (rowPos.get(a.row) ?? 0) - (rowPos.get(b.row) ?? 0);
  if (!bars.some((b) => b.facet != null)) return [...bars].sort(byRow);

  // One row order for every facet, from the rows summed over the facets.
  const sums = new Map<string, Bar>();
  for (const b of bars) {
    const s = sums.get(b.row) ?? { row: b.row, facet: null, counts: b.counts.map(() => 0), total: 0 };
    b.counts.forEach((n, i) => (s.counts[i] = (s.counts[i] ?? 0) + n));
    s.total += b.total;
    sums.set(b.row, s);
  }
  const rank = new Map([...sums.values()].sort(byRow).map((b, i) => [b.row, i]));
  const facetPos = new Map((opts.facetOrder ?? []).map((f, i) => [f, i]));
  return [...bars].sort(
    (a, b) =>
      (facetPos.get(a.facet ?? "") ?? 0) - (facetPos.get(b.facet ?? "") ?? 0) ||
      (rank.get(a.row) ?? 0) - (rank.get(b.row) ?? 0),
  );
}

/** A bar's stacked value for column `ci`: the count, or its percent of the bar. */
export function barValue(b: Bar, ci: number, percent: boolean): number {
  const n = b.counts[ci] ?? 0;
  return percent ? (b.total > 0 ? (n / b.total) * 100 : 0) : n;
}

/** A bar's label for hover, CSV and the waffle picker. */
export function barLabel(b: Bar): string {
  return b.facet == null ? b.row : `${b.row} (${b.facet})`;
}

/**
 * Squares out of `n` per category, by the largest remainder so they add up to
 * `n` exactly. A category with any cells but under half a square still rounds
 * to 0: the waffle shows shares in steps of 1/n.
 */
export function waffleSquares(counts: readonly number[], n = 100): number[] {
  const total = counts.reduce((a, b) => a + Math.max(0, b), 0);
  if (total <= 0) return counts.map(() => 0);
  const exact = counts.map((c) => (Math.max(0, c) / total) * n);
  const squares = exact.map(Math.floor);
  let left = n - squares.reduce((a, b) => a + b, 0);
  const byRemainder = exact
    .map((e, i) => ({ i, r: e - Math.floor(e) }))
    .sort((a, b) => b.r - a.r || a.i - b.i);
  for (const { i } of byRemainder) {
    if (left <= 0) break;
    squares[i]! += 1;
    left -= 1;
  }
  return squares;
}
