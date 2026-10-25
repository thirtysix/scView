/**
 * Linked selection (C035): the one cell selection (lasso, legend click, gate,
 * cell filter, a box drawn on a Plotly scatter) shown in every panel that draws
 * cells or groups. Plain typed-array helpers, so the panels' derived traces can
 * be tested without Plotly and stay fast at 100k+ cells: a selection is turned
 * into a byte mask once per Set (cached on the Set itself), never into objects
 * per cell.
 */

/** What the panels call the selection: its row, column or trace name. */
export const SELECTION_LABEL = "Selection";
/** The selection's colour in Plotly panels: the lasso's orange, legible on light and dark. */
export const SELECTION_COLOR = "#ea580c";
export const SELECTION_FILL = "rgba(234,88,12,0.35)";

// One mask / sorted copy per selection Set: a selection is replaced, never
// mutated (selectionStore.setSelection), so the Set's identity is its version.
const masks = new WeakMap<ReadonlySet<number>, Map<number, Uint8Array>>();
const sortedCache = new WeakMap<ReadonlySet<number>, Uint32Array>();

/** 1 for each selected cell of `n`, else 0 (indices outside 0..n-1 are ignored). */
export function selectionMask(selected: ReadonlySet<number>, n: number): Uint8Array {
  let byN = masks.get(selected);
  if (!byN) masks.set(selected, (byN = new Map()));
  let mask = byN.get(n);
  if (!mask) {
    mask = new Uint8Array(n);
    for (const i of selected) if (i >= 0 && i < n) mask[i] = 1;
    byN.set(n, mask);
  }
  return mask;
}

/** The selected cells in ascending order (what an endpoint taking a cell set is sent). */
export function selectedSorted(selected: ReadonlySet<number>): Uint32Array {
  let out = sortedCache.get(selected);
  if (!out) {
    out = Uint32Array.from(selected).sort();
    sortedCache.set(selected, out);
  }
  return out;
}

/** Per group code: how many cells the group has and how many of them are selected.
 *  Negative codes (no value) are not counted. */
export function countSelectedByCode(
  codes: ArrayLike<number>,
  mask: Uint8Array,
  nCodes: number,
): { selected: Uint32Array; total: Uint32Array } {
  const selected = new Uint32Array(nCodes);
  const total = new Uint32Array(nCodes);
  const n = Math.min(codes.length, mask.length);
  for (let i = 0; i < n; i++) {
    const c = codes[i]!;
    if (c < 0 || c >= nCodes) continue;
    total[c]!++;
    if (mask[i]) selected[c]!++;
  }
  return { selected, total };
}

/** Per pair of codes (a facet code and a row code, e.g. condition x donor): the selected cells. */
export function countSelectedByPair(
  rowCodes: ArrayLike<number>,
  facetCodes: ArrayLike<number>,
  mask: Uint8Array,
  nRows: number,
  nFacets: number,
): Uint32Array {
  const out = new Uint32Array(nRows * nFacets);
  const n = Math.min(rowCodes.length, facetCodes.length, mask.length);
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    const r = rowCodes[i]!;
    const f = facetCodes[i]!;
    if (r < 0 || r >= nRows || f < 0 || f >= nFacets) continue;
    out[f * nRows + r]!++;
  }
  return out;
}

/** The values of the selected cells, in cell order (cells past `values` are skipped). */
export function selectedValues(values: ArrayLike<number>, selected: ReadonlySet<number>): Float32Array {
  const mask = selectionMask(selected, values.length);
  let k = 0;
  for (let i = 0; i < mask.length; i++) k += mask[i]!;
  const out = new Float32Array(k);
  k = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) out[k++] = values[i]!;
  return out;
}

/** Every `step`-th selected cell, at most `max` of them (for drawing points). */
export function sampleSelected(selected: ReadonlySet<number>, n: number, max: number): Uint32Array {
  const sorted = selectedSorted(selected);
  let k = 0;
  while (k < sorted.length && sorted[k]! < n) k++;
  // `sorted` is ascending: the first k are the ones within 0..n-1.
  if (k <= max) return sorted.subarray(0, k);
  const out = new Uint32Array(max);
  const step = k / max;
  for (let s = 0; s < max; s++) out[s] = sorted[Math.floor(s * step)]!;
  return out;
}

// ---------------------------------------------------------------------------
// Reverse direction: a box or lasso drawn on a Plotly per-cell scatter
// ---------------------------------------------------------------------------

/** What Plotly's `plotly_selected` event carries, in data units. */
export interface PlotlySelection {
  range?: { x?: number[]; y?: number[] };
  lassoPoints?: { x?: number[]; y?: number[] };
}

function inPolygon(px: number, py: number, xs: readonly number[], ys: readonly number[]): boolean {
  let inside = false;
  for (let i = 0, j = xs.length - 1; i < xs.length; j = i++) {
    const xi = xs[i]!, yi = ys[i]!, xj = xs[j]!, yj = ys[j]!;
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * Every cell whose (x, y) lies inside a Plotly box or lasso selection, among
 * ALL cells, not only the points drawn (a scatter draws a sample on large
 * data). `toAxis` maps a value to the axis' units when the axis is not linear
 * (a pseudo-log axis); null when the event holds no box or lasso.
 */
export function cellsInPlotlySelection(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  sel: PlotlySelection | null | undefined,
  toAxis: { x?: (v: number) => number; y?: (v: number) => number } = {},
): Set<number> | null {
  if (!sel) return null;
  const fx = toAxis.x ?? ((v: number) => v);
  const fy = toAxis.y ?? ((v: number) => v);
  const n = Math.min(x.length, y.length);
  const out = new Set<number>();
  const rx = sel.range?.x;
  const ry = sel.range?.y;
  if (rx && ry && rx.length >= 2 && ry.length >= 2) {
    const [x0, x1] = [Math.min(rx[0]!, rx[1]!), Math.max(rx[0]!, rx[1]!)];
    const [y0, y1] = [Math.min(ry[0]!, ry[1]!), Math.max(ry[0]!, ry[1]!)];
    for (let i = 0; i < n; i++) {
      const vx = fx(x[i]!), vy = fy(y[i]!);
      if (vx >= x0 && vx <= x1 && vy >= y0 && vy <= y1) out.add(i);
    }
    return out;
  }
  const lx = sel.lassoPoints?.x;
  const ly = sel.lassoPoints?.y;
  if (lx && ly && lx.length >= 3 && lx.length === ly.length) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let k = 0; k < lx.length; k++) {
      x0 = Math.min(x0, lx[k]!); x1 = Math.max(x1, lx[k]!);
      y0 = Math.min(y0, ly[k]!); y1 = Math.max(y1, ly[k]!);
    }
    for (let i = 0; i < n; i++) {
      const vx = fx(x[i]!), vy = fy(y[i]!);
      if (vx < x0 || vx > x1 || vy < y0 || vy > y1) continue;
      if (inPolygon(vx, vy, lx, ly)) out.add(i);
    }
    return out;
  }
  return null;
}

/** Counts of a selection in each bin of a histogram with the given edges (values outside fall in the end bins). */
export function binSelected(values: ArrayLike<number>, mask: Uint8Array, edges: readonly number[]): number[] {
  const nb = edges.length - 1;
  const out = new Array<number>(Math.max(nb, 0)).fill(0);
  if (nb <= 0) return out;
  const lo = edges[0]!;
  const hi = edges[nb]!;
  const n = Math.min(values.length, mask.length);
  for (let i = 0; i < n; i++) {
    if (!mask[i]) continue;
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    let b: number;
    if (v <= lo) b = 0;
    else if (v >= hi) b = nb - 1;
    else {
      // Binary search: edges are ascending.
      let a = 0, z = nb;
      while (z - a > 1) {
        const m = (a + z) >> 1;
        if (edges[m]! <= v) a = m;
        else z = m;
      }
      b = a;
    }
    out[b]!++;
  }
  return out;
}
