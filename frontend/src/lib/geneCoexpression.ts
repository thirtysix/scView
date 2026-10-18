/**
 * Several genes at once: feature-plot draw order and colour scales (C017), and
 * two genes against each other (C068): correlations, quadrant counts, the
 * co-expressing cells, and the Seurat-style two-gene blend colours.
 */
import { spearman } from "@/lib/protein";

export { spearman };

type RGB = [number, number, number];

/** Most genes the feature grid and the stacked violin take at once. */
export const MAX_GRID_GENES = 9;
/** Fewest genes worth a grid (one gene is the single plot). */
export const MIN_GRID_GENES = 2;

/** Pearson's r over the cells `subset` (all cells when null); null without variation. */
export function pearson(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  subset: ArrayLike<number> | null = null,
): number | null {
  const n = subset ? subset.length : Math.min(x.length, y.length);
  if (n < 3) return null;
  let sx = 0;
  let sy = 0;
  for (let k = 0; k < n; k++) {
    const i = subset ? subset[k]! : k;
    sx += x[i]!;
    sy += y[i]!;
  }
  const mx = sx / n;
  const my = sy / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let k = 0; k < n; k++) {
    const i = subset ? subset[k]! : k;
    const a = x[i]! - mx;
    const b = y[i]! - my;
    sxy += a * b;
    sxx += a * a;
    syy += b * b;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
}

/** Spearman's rho over the cells `subset` (all cells when null). */
export function spearmanOf(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  subset: ArrayLike<number> | null = null,
): number | null {
  if (!subset) return spearman(x, y);
  const xs = new Float32Array(subset.length);
  const ys = new Float32Array(subset.length);
  for (let k = 0; k < subset.length; k++) {
    xs[k] = x[subset[k]!]!;
    ys[k] = y[subset[k]!]!;
  }
  return spearman(xs, ys);
}

/** Cells in each quadrant of gene A vs gene B; a gene is "positive" when its value is above 0. */
export interface QuadrantCounts {
  both: number;
  aOnly: number;
  bOnly: number;
  neither: number;
}

export type Quadrant = keyof QuadrantCounts;

function quadrantOf(a: number, b: number): Quadrant {
  if (a > 0) return b > 0 ? "both" : "aOnly";
  return b > 0 ? "bOnly" : "neither";
}

/** How many cells (of `subset`, or all) express both genes, one, or neither. */
export function quadrantCounts(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  subset: ArrayLike<number> | null = null,
): QuadrantCounts {
  const out: QuadrantCounts = { both: 0, aOnly: 0, bOnly: 0, neither: 0 };
  const n = subset ? subset.length : Math.min(a.length, b.length);
  for (let k = 0; k < n; k++) {
    const i = subset ? subset[k]! : k;
    out[quadrantOf(a[i]!, b[i]!)]++;
  }
  return out;
}

/** The cells of one quadrant, as a selection (the co-expressing cells for "both"). */
export function quadrantCells(a: ArrayLike<number>, b: ArrayLike<number>, quadrant: Quadrant = "both"): Set<number> {
  const out = new Set<number>();
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (quadrantOf(a[i]!, b[i]!) === quadrant) out.add(i);
  return out;
}

/** The cells whose code in `codes` is `code`, as indices. */
export function cellsWithCode(codes: ArrayLike<number>, code: number): Uint32Array {
  let n = 0;
  for (let i = 0; i < codes.length; i++) if (codes[i] === code) n++;
  const out = new Uint32Array(n);
  let k = 0;
  for (let i = 0; i < codes.length; i++) if (codes[i] === code) out[k++] = i;
  return out;
}

/**
 * The `q` quantile (0..1) of `values`, linear between order statistics (as
 * numpy's default). Values not finite are skipped; NaN when none are left.
 */
export function quantile(values: ArrayLike<number>, q: number): number {
  const v: number[] = [];
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i]!)) v.push(values[i]!);
  if (v.length === 0) return NaN;
  v.sort((x, y) => x - y);
  const pos = Math.max(0, Math.min(1, q)) * (v.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, v.length - 1);
  return v[lo]! + (v[hi]! - v[lo]!) * (pos - lo);
}

/**
 * The 0..1 scale of one gene for the blend: 0 at no expression (the gene's
 * minimum when a layer has negative values), 1 at the `clip` quantile of the
 * expressing cells (above 0), so a handful of extreme cells do not wash out the
 * rest. `clip` 1 scales to the maximum.
 */
export function blendScale(values: ArrayLike<number>, clip = 0.99): { lo: number; hi: number } {
  let min = Infinity;
  let max = -Infinity;
  const positive: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    if (v > 0) positive.push(v);
  }
  if (!(min <= max)) return { lo: 0, hi: 1 };
  const lo = Math.min(0, min);
  let hi = clip >= 1 || positive.length === 0 ? max : quantile(positive, clip);
  if (!(hi > lo)) hi = max > lo ? max : lo + 1;
  return { lo, hi };
}

/** Where `v` sits on a blend scale, clamped to 0..1. */
export function scaled(v: number, s: { lo: number; hi: number }): number {
  if (!Number.isFinite(v)) return 0;
  const t = (v - s.lo) / (s.hi - s.lo);
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** The four corners of the blend map (Seurat FeaturePlot blend: red, green, yellow, light grey). */
export const BLEND_CORNERS: { neither: RGB; a: RGB; b: RGB; both: RGB } = {
  neither: [211, 211, 211],
  a: [255, 0, 0],
  b: [0, 255, 0],
  both: [255, 255, 0],
};

/** The colour of a cell at `ta` of gene A's scale and `tb` of gene B's (each 0..1): bilinear between the corners. */
export function blendColor(ta: number, tb: number): RGB {
  const a = Math.max(0, Math.min(1, ta));
  const b = Math.max(0, Math.min(1, tb));
  const w0 = (1 - a) * (1 - b);
  const wa = a * (1 - b);
  const wb = (1 - a) * b;
  const wab = a * b;
  const { neither, a: ca, b: cb, both } = BLEND_CORNERS;
  return [0, 1, 2].map((k) =>
    Math.round(w0 * neither[k]! + wa * ca[k]! + wb * cb[k]! + wab * both[k]!),
  ) as RGB;
}

export interface BlendOptions {
  /** Quantile of the expressing cells each gene's scale tops out at */
  clip?: number;
  opacity: number;
  /** Cells outside a non-empty selection are faded, as in the other grids */
  selected?: ReadonlySet<number> | null;
}

/** RGBA for the cells `indices` (draw order) from the blend of genes `a` and `b`. */
export function blendColors(
  indices: Uint32Array,
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  opts: BlendOptions,
): Uint8Array {
  const sa = blendScale(a, opts.clip);
  const sb = blendScale(b, opts.clip);
  const rgba = new Uint8Array(indices.length * 4);
  const full = Math.round(opts.opacity * 255);
  const faded = Math.round(opts.opacity * 0.15 * 255);
  const anySelected = !!opts.selected && opts.selected.size > 0;
  for (let s = 0; s < indices.length; s++) {
    const i = indices[s]!;
    const [r, g, bl] = blendColor(scaled(a[i]!, sa), scaled(b[i]!, sb));
    rgba[s * 4] = r;
    rgba[s * 4 + 1] = g;
    rgba[s * 4 + 2] = bl;
    rgba[s * 4 + 3] = anySelected && !opts.selected!.has(i) ? faded : full;
  }
  return rgba;
}

/**
 * `order` re-ordered so higher values come last (drawn on top), ties keeping
 * their place in `order`: a feature plot shows its expressing cells over the
 * many that do not express the gene.
 */
export function valueOrder(order: Uint32Array, value: (cell: number) => number): Uint32Array {
  const keyed = Array.from(order, (cell, slot) => ({ cell, slot, v: value(cell) }));
  keyed.sort((p, q) => {
    const a = Number.isFinite(p.v) ? p.v : -Infinity;
    const b = Number.isFinite(q.v) ? q.v : -Infinity;
    return a === b ? p.slot - q.slot : a - b;
  });
  return Uint32Array.from(keyed, (k) => k.cell);
}

/** Min and max over several genes: the shared colour scale of the feature grid. */
export function sharedRange(values: ArrayLike<number>[]): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  for (const vals of values) {
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i]!;
      if (!Number.isFinite(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min <= max ? { min, max } : { min: 0, max: 0 };
}

/** Values of the cells `cells` (all when null) by group name, for a violin. */
export function valuesByGroup(
  values: ArrayLike<number>,
  codes: ArrayLike<number>,
  categories: readonly string[],
  cells: ArrayLike<number> | null = null,
): Record<string, number[]> {
  const out: Record<string, number[]> = {};
  const n = cells ? cells.length : Math.min(values.length, codes.length);
  for (let k = 0; k < n; k++) {
    const i = cells ? cells[k]! : k;
    const code = codes[i]!;
    const name = categories[code] ?? String(code);
    (out[name] ??= []).push(values[i]!);
  }
  return out;
}

/** Every `step`-th cell so at most `max` are kept (all when fewer); null keeps all. */
export function strideSample(total: number, max: number): Uint32Array | null {
  if (total <= max) return null;
  const step = Math.ceil(total / max);
  const out = new Uint32Array(Math.ceil(total / step));
  for (let k = 0, i = 0; i < total; i += step, k++) out[k] = i;
  return out;
}
