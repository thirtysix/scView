/**
 * Selection-vs-rest overlay histograms (N18, after FCS Express): one value per
 * cell (a gene, a score or a numeric QC column) split into two sides, each
 * side's distribution normalised to its own cell count so a small selection and
 * a large rest can be compared, plus a descriptive effect summary.
 *
 * Nothing here is a test: the KS distance and the Mann-Whitney AUC say how far
 * apart the two distributions are, not whether the difference is "significant"
 * (cells from one sample are not independent replicates).
 */
import { pseudoLog } from "@/lib/logAxis";

/** Which side each cell is on: 1 = A (the selection or group), 2 = B, 0 = neither. */
export type Sides = Uint8Array;

/** The selection against every other cell. */
export function sidesFromSelection(n: number, selected: ReadonlySet<number>): Sides {
  const out = new Uint8Array(n).fill(2);
  for (const i of selected) if (i >= 0 && i < n) out[i] = 1;
  return out;
}

/** A group (category code) against the rest, or against another group. */
export function sidesFromGroups(codes: ArrayLike<number>, a: number, b: number | "rest"): Sides {
  const out = new Uint8Array(codes.length);
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i]!;
    if (c === a) out[i] = 1;
    else if (b === "rest" ? c >= 0 : c === b) out[i] = 2;
  }
  return out;
}

/** The finite values of each side, sorted ascending. */
export function splitSides(values: ArrayLike<number>, sides: Sides): { a: Float64Array; b: Float64Array } {
  const n = Math.min(values.length, sides.length);
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(values[i]!)) continue;
    if (sides[i] === 1) na++;
    else if (sides[i] === 2) nb++;
  }
  const a = new Float64Array(na);
  const b = new Float64Array(nb);
  let ia = 0;
  let ib = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    if (sides[i] === 1) a[ia++] = v;
    else if (sides[i] === 2) b[ib++] = v;
  }
  return { a: a.sort(), b: b.sort() };
}

/** Interpolated quantile of sorted values (NaN when empty). */
export function quantile(sorted: ArrayLike<number>, p: number): number {
  const n = sorted.length;
  if (n === 0) return NaN;
  const i = (n - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.min(lo + 1, n - 1);
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (i - lo);
}

/**
 * Kolmogorov-Smirnov distance: the largest gap between the two empirical
 * cumulative distributions (0 = identical, 1 = no overlap).
 */
export function ksDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = a.length;
  const nb = b.length;
  if (!na || !nb) return NaN;
  let i = 0;
  let j = 0;
  let d = 0;
  while (i < na && j < nb) {
    const v = Math.min(a[i]!, b[j]!);
    while (i < na && a[i]! <= v) i++; // ties advance together
    while (j < nb && b[j]! <= v) j++;
    d = Math.max(d, Math.abs(i / na - j / nb));
  }
  return d;
}

/**
 * Mann-Whitney AUC: the chance that a random A cell's value is above a random
 * B cell's (ties count half). 0.5 = no shift; 1 = every A above every B.
 * Both inputs sorted ascending.
 */
export function mannWhitneyAuc(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const na = a.length;
  const nb = b.length;
  if (!na || !nb) return NaN;
  // For each A value: B values below it, plus half those equal to it.
  let below = 0;
  let equalEnd = 0;
  let u = 0;
  for (let i = 0; i < na; i++) {
    const v = a[i]!;
    while (below < nb && b[below]! < v) below++;
    if (equalEnd < below) equalEnd = below;
    while (equalEnd < nb && b[equalEnd]! <= v) equalEnd++;
    u += below + (equalEnd - below) / 2;
  }
  return u / (na * nb);
}

export interface SideSummary {
  n: number;
  median: number;
  mean: number;
  /** Share of cells above 0: for genes (and counts), the cells expressing it. */
  fracPositive: number;
}

export function summarise(sorted: ArrayLike<number>): SideSummary {
  const n = sorted.length;
  let sum = 0;
  let pos = 0;
  for (let i = 0; i < n; i++) {
    sum += sorted[i]!;
    if (sorted[i]! > 0) pos++;
  }
  return { n, median: quantile(sorted, 0.5), mean: n ? sum / n : NaN, fracPositive: n ? pos / n : NaN };
}

export interface Comparison {
  a: SideSummary;
  b: SideSummary;
  /** median(A) - median(B), in the values' own units */
  medianDiff: number;
  ks: number;
  auc: number;
}

export function compareSides(a: ArrayLike<number>, b: ArrayLike<number>): Comparison {
  const sa = summarise(a);
  const sb = summarise(b);
  return { a: sa, b: sb, medianDiff: sa.median - sb.median, ks: ksDistance(a, b), auc: mannWhitneyAuc(a, b) };
}

/** Three significant digits; whole numbers with separators from 1,000. */
export function fmtValue(v: number): string {
  if (!Number.isFinite(v)) return "–";
  return Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : String(Number(v.toPrecision(3)));
}

/**
 * The effect summary in words: the median difference, the KS distance and the
 * AUC, each said as what it describes rather than as a test result.
 */
export function effectLines(c: Comparison, nameA: string, nameB: string): string[] {
  if (!c.a.n || !c.b.n) return [];
  const sign = c.medianDiff > 0 ? "+" : "";
  const auc = Math.round(c.auc * 100);
  return [
    `Difference in medians (${nameA} − ${nameB}): ${sign}${fmtValue(c.medianDiff)}`,
    `KS distance ${c.ks.toFixed(2)}: the largest gap between the two cumulative distributions (0 = same, 1 = no overlap)`,
    `Mann-Whitney AUC ${c.auc.toFixed(2)}: of one cell from each side, the ${nameA} one has the higher value ${auc}% of the time (50% = no shift)`,
  ];
}

export interface Curves {
  /** Bin centres (histogram) or grid points (density), on the plotted scale. */
  x: number[];
  a: number[];
  b: number[];
  /** Bin width on the plotted scale (histogram), for the bars. */
  width: number;
}

/** The plotted scale: linear, or sign(v) log10(1 + |v|) as the violins' log axis. */
export function toScale(v: number, log: boolean): number {
  return log ? pseudoLog(v) : v;
}

function range(a: ArrayLike<number>, b: ArrayLike<number>, log: boolean): [number, number] {
  const ends = [a[0], a[a.length - 1], b[0], b[b.length - 1]].filter(
    (v): v is number => v !== undefined && Number.isFinite(v),
  );
  if (!ends.length) return [0, 1];
  let lo = toScale(Math.min(...ends), log);
  let hi = toScale(Math.max(...ends), log);
  if (hi === lo) {
    lo -= 0.5;
    hi += 0.5;
  }
  return [lo, hi];
}

/**
 * Shared-bin histograms, each side as the fraction of its own cells per bin
 * (each side sums to 1), on the plotted scale.
 */
export function histograms(a: ArrayLike<number>, b: ArrayLike<number>, nBins = 40, log = false): Curves {
  const [lo, hi] = range(a, b, log);
  const width = (hi - lo) / nBins;
  const count = (vals: ArrayLike<number>) => {
    const out = new Array<number>(nBins).fill(0);
    for (let i = 0; i < vals.length; i++) {
      const k = Math.min(nBins - 1, Math.max(0, Math.floor((toScale(vals[i]!, log) - lo) / width)));
      out[k]! += 1;
    }
    return vals.length ? out.map((c) => c / vals.length) : out;
  };
  const x = Array.from({ length: nBins }, (_, k) => lo + (k + 0.5) * width);
  return { x, a: count(a), b: count(b), width };
}

/** Silverman's rule-of-thumb bandwidth (on the plotted scale), floored above 0. */
export function bandwidth(scaled: readonly number[], span: number): number {
  const n = scaled.length;
  if (n < 2) return span / 20 || 1;
  const s = [...scaled].sort((x, y) => x - y);
  const mean = s.reduce((t, v) => t + v, 0) / n;
  const sd = Math.sqrt(s.reduce((t, v) => t + (v - mean) ** 2, 0) / (n - 1));
  const iqr = quantile(s, 0.75) - quantile(s, 0.25);
  const spread = Math.min(sd, iqr > 0 ? iqr / 1.34 : sd);
  const h = 0.9 * spread * n ** -0.2;
  return h > 0 ? h : span / 50 || 1;
}

/**
 * Gaussian kernel density of each side on a shared grid (each integrates to
 * about 1). The values are first binned on a fine grid, so the cost is the grid
 * size squared, not the cell count.
 */
export function densities(a: ArrayLike<number>, b: ArrayLike<number>, nGrid = 200, log = false): Curves {
  const [lo0, hi0] = range(a, b, log);
  const scale = (vals: ArrayLike<number>) => Array.from(vals, (v) => toScale(v, log));
  const sa = scale(a);
  const sb = scale(b);
  const ha = bandwidth(sa, hi0 - lo0);
  const hb = bandwidth(sb, hi0 - lo0);
  // Room for the kernels' tails, so each curve comes down to 0 and keeps its area.
  const pad = Math.max((hi0 - lo0) * 0.05, 3 * Math.max(sa.length ? ha : 0, sb.length ? hb : 0));
  const lo = lo0 - pad;
  const hi = hi0 + pad;
  const step = (hi - lo) / (nGrid - 1);
  const x = Array.from({ length: nGrid }, (_, k) => lo + k * step);
  const kde = (scaled: number[], h0: number) => {
    // A kernel narrower than the grid would be one spike per grid point.
    const h = Math.max(h0, step);
    const n = scaled.length;
    if (!n) return new Array<number>(nGrid).fill(0);
    const counts = new Array<number>(nGrid).fill(0);
    for (const v of scaled) counts[Math.min(nGrid - 1, Math.max(0, Math.round((v - lo) / step)))]! += 1;
    const norm = 1 / (n * h * Math.sqrt(2 * Math.PI));
    const reach = Math.ceil((4 * h) / step);
    const out = new Array<number>(nGrid).fill(0);
    for (let k = 0; k < nGrid; k++) {
      if (!counts[k]) continue;
      for (let j = Math.max(0, k - reach); j <= Math.min(nGrid - 1, k + reach); j++) {
        const z = ((j - k) * step) / h;
        out[j]! += counts[k]! * Math.exp(-0.5 * z * z) * norm;
      }
    }
    return out;
  };
  return { x, a: kde(sa, ha), b: kde(sb, hb), width: step };
}
