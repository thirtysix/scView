/**
 * The one decision for a continuous colouring (C025): which colormap, what
 * range, and where its centre is, from what the overlay IS (a gene, a score, a
 * QC column), its values and the user's colour settings. Every place that
 * colours cells by a value (the scatter, split grid, gene grid, legends, PNG
 * export) draws from one ResolvedColorScale, so they agree with each other.
 *
 * Two parts, so the sort over up to 200k cells happens once per overlay
 * change and not per setting change: overlayStats (sorted copy, range, signs)
 * then resolveColorScale (cheap: quantiles on the sorted copy).
 */
import type { AnyColormapName, ColormapName, DivergingColormapName } from "@/lib/colors";
import { fmtValue, quantile } from "@/lib/overlayHistogram";

/**
 * What the values are. Scores (gene-set, TF / pathway activity, cell cycle,
 * stress) and protein CLR are signed with a meaningful 0; expression is not,
 * even when a value dips below 0 (scaled data). An obs column is signed when
 * its values have both signs (S_score, G2M_score); QC metrics never do.
 */
export type OverlayKind = "expression" | "protein" | "score" | "obs";

/** How signed values are drawn: "auto" = diverging for values of both signs, except expression. */
export type SignedColormapMode = "auto" | "sequential" | "diverging";

export const SIGNED_MODE_OPTIONS: { value: SignedColormapMode; label: string; title: string }[] = [
  { value: "auto", label: "Auto", title: "Scores, activities, protein CLR and cell columns (such as S_score) with values of both signs use the diverging map, centred on 0; expression and QC columns (never negative) stay sequential" },
  { value: "sequential", label: "Sequential", title: "Always the sequential map, lowest to highest value" },
  { value: "diverging", label: "Diverging", title: "Any overlay with values of both signs uses the diverging map, centred on 0" },
];

export type ClipPreset = "none" | "p1" | "p2" | "p5" | "manual";

/** The colour range: the data range, a percentile pair, or a range set by hand. */
export interface ClipSetting {
  preset: ClipPreset;
  /** Manual only; a missing end is the data's own */
  min?: number;
  max?: number;
}

export const DEFAULT_CLIP: ClipSetting = { preset: "none" };

export const CLIP_PERCENTILES: Record<"p1" | "p2" | "p5", [number, number]> = {
  p1: [0.01, 0.99],
  p2: [0.02, 0.98],
  p5: [0.05, 0.95],
};

export const CLIP_PRESET_OPTIONS: { value: ClipPreset; label: string; title: string }[] = [
  { value: "none", label: "No clipping", title: "The colour range is the data range (one bright outlier can wash out the rest)" },
  { value: "p1", label: "1–99%", title: "Colour range from the 1st to the 99th percentile; cells outside take the end colours" },
  { value: "p2", label: "2–98%", title: "Colour range from the 2nd to the 98th percentile; cells outside take the end colours" },
  { value: "p5", label: "5–95%", title: "Colour range from the 5th to the 95th percentile; cells outside take the end colours" },
  { value: "manual", label: "Manual", title: "A colour range typed in; cells outside take the end colours" },
];

/** Settings the decision reads (the settings store, with a view's clip override applied). */
export interface ColorScaleSettings {
  sequential: ColormapName;
  diverging: DivergingColormapName;
  signedMode: SignedColormapMode;
  clip: ClipSetting;
}

/** One overlay's values, prepared once: the finite ones sorted, their range and signs. */
export interface OverlayStats {
  /** The finite values, ascending (a copy; the overlay array is untouched) */
  sorted: Float32Array;
  /** Finite values */
  n: number;
  /** All values, missing ones included */
  nAll: number;
  min: number;
  max: number;
  hasNegative: boolean;
  hasPositive: boolean;
}

export function overlayStats(values: ArrayLike<number> | null | undefined): OverlayStats {
  const nAll = values?.length ?? 0;
  const sorted = new Float32Array(nAll);
  let n = 0;
  if (values) {
    for (let i = 0; i < nAll; i++) {
      const v = values[i]!;
      if (Number.isFinite(v)) sorted[n++] = v;
    }
  }
  const finite = n === nAll ? sorted : sorted.subarray(0, n);
  finite.sort();
  return {
    sorted: finite,
    n,
    nAll,
    min: n ? finite[0]! : NaN,
    max: n ? finite[n - 1]! : NaN,
    hasNegative: n > 0 && finite[0]! < 0,
    hasPositive: n > 0 && finite[n - 1]! > 0,
  };
}

/** Stats of several overlays together (the gene grid's shared scale). */
export function mergedStats(all: readonly OverlayStats[]): OverlayStats {
  const nAll = all.reduce((t, s) => t + s.nAll, 0);
  const n = all.reduce((t, s) => t + s.n, 0);
  const sorted = new Float32Array(n);
  let at = 0;
  for (const s of all) {
    sorted.set(s.sorted, at);
    at += s.n;
  }
  sorted.sort();
  return {
    sorted,
    n,
    nAll,
    min: n ? sorted[0]! : NaN,
    max: n ? sorted[n - 1]! : NaN,
    hasNegative: all.some((s) => s.hasNegative),
    hasPositive: all.some((s) => s.hasPositive),
  };
}

export interface ResolvedColorScale {
  colormap: AnyColormapName;
  diverging: boolean;
  /** The colour range: values outside take the end colours */
  min: number;
  max: number;
  /** 0 on a diverging scale (its midpoint colour), else null */
  centre: number | null;
  /** The data's own range (NaN with no finite values) */
  dataMin: number;
  dataMax: number;
  /** How the range was narrowed, or null when it is the data range */
  clip: {
    preset: ClipPreset;
    /** Share of the finite values below min / above max */
    fracBelow: number;
    fracAbove: number;
  } | null;
  /** Finite values the range was taken over */
  n: number;
  /** A range fixed by what the values are (a 0-1 share), not taken from the data: never clipped */
  fixed?: boolean;
}

/** Index of the first sorted value >= x (so values below x count `lowerBound`). */
function lowerBound(sorted: Float32Array, x: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid]! < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Index of the first sorted value > x. */
function upperBound(sorted: Float32Array, x: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sorted[mid]! <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Diverging for these values under this mode? (The "auto" rule lives here.) */
export function wantsDiverging(kind: OverlayKind, stats: OverlayStats, mode: SignedColormapMode): boolean {
  if (mode === "sequential") return false;
  const signed = stats.hasNegative && stats.hasPositive;
  if (mode === "diverging") return signed;
  return signed && kind !== "expression";
}

/**
 * The colormap, range and centre for one overlay. With no finite values the
 * range is 0-1 (the scatter's old fallback). On a diverging scale the range is
 * symmetric about 0: the larger magnitude of the clipped ends either side.
 */
export function resolveColorScale(
  kind: OverlayKind,
  stats: OverlayStats,
  settings: ColorScaleSettings,
): ResolvedColorScale {
  const diverging = wantsDiverging(kind, stats, settings.signedMode);
  const colormap: AnyColormapName = diverging ? settings.diverging : settings.sequential;
  if (stats.n === 0) {
    return { colormap, diverging, min: 0, max: 1, centre: diverging ? 0 : null, dataMin: NaN, dataMax: NaN, clip: null, n: 0 };
  }
  const { sorted } = stats;
  const { preset } = settings.clip;
  let lo = stats.min;
  let hi = stats.max;
  if (preset === "manual") {
    const a = Number.isFinite(settings.clip.min) ? settings.clip.min! : stats.min;
    const b = Number.isFinite(settings.clip.max) ? settings.clip.max! : stats.max;
    lo = Math.min(a, b);
    hi = Math.max(a, b);
  } else if (preset !== "none") {
    const [p, q] = CLIP_PERCENTILES[preset];
    lo = quantile(sorted, p);
    hi = quantile(sorted, q);
  }
  let min = lo;
  let max = hi;
  if (diverging) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    min = -m;
    max = m;
  }
  // A manual range is always reported, even one wider than the data.
  const clipped = preset === "manual" || (preset !== "none" && (min > stats.min || max < stats.max));
  const clip = clipped
    ? {
        preset,
        fracBelow: lowerBound(sorted, min) / stats.n,
        fracAbove: (stats.n - upperBound(sorted, max)) / stats.n,
      }
    : null;
  return { colormap, diverging, min, max, centre: diverging ? 0 : null, dataMin: stats.min, dataMax: stats.max, clip, n: stats.n };
}

/**
 * A sequential scale over a range fixed in advance (a share, 0 to 1), whatever
 * the values: two plots of it compare by colour, and no clipping applies.
 */
export function fixedColorScale(
  stats: OverlayStats,
  colormap: ColormapName,
  min: number,
  max: number,
): ResolvedColorScale {
  return { colormap, diverging: false, min, max, centre: null, dataMin: stats.min, dataMax: stats.max, clip: null, n: stats.n, fixed: true };
}

/** "1.2%" or "<0.1%"; "0%" only when truly none. */
function pct(frac: number): string {
  if (!(frac > 0)) return "0%";
  const p = frac * 100;
  if (p < 0.1) return "<0.1%";
  return `${p < 10 ? p.toFixed(1) : Math.round(p)}%`;
}

const ORDINAL: Record<"p1" | "p2" | "p5", string> = {
  p1: "1st–99th percentiles",
  p2: "2nd–98th percentiles",
  p5: "5th–95th percentiles",
};

/**
 * What the legend says about the range, honestly: the colour range, how it
 * was narrowed and the share of cells outside it, e.g.
 * "Colour range 0 to 3.2, clipped at the 1st–99th percentiles; 1.2% of cells above."
 */
export function colorScaleCaption(scale: ResolvedColorScale, noun = "cells"): string {
  if (scale.n === 0) return "No values to colour by.";
  const range = `Colour range ${fmtValue(scale.min)} to ${fmtValue(scale.max)}`;
  if (scale.fixed) return `${range}, fixed (not clipped to the data).`;
  const centred = scale.diverging ? ", centred on 0" : "";
  if (!scale.clip) {
    const data = scale.diverging && (scale.dataMin !== scale.min || scale.dataMax !== scale.max)
      ? ` (data ${fmtValue(scale.dataMin)} to ${fmtValue(scale.dataMax)})`
      : "";
    return `${range}${centred}${data}.`;
  }
  const how =
    scale.clip.preset === "manual"
      ? "set by hand"
      : `clipped at the ${ORDINAL[scale.clip.preset as "p1" | "p2" | "p5"]}`;
  const outside: string[] = [];
  if (scale.clip.fracAbove > 0) outside.push(`${pct(scale.clip.fracAbove)} of ${noun} above`);
  if (scale.clip.fracBelow > 0) outside.push(`${pct(scale.clip.fracBelow)} below`);
  const tail = outside.length ? `; ${outside.join(", ")}` : `; no ${noun} outside`;
  return `${range}${centred}, ${how} (data ${fmtValue(scale.dataMin)} to ${fmtValue(scale.dataMax)})${tail}.`;
}

/** The scale's parts the scatter needs (a stable, minimal object for props). */
export function scatterScale(scale: ResolvedColorScale): { colormap: AnyColormapName; min: number; max: number } {
  return { colormap: scale.colormap, min: scale.min, max: scale.max };
}
