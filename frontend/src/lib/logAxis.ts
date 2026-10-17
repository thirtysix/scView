/**
 * A log value axis for the violins (N02). Plotly's own log axis drops zeros
 * and still draws a violin's density from the linear values, so a log toggle
 * there would hide the many cells at 0 and smear the rest. The violins switch
 * to plotting sign(v) * log10(1 + |v|) instead, on a linear axis whose ticks
 * are labelled in the original units: zeros and negative scores stay, and the
 * density is estimated on the log scale it is drawn on.
 */

/** sign(v) * log10(1 + |v|): log-like for large values, 0 at 0, symmetric for negatives. */
export function pseudoLog(v: number): number {
  return Math.sign(v) * Math.log10(1 + Math.abs(v));
}

/** A tick label: 1, 0.05, 2.5k, 1M. */
export function tickLabel(v: number): string {
  const a = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  const trim = (x: number) => String(Number(x.toPrecision(3)));
  if (a >= 1e6) return `${sign}${trim(a / 1e6)}M`;
  if (a >= 1e3) return `${sign}${trim(a / 1e3)}k`;
  return `${sign}${trim(a)}`;
}

/** Ticks closer than this share of the axis are dropped (the larger one stays). */
const MIN_GAP = 0.06;

/**
 * Ticks for a pseudo-log axis showing values from `min` to `max` (original
 * units): 0 when in range, then 1-2-5 steps per decade over a short range or
 * powers of ten over a long one, mirrored for negatives. `tickvals` are on the
 * plotted (pseudo-log) scale, `ticktext` in the original units.
 */
export function pseudoLogTicks(min: number, max: number): { tickvals: number[]; ticktext: string[] } {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { tickvals: [], ticktext: [] };
  const hi = Math.max(Math.abs(min), Math.abs(max));
  if (hi === 0) return { tickvals: [0], ticktext: ["0"] };
  const kmax = Math.floor(Math.log10(hi));
  const kmin = hi >= 10 ? 0 : kmax - 1; // below 10, a decade of fractions too
  const mantissas = pseudoLog(hi) <= 2.3 ? [1, 2, 5] : [1];
  const magnitudes: number[] = [];
  for (let k = kmin; k <= kmax; k++) for (const m of mantissas) magnitudes.push(m * 10 ** k);

  const tol = (pseudoLog(max) - pseudoLog(min)) * 0.02;
  const inRange = (v: number) => pseudoLog(v) >= pseudoLog(min) - tol && pseudoLog(v) <= pseudoLog(max) + tol;
  const values = [...magnitudes.map((m) => -m), 0, ...magnitudes].filter(inRange).sort((a, b) => a - b);

  // Thin out crowded ticks (the small magnitudes bunch up near 0), keeping 0
  // and, outwards from it, the next tick far enough from the last one kept.
  const span = pseudoLog(max) - pseudoLog(min) || 1;
  const keepFrom = (side: number[]) => {
    const kept: number[] = [];
    let last = 0;
    for (const v of side) {
      if (Math.abs(pseudoLog(v) - last) >= MIN_GAP * span) {
        kept.push(v);
        last = pseudoLog(v);
      }
    }
    return kept;
  };
  const pos = keepFrom(values.filter((v) => v > 0));
  const neg = keepFrom(values.filter((v) => v < 0).reverse()).reverse();
  const ticks = [...neg, ...(values.includes(0) ? [0] : []), ...pos];
  return { tickvals: ticks.map(pseudoLog), ticktext: ticks.map(tickLabel) };
}

/** Quartiles (interpolated between the sorted values) and range of `values`. */
export function summary(values: readonly number[]): { min: number; q1: number; median: number; q3: number; max: number } | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return null;
  const q = (p: number) => {
    const i = (v.length - 1) * p;
    const lo = Math.floor(i);
    return v[lo]! + (v[Math.min(lo + 1, v.length - 1)]! - v[lo]!) * (i - lo);
  };
  return { min: v[0]!, q1: q(0.25), median: q(0.5), q3: q(0.75), max: v[v.length - 1]! };
}
