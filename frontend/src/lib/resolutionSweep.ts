/** Helpers for the clustering resolution sweep (components/unified/ResolutionSweep). */

export interface ResolutionLimits {
  min: number;
  max: number;
  maxCount: number;
}

/**
 * The resolutions typed as "0.2, 0.4 1.0": rounded to 0.01, deduplicated and
 * sorted, as the backend will use them, or the reason they can't be used.
 */
export function parseResolutions(
  text: string,
  limits: ResolutionLimits,
): { values: number[]; error: string | null } {
  const parts = text.split(/[\s,;]+/).filter(Boolean);
  if (parts.length === 0) return { values: [], error: "Enter at least one resolution." };
  const values = new Set<number>();
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v)) return { values: [], error: `"${p}" is not a number.` };
    const r = Math.round(v * 100) / 100;
    if (r < limits.min || r > limits.max) {
      return { values: [], error: `Resolutions must be between ${limits.min} and ${limits.max}.` };
    }
    values.add(r);
  }
  if (values.size > limits.maxCount) {
    return { values: [], error: `Try at most ${limits.maxCount} resolutions at a time.` };
  }
  return { values: [...values].sort((a, b) => a - b), error: null };
}

/** The resolutions as an editable list: "0.2, 0.4, 1.0". */
export function formatResolutions(values: readonly number[]): string {
  return values.map((v) => (Number.isInteger(v) ? v.toFixed(1) : String(v))).join(", ");
}

/** Index of the row with the highest silhouette score, or -1 when none has one. */
export function bestSilhouette(rows: readonly { silhouette: number | null }[]): number {
  let best = -1;
  rows.forEach((r, i) => {
    if (r.silhouette != null && (best < 0 || r.silhouette > rows[best]!.silhouette!)) best = i;
  });
  return best;
}
