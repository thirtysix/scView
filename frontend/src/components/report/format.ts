/** Number formats shared by the report sections ("–" for a missing value). */
export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || !Number.isFinite(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
export const num = (v: number | null | undefined, digits = 2) =>
  v == null || !Number.isFinite(v) ? "–" : Number.isInteger(v) && digits === 0 ? v.toLocaleString() : v.toFixed(digits);
export const pval = (p: number | null | undefined) =>
  p == null || !Number.isFinite(p) ? "–" : p === 0 ? "< 1e-300" : p < 0.001 ? p.toExponential(2) : p.toFixed(3);
export const count = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString());
