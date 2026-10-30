/** Helpers for the Spot composition subtab (backend core/spot_deconvolution.py). */
import type { ShareKind, SpotSummary } from "@/api/spotDeconvolution";

/** How to read shares of a Visium HD bin: few cells and few counts in each. */
export function binCaveat(um: number | undefined): string {
  if (um != null && um <= 8) {
    return `At ${um} µm a bin holds about one cell and few counts: read the dominant type of a bin; its fractions are noisy.`;
  }
  if (um != null && um <= 16) return `At ${um} µm a bin holds about 2 to 4 cells, so its fractions mean more than at 8 µm.`;
  return "A bin holds few cells and few counts: read its dominant type before its fractions.";
}

export const SPOT_METHOD_LABELS = {
  rctd: "RCTD-style (Poisson-lognormal)",
  nnls: "NNLS (baseline)",
} as const;

/** "12%", "0.4%", "<0.1%", "-" for none. */
export function fmtShare(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "-";
  const p = v * 100;
  if (p > 0 && p < 0.1) return "<0.1%";
  return `${p < 10 ? p.toFixed(1) : Math.round(p)}%`;
}

export interface TypeRow {
  type: string;
  index: number;
  mean: number | null;
  max: number | null;
  nTop: number;
  nCells: number;
}

/** One row per type with its mean and largest share of the chosen kind, largest mean first. */
export function typeRows(s: SpotSummary, kind: ShareKind): TypeRow[] {
  return s.type_stats
    .map((t, index) => ({
      type: t.type,
      index,
      mean: kind === "rna" ? t.rna_mean : t.cell_mean,
      max: kind === "rna" ? t.rna_max : t.cell_max,
      nTop: t.n_top,
      nCells: t.n_reference_cells,
    }))
    .sort((a, b) => (b.mean ?? -1) - (a.mean ?? -1));
}

/** The overlay's name on the plot and in its legend. */
export function shareLabel(type: string, kind: ShareKind): string {
  return `${type} ${kind === "rna" ? "RNA share" : "cell share (est.)"}`;
}

/** Share values from the API (null = not fitted) as the plot's overlay (NaN = no colour). */
export function toOverlayValues(values: (number | null)[]): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    out[i] = v == null ? NaN : v;
  }
  return out;
}

/** The heatmap's clusters in their saved order and the types to show (largest overall share first, at most `max`). */
export function heatmapLayout(
  s: SpotSummary,
  kind: ShareKind,
  max = 20,
): { clusters: string[]; types: { type: string; index: number }[]; values: Record<string, number[]> } {
  const values = kind === "rna" ? s.clusters.rna : s.clusters.cells;
  const clusters = Object.keys(values);
  const types = typeRows(s, kind)
    .slice(0, max)
    .map((r) => ({ type: r.type, index: r.index }));
  return { clusters, types, values };
}

/** A share (0-1) as a cell's background: one hue whose opacity is the share, readable in both themes. */
export function shareFill(v: number | null | undefined): string {
  const a = v == null || !Number.isFinite(v) ? 0 : Math.max(0, Math.min(1, v));
  return `rgba(37, 99, 235, ${(0.08 + 0.92 * a).toFixed(3)})`;
}

/** Text colour for a heatmap cell: white on the darker fills. */
export function shareText(v: number | null | undefined): string {
  return v != null && v > 0.5 ? "#ffffff" : "";
}
