import type { Data } from "plotly.js";
import type { DeconvCheck, DeconvResult } from "@/api/deconvolution";

export type FractionKind = "cell" | "rna";

/** One bar trace per cell type, stacked: each bulk sample's fractions (in %). A flagged
 *  segment (far above the type's share of the reference cells) is hatched. */
export function stackedBarTraces(result: DeconvResult, kind: FractionKind, colorOf: (type: string) => string): Data[] {
  const x = result.samples.map((s) => s.sample);
  return result.types.map((t) => {
    const flagged = result.samples.map((s) => (s.flagged ?? []).includes(t));
    return {
      type: "bar" as const,
      name: t,
      x,
      y: result.samples.map((s) => {
        const v = (kind === "cell" ? s.cell_fractions : s.rna_fractions)[t];
        return v == null ? null : v * 100;
      }),
      marker: {
        color: colorOf(t),
        // Plotly takes one shape per bar; the typings only know a single one.
        ...(flagged.some(Boolean) ? { pattern: { shape: flagged.map((f) => (f ? "/" : "")) as unknown as "/" } } : {}),
      },
      customdata: flagged.map((f) => (f ? " (doubtful: far above its share of the reference cells)" : "")),
      hovertemplate: `%{x}<br>${t}: %{y:.1f}%%{customdata}<extra></extra>`,
    };
  });
}

/** "stim = CTRL" as the select's value, and back. */
export function cellsKey(column: string, value: string): string {
  return JSON.stringify([column, value]);
}

export function parseCellsKey(key: string): { cells_column: string | null; cells_value: string | null } {
  if (!key) return { cells_column: null, cells_value: null };
  try {
    const [column, value] = JSON.parse(key) as [string, string];
    return { cells_column: column, cells_value: value };
  } catch {
    return { cells_column: null, cells_value: null };
  }
}

/** Truth against estimate of the self-check, one trace per cell type (fractions in %). */
export function checkScatterTraces(check: DeconvCheck, method: string, colorOf: (type: string) => string): Data[] {
  const pts = check.points[method];
  if (!pts) return [];
  const out: Data[] = check.types.map((t, k) => {
    const idx = pts.type.flatMap((ti, i) => (ti === k && pts.truth[i] != null && pts.estimate[i] != null ? [i] : []));
    return {
      type: "scattergl" as const,
      mode: "markers" as const,
      name: t,
      x: idx.map((i) => (pts.truth[i] as number) * 100),
      y: idx.map((i) => (pts.estimate[i] as number) * 100),
      marker: { color: colorOf(t), size: 4, opacity: 0.6 },
      hovertemplate: `${t}<br>true %{x:.1f}%<br>estimated %{y:.1f}%<extra></extra>`,
    };
  });
  out.push({
    type: "scatter" as const,
    mode: "lines" as const,
    name: "estimate = truth",
    x: [0, 100],
    y: [0, 100],
    line: { color: "rgba(100,116,139,0.6)", dash: "dot", width: 1 },
    hoverinfo: "skip" as const,
    showlegend: false,
  });
  return out;
}

export function pct(v: number | null | undefined, digits = 1): string {
  return v == null ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 2): string {
  return v == null ? "–" : v.toFixed(digits);
}

/** "about 20 s" / "about 2 min" */
export function aboutSeconds(s: number | null | undefined): string | null {
  if (s == null) return null;
  return s < 90 ? `about ${Math.max(5, Math.ceil(s / 5) * 5)} s` : `about ${Math.ceil(s / 60)} min`;
}

/** The largest fraction of each sample, for a one-line summary. */
export function topType(sample: DeconvResult["samples"][number], kind: FractionKind): { type: string; value: number } | null {
  const fr = kind === "cell" ? sample.cell_fractions : sample.rna_fractions;
  let best: { type: string; value: number } | null = null;
  for (const [type, value] of Object.entries(fr)) {
    if (value != null && (best == null || value > best.value)) best = { type, value };
  }
  return best;
}
