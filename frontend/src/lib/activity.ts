/**
 * Helpers for the TF / pathway activity views: which sources the heatmap shows,
 * the per-source summary rows, and the DE profile sent for scoring.
 */
import type { ActivitySummary, ActivityNetwork } from "@/api/activity";
import type { DEGene } from "@/api/de";

export interface SourceRow {
  source: string;
  index: number;
  n_targets: number;
  /** Spread of the group means (SD): how much the source separates the groups. */
  spread: number;
  /** The group with the highest mean, and that mean. */
  top_group: string | null;
  top_mean: number | null;
}

function sd(values: number[]): number {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length < 2) return 0;
  const m = finite.reduce((a, b) => a + b, 0) / finite.length;
  return Math.sqrt(finite.reduce((a, b) => a + (b - m) ** 2, 0) / finite.length);
}

/** One row per scored source, most group-separating first (by name without groups). */
export function sourceRows(s: ActivitySummary): SourceRow[] {
  const rows = s.sources.map((source, j) => {
    const col = s.mean?.map((row) => row[j]!) ?? [];
    let top = -1;
    col.forEach((v, i) => {
      if (Number.isFinite(v) && (top < 0 || v > col[top]!)) top = i;
    });
    return {
      source,
      index: j,
      n_targets: s.n_targets[j] ?? 0,
      spread: sd(col),
      top_group: top >= 0 ? (s.groups?.[top] ?? null) : null,
      top_mean: top >= 0 ? col[top]! : null,
    };
  });
  return s.mean
    ? rows.sort((a, b) => b.spread - a.spread || a.source.localeCompare(b.source))
    : rows.sort((a, b) => a.source.localeCompare(b.source));
}

/** Rows whose source name contains the query (case-insensitive). */
export function filterRows(rows: SourceRow[], query: string): SourceRow[] {
  const q = query.trim().toLowerCase();
  return q ? rows.filter((r) => r.source.toLowerCase().includes(q)) : rows;
}

/** Heatmap data: the top-`n` sources by spread as rows (strongest on top), groups as columns. */
export function heatmapData(s: ActivitySummary, n: number) {
  if (!s.mean || !s.groups) return null;
  const top = sourceRows(s).slice(0, n);
  const z = top.map((r) => s.mean!.map((row) => row[r.index]!));
  const max = Math.max(1e-9, ...z.flat().filter(Number.isFinite).map(Math.abs));
  // Plotly draws the first row at the bottom: reverse so the strongest is on top.
  return { x: s.groups, y: top.map((r) => r.source).reverse(), z: z.reverse(), zmax: max };
}

/** The DE profile to score: every gene with a finite test statistic, to 4
 *  significant digits. Full float precision nearly doubles the request (Kang's
 *  14k genes: 425 KB), which must stay under the backend's 512 KB JSON cap;
 *  the activity t-values move by less than 0.1 %. */
export function deProfile(genes: DEGene[]): { genes: string[]; stats: number[] } {
  const out = { genes: [] as string[], stats: [] as number[] };
  for (const g of genes) {
    if (typeof g.stat === "number" && Number.isFinite(g.stat)) {
      out.genes.push(g.gene);
      out.stats.push(Number(g.stat.toPrecision(4)));
    }
  }
  return out;
}

/** The overlay label for a source's per-cell activity, e.g. "STAT1 (TF activity)". */
export function activityLabel(network: ActivityNetwork, source: string): string {
  return `${source} (${network === "collectri" ? "TF" : "pathway"} activity)`;
}
