/** Helpers for the count-split marker check (components/unified/CountSplitCheck). */
import type { SplitGene, SplitGroup, SplitMarkersResult } from "@/api/splitMarkers";

/** -log10 p, with p = 0 (an underflow) shown at the cap rather than at infinity. */
export const NEG_LOG10_CAP = 300;

export function negLog10(p: number | null | undefined): number | null {
  if (p == null || !Number.isFinite(p) || p < 0) return null;
  if (p <= 0) return NEG_LOG10_CAP;
  return Math.min(NEG_LOG10_CAP, -Math.log10(p));
}

export type PointKind = "kept" | "lost" | "other";

export interface ComparisonPoint {
  gene: string;
  x: number; // ordinary -log10 adjusted p
  y: number; // split -log10 adjusted p
  kind: PointKind;
}

/** One point per gene with both adjusted p-values: kept / lost marker, or another gene. */
export function comparisonPoints(genes: readonly SplitGene[]): ComparisonPoint[] {
  const out: ComparisonPoint[] = [];
  for (const g of genes) {
    const x = negLog10(g.pval_adj);
    const y = negLog10(g.split_pval_adj);
    if (x == null || y == null) continue;
    out.push({ gene: g.gene, x, y, kind: g.marker ? (g.kept ? "kept" : "lost") : "other" });
  }
  return out;
}

/** Gene -> split adjusted p for the Markers table's "split p-values" column. */
export function splitPValueMap(group: SplitGroup | undefined): Map<string, number | null> {
  const m = new Map<string, number | null>();
  for (const g of group?.genes ?? []) m.set(g.gene, g.split_pval_adj);
  return m;
}

/** "keeps 30 of 38 markers" / "no markers to check" / "not tested: <why>". */
export function keptText(g: SplitGroup): string {
  if (!g.tested) return g.reason ? `not tested: ${g.reason}` : "not tested";
  if (g.n_ordinary === 0) return "no markers to check";
  return `keeps ${g.n_kept.toLocaleString()} of ${g.n_ordinary.toLocaleString()} marker${g.n_ordinary === 1 ? "" : "s"}`;
}

/** The headline: "7 of 9 groups keep at least 10 markers on held-out counts". */
export function summaryText(r: SplitMarkersResult): string {
  return (
    `${r.n_keep_enough} of ${r.n_tested_groups} group${r.n_tested_groups === 1 ? "" : "s"} keep at least ` +
    `${r.thresholds.keeps_enough} markers on held-out counts; ${r.n_kept.toLocaleString()} of ` +
    `${r.n_ordinary.toLocaleString()} ordinary markers kept`
  );
}

/** Groups that keep fewest markers first (the ones to look at), then by name order. */
export function groupsByConcern(groups: readonly SplitGroup[], keepsEnough = 10): SplitGroup[] {
  return groups
    .map((g, i) => ({ g, i }))
    .sort(
      (a, b) =>
        Number(b.g.tested) - Number(a.g.tested) ||
        Number(a.g.n_kept >= keepsEnough) - Number(b.g.n_kept >= keepsEnough) ||
        a.i - b.i,
    )
    .map(({ g }) => g);
}
