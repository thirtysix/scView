import type { TrendGene, TrendGenesResult } from "@/api/trajectory";
import type { CsvCell } from "@/lib/csv";

/** Genes in heatmap order: by the pseudotime of their fitted peak (early
 *  first), then the most changing first. */
export function heatmapOrder(genes: readonly TrendGene[]): TrendGene[] {
  return [...genes].sort(
    (a, b) =>
      (a.peak_pseudotime ?? Infinity) - (b.peak_pseudotime ?? Infinity) ||
      (b.f_stat ?? Infinity) - (a.f_stat ?? Infinity) ||
      a.gene.localeCompare(b.gene),
  );
}

/** A row scaled to 0..1 (its min..max), so genes of any level share one colour
 *  scale; a flat row is all 0. */
export function scaleRow(values: readonly number[]): number[] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo;
  return values.map((v) => (span > 0 ? (v - lo) / span : 0));
}

export const PATTERN_LABEL: Record<TrendGene["pattern"], string> = {
  rising: "rises",
  falling: "falls",
  transient: "peaks mid-way",
};

/** Up to `n` evenly spaced bin indices for axis ticks, always the first and last. */
export function tickBins(nBins: number, n = 6): number[] {
  if (nBins <= 0) return [];
  if (nBins <= n) return Array.from({ length: nBins }, (_, i) => i);
  const out = new Set<number>();
  for (let k = 0; k < n; k++) out.add(Math.round((k * (nBins - 1)) / (n - 1)));
  return [...out];
}

/** The ranked list as a CSV table (headers + rows), most changing first. */
export function trendGenesCsv(result: TrendGenesResult): { headers: string[]; rows: CsvCell[][] } {
  return {
    headers: ["rank", "gene", "pattern", "peak_pseudotime", "amplitude", "f_stat", "pval", "pval_adj", "r2", "n_detected"],
    rows: result.genes.map((g, i) => [
      i + 1, g.gene, g.pattern, g.peak_pseudotime, g.amplitude, g.f_stat, g.pval, g.pval_adj, g.r2, g.n_detected,
    ]),
  };
}
