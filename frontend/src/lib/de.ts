import type { DEGene, DEResponse } from "@/api/de";
import type { ObsColumnInfo } from "@/api/types";
import { formatPValue } from "@/lib/formatting";

/** Most-specific first: a donor column beats a per-sample one, which may also
 *  encode the condition (sample = donor x condition still works, unpaired). */
const REPLICATE_PATTERNS: RegExp[] = [
  /donor/i,
  /patient/i,
  /individual/i,
  /^ind$/i,
  /subject/i,
  /participant/i,
  /replicate/i,
  /sample/i,
  /orig\.ident/i,
];

/** The backend caps pseudobulk samples at this (core/pseudobulk.py MAX_SAMPLES);
 *  a column with more levels is per-cell, not per-donor. */
const MAX_REPLICATE_LEVELS = 300;

/** Columns that could hold donors/samples: 2..300 distinct values, excluding the
 *  columns already used to split or scope the comparison. */
export function replicateColumns(
  cols: ObsColumnInfo[] | undefined,
  exclude: string[],
): ObsColumnInfo[] {
  return (cols ?? []).filter(
    (c) =>
      !exclude.includes(c.name) &&
      c.n_unique >= 2 &&
      c.n_unique <= MAX_REPLICATE_LEVELS &&
      !/float|bool/i.test(c.dtype),
  );
}

/** Pre-select the most donor-like column by name, or "" when none looks like one. */
export function suggestReplicateColumn(cols: ObsColumnInfo[]): string {
  for (const re of REPLICATE_PATTERNS) {
    const hit = cols.find((c) => re.test(c.name));
    if (hit) return hit.name;
  }
  return "";
}

// ---------------------------------------------------------------------------
// "Ask the co-pilot" about a DE result. DE results live only in the subtab (they
// are not stored in the dataset), so, like the other subtabs, the question
// itself carries the specifics the co-pilot needs.
// ---------------------------------------------------------------------------

/** What was compared, captured when the DE was run. */
export interface DEComparison {
  a: string;
  b: string;
  /** e.g. "cell_type = CD14 Mono" when the comparison was restricted. */
  scope?: string;
  /** Between groups of a column (vs a cell selection against the rest). */
  betweenGroups: boolean;
}

const num = (n: number | null | undefined) => (n ?? 0).toLocaleString("en-US");

/** One-line description of the comparison and the test behind it. */
export function describeComparison(c: DEComparison, r: DEResponse): string {
  const where = c.scope ? ` within ${c.scope}` : "";
  const cells = `${num(r.n_selected)} vs ${num(r.n_rest)} cells`;
  if (r.method === "pseudobulk") {
    const paired = r.design?.includes("+") ? ", paired by donor" : "";
    return (
      `${c.a} vs ${c.b}${where}, pseudobulk DESeq2 over ${r.n_samples_a} vs ` +
      `${r.n_samples_b} donors (column ${r.replicate_key}${paired}; ${cells})`
    );
  }
  // Between groups, say how far a cell-level test can be trusted.
  const caveat = !c.betweenGroups
    ? ""
    : r.underreplicated
      ? "; under-replicated, so p-values are optimistic"
      : "; p-values are optimistic if the groups also differ by donor";
  return `${c.a} vs ${c.b}${where}, cell-level Wilcoxon (${cells}${caveat})`;
}

/** Question about one gene of a DE result. */
export function deGeneQuestion(c: DEComparison, r: DEResponse, g: DEGene): string {
  const higher = g.logfoldchange >= 0 ? c.a : c.b;
  return (
    `In a differential-expression comparison of ${describeComparison(c, r)}, ${g.gene} has ` +
    `log2 fold change ${g.logfoldchange.toFixed(2)} (higher in ${higher}), adjusted p ` +
    `${formatPValue(g.pval_adj)}. What is ${g.gene}, and what could this difference mean biologically?`
  );
}

/** Question about a whole DE result: its most significant genes on each side. */
export function deResultQuestion(
  c: DEComparison,
  r: DEResponse,
  significant: DEGene[],
  fcThreshold: number,
  maxPerSide = 12,
): string {
  const up = significant.filter((g) => g.logfoldchange > 0).slice(0, maxPerSide);
  const down = significant.filter((g) => g.logfoldchange < 0).slice(0, maxPerSide);
  const list = (gs: DEGene[]) => (gs.length ? gs.map((g) => g.gene).join(", ") : "none");
  return (
    `I ran differential expression: ${describeComparison(c, r)}. ` +
    `${num(significant.length)} of ${num(r.genes.length)} genes pass adjusted p <= 0.05 and ` +
    `|log2FC| >= ${fcThreshold}. Most significant higher in ${c.a}: ${list(up)}. ` +
    `Most significant higher in ${c.b}: ${list(down)}. ` +
    `What biological programmes or cell states do these differences suggest, and what should I check next?`
  );
}
