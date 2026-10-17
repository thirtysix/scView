import type { DEResponse } from "@/api/de";
import type { GseaGene, GseaRequest, GseaResponse, GseaTerm } from "@/api/enrichment";
import { collectionLabel } from "@/api/genesets";
import type { CsvCell } from "@/lib/csv";
import { formatPct } from "@/lib/deDetection";
import { formatPValue } from "@/lib/formatting";

/**
 * Preranked GSEA (backend core/gsea.py) of a DE result or a marker group. The
 * ranking is the whole tested gene list, so unlike the ORA hand-off it does not
 * depend on the table's cut-offs.
 */

/** A DE result's genes as the GSEA request sends them: every tested gene. */
export function gseaGenesFromDE(r: DEResponse): GseaGene[] {
  // GSEA ranks a Wilcoxon result by its z-score, a pseudobulk one by signed -log10 p.
  const useStat = r.method !== "pseudobulk";
  return r.genes.map((g) => ({
    gene: g.gene,
    logfoldchange: g.logfoldchange,
    pval: g.pval,
    score: useStat ? (g.stat ?? null) : null,
  }));
}

/**
 * Which DE result GSEA ranks: by the id the server kept it under, since a whole
 * transcriptome sent back is past the request size cap; the genes themselves
 * only for a result without one.
 */
export function gseaSourceFromDE(r: DEResponse): Pick<GseaRequest, "de_result_id" | "genes"> {
  return r.result_id ? { de_result_id: r.result_id } : { genes: gseaGenesFromDE(r) };
}

export type GseaSortField = "term" | "nes" | "fdr" | "leading_edge";
export type GseaSortDir = "asc" | "desc";

/** A copy of `results` in table order. NES sorts by size, so both directions read alike. */
export function sortGseaResults(
  results: readonly GseaTerm[],
  field: GseaSortField,
  dir: GseaSortDir,
): GseaTerm[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...results].sort((a, b) => {
    if (field === "term") return sign * a.term.localeCompare(b.term);
    if (field === "nes") return sign * (Math.abs(a.nes) - Math.abs(b.nes));
    if (field === "fdr") return sign * (a.fdr - b.fdr || a.pvalue - b.pvalue);
    return sign * (a.leading_edge.length - b.leading_edge.length);
  });
}

/** Same column flips; a new one starts at its natural order (smallest FDR first, else largest). */
export function nextGseaSort(
  current: { field: GseaSortField; dir: GseaSortDir },
  field: GseaSortField,
): { field: GseaSortField; dir: GseaSortDir } {
  if (current.field === field) return { field, dir: current.dir === "asc" ? "desc" : "asc" };
  return { field, dir: field === "fdr" || field === "term" ? "asc" : "desc" };
}

/**
 * A permutation p-value or FDR. 0 means no permutation scored as high, i.e. below
 * the resolution 1/permutations, so it is shown as a bound, not as zero.
 */
export function formatPermP(p: number, permutations: number): string {
  if (p > 0) return formatPValue(p);
  return `< ${formatPValue(1 / Math.max(permutations, 1))}`;
}

/** One line on how the run was made, for under the results. */
export function describeGseaRun(d: GseaResponse): string {
  const fmt = (n: number) => n.toLocaleString("en-US");
  const out = d.n_excluded_ribo_mito > 0 ? ` (${fmt(d.n_excluded_ribo_mito)} ribosomal / mito left out)` : "";
  const took = d.cached ? "from cache" : `${d.seconds.toFixed(1)} s`;
  // A prefiltered DE result ranks only the genes it tested.
  const pre = d.de_min_pct
    ? ` The DE tested only genes detected in at least ${formatPct(d.de_min_pct)} of either ` +
      `group (${fmt(d.de_n_genes_skipped ?? 0)} left out), so only those are ranked.`
    : "";
  return (
    `${fmt(d.n_genes_ranked)} genes ranked by ${d.rank_metric_label}${out}. ` +
    `${fmt(d.n_sets_tested)} gene sets with ${d.min_size}-${d.max_size} genes in the ranking, ` +
    `${fmt(d.permutations)} permutations (${took}).${pre}`
  );
}

export const GSEA_CSV_HEADERS = [
  "direction", "term", "collection", "nes", "es", "nominal_p", "fdr", "set_size",
  "leading_edge_size", "leading_edge",
];

/** Both directions as CSV rows; `labels` names what a positive and a negative NES mean. */
export function gseaCsvRows(d: GseaResponse, labels: { up: string; down: string }): CsvCell[][] {
  return (["up", "down"] as const).flatMap((s) =>
    d[s].results.map((r) => [
      labels[s],
      r.term,
      collectionLabel(r),
      r.nes.toFixed(4),
      r.es.toFixed(4),
      r.pvalue.toExponential(4),
      r.fdr.toExponential(4),
      r.set_size,
      r.leading_edge.length,
      r.leading_edge.join(";"),
    ]),
  );
}
