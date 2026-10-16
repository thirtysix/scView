import type { EnrichmentResult } from "@/api/enrichment";

export type EnrichmentSortField = "term" | "adj_pval" | "overlap";
export type SortDir = "asc" | "desc";

/** A copy of `results` in the order the terms table shows them. */
export function sortEnrichmentResults(
  results: readonly EnrichmentResult[],
  field: EnrichmentSortField,
  dir: SortDir,
): EnrichmentResult[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...results].sort((a, b) => {
    if (field === "term") return sign * a.term.localeCompare(b.term);
    if (field === "adj_pval") return sign * (a.adjusted_pvalue - b.adjusted_pvalue);
    return sign * (a.overlap_count - b.overlap_count);
  });
}

/** A header click: the same column flips direction; a new one starts at its natural
 *  order (smallest p first, otherwise largest first). */
export function nextSort(
  current: { field: EnrichmentSortField; dir: SortDir },
  field: EnrichmentSortField,
): { field: EnrichmentSortField; dir: SortDir } {
  if (current.field === field) return { field, dir: current.dir === "asc" ? "desc" : "asc" };
  return { field, dir: field === "adj_pval" ? "asc" : "desc" };
}
