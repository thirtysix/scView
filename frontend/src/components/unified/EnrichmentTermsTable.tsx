import { ArrowDown, ArrowUp, ArrowUpDown, Loader2, Sparkles } from "lucide-react";
import { collectionLabel } from "@/api/genesets";
import type { EnrichmentResult } from "@/api/enrichment";
import { formatPValue } from "@/lib/formatting";
import type { EnrichmentSortField, SortDir } from "@/lib/enrichment";

interface Props {
  /** Already in display order (lib/enrichment.ts sortEnrichmentResults). */
  results: EnrichmentResult[];
  sortField: EnrichmentSortField;
  sortDir: SortDir;
  onSort: (field: EnrichmentSortField) => void;
  /** A row click: score the term's genes on the scatter. */
  onTermClick: (r: EnrichmentResult) => void;
  onAsk: (r: EnrichmentResult) => void;
  /** The term being scored, if any (its row spins, the others dim). */
  scoringTerm: string | null;
  /** Classes for the scroll container (height cap). */
  className?: string;
}

/**
 * Enriched gene sets as a sortable table: term + collection, adjusted p, overlap.
 * Used by the Enrichment subtab (marker genes) and the DE subtab (DE genes).
 */
export function EnrichmentTermsTable({
  results,
  sortField,
  sortDir,
  onSort,
  onTermClick,
  onAsk,
  scoringTerm,
  className = "max-h-[calc(100vh-500px)]",
}: Props) {
  const sortIcon = (field: EnrichmentSortField) => {
    if (sortField !== field) return <ArrowUpDown className="h-2.5 w-2.5 text-slate-400" />;
    return sortDir === "asc" ? (
      <ArrowUp className="h-2.5 w-2.5 text-blue-500" />
    ) : (
      <ArrowDown className="h-2.5 w-2.5 text-blue-500" />
    );
  };

  return (
    <div className={`overflow-auto rounded-lg border border-slate-200 ${className}`}>
      <table className="w-full text-xs">
        <thead className="sticky top-0 z-10">
          <tr className="border-b border-slate-200 bg-slate-50">
            <th
              onClick={() => onSort("term")}
              title="Gene set (term) and its collection; click a row to score the set's genes on the scatter"
              className="cursor-pointer px-2 py-1.5 text-left font-medium text-slate-600 hover:text-slate-800"
            >
              <span className="flex items-center gap-1">Term {sortIcon("term")}</span>
            </th>
            <th
              onClick={() => onSort("adj_pval")}
              title="Hypergeometric p of the overlap between the query genes and the set, with the tested genes as background; Benjamini-Hochberg adjusted over the sets tested"
              className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
            >
              <span className="flex items-center justify-end gap-1">Adj.P {sortIcon("adj_pval")}</span>
            </th>
            <th
              onClick={() => onSort("overlap")}
              title="Query genes in the set / genes in the set"
              className="cursor-pointer px-2 py-1.5 text-right font-medium text-slate-600 hover:text-slate-800"
            >
              <span className="flex items-center justify-end gap-1">Overlap {sortIcon("overlap")}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => {
            const isScoring = scoringTerm === r.term;
            return (
              <tr
                key={`${r.collection}-${r.term}`}
                onClick={() => onTermClick(r)}
                className={`group cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50 ${isScoring ? "bg-blue-50/70" : ""} ${scoringTerm && !isScoring ? "opacity-50" : ""}`}
                title={`Click to score on scatter\nCollection: ${collectionLabel(r)}\nGenes: ${r.genes.slice(0, 10).join(", ")}${r.genes.length > 10 ? "..." : ""}`}
              >
                <td className="max-w-[200px] px-2 py-1.5">
                  <div className="flex items-center gap-1.5">
                    {isScoring && <Loader2 className="h-3 w-3 flex-shrink-0 animate-spin text-blue-500" />}
                    <div className="min-w-0 flex-1 truncate font-medium text-slate-800">{r.term}</div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onAsk(r);
                      }}
                      title="Ask the co-pilot about this pathway"
                      className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                    >
                      <Sparkles className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="text-[10px] text-slate-400">{isScoring ? "Scoring..." : collectionLabel(r)}</div>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={r.adjusted_pvalue < 0.05 ? "font-medium text-blue-600" : "text-slate-600"}>
                    {formatPValue(r.adjusted_pvalue)}
                  </span>
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums text-slate-600">
                  {r.overlap_count}/{r.gene_count}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
