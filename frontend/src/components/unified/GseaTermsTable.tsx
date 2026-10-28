import { ArrowDown, ArrowUp, ArrowUpDown, Loader2, Sparkles } from "lucide-react";
import { collectionLabel } from "@/api/genesets";
import { PathwaySourceAction } from "@/components/unified/PathwaySourceAction";
import type { GseaTerm } from "@/api/enrichment";
import { formatPermP, type GseaSortDir, type GseaSortField } from "@/lib/gsea";

interface Props {
  /** Already in display order (lib/gsea.ts sortGseaResults). */
  results: GseaTerm[];
  permutations: number;
  sortField: GseaSortField;
  sortDir: GseaSortDir;
  onSort: (field: GseaSortField) => void;
  /** A row click: score the term's leading-edge genes on the scatter. */
  onTermClick: (r: GseaTerm) => void;
  onAsk: (r: GseaTerm) => void;
  scoringTerm: string | null;
  /** A Reactome row's "Diagram" button (KEGG / WikiPathways rows get an outbound link). */
  onDiagram?: (r: GseaTerm) => void;
  className?: string;
}

/**
 * GSEA results as a sortable table in the style of EnrichmentTermsTable: term +
 * collection, NES, FDR, leading edge (genes driving the score / set genes ranked).
 */
export function GseaTermsTable({
  results,
  permutations,
  sortField,
  sortDir,
  onSort,
  onTermClick,
  onAsk,
  scoringTerm,
  onDiagram,
  className = "max-h-72",
}: Props) {
  const sortIcon = (field: GseaSortField) => {
    if (sortField !== field) return <ArrowUpDown className="h-2.5 w-2.5 text-slate-400" />;
    return sortDir === "asc" ? (
      <ArrowUp className="h-2.5 w-2.5 text-blue-500" />
    ) : (
      <ArrowDown className="h-2.5 w-2.5 text-blue-500" />
    );
  };
  const th = "cursor-pointer px-2 py-1.5 font-medium text-slate-600 hover:text-slate-800";

  return (
    <div className={`overflow-auto rounded-lg border border-slate-200 ${className}`}>
      <table className="w-full text-xs">
        <thead className="sticky top-0 z-10">
          <tr className="border-b border-slate-200 bg-slate-50">
            <th onClick={() => onSort("term")} title="Gene set (term) and its collection; click a row to score the set's genes on the scatter" className={`${th} text-left`}>
              <span className="flex items-center gap-1">Term {sortIcon("term")}</span>
            </th>
            <th
              onClick={() => onSort("nes")}
              title="Normalised enrichment score: how strongly the set's genes bunch at the top (positive) or bottom (negative) of the ranking, scaled for set size."
              className={`${th} text-right`}
            >
              <span className="flex items-center justify-end gap-1">NES {sortIcon("nes")}</span>
            </th>
            <th
              onClick={() => onSort("fdr")}
              title="False discovery rate from permutations of the gene sets (GSEA's q-value)."
              className={`${th} text-right`}
            >
              <span className="flex items-center justify-end gap-1">FDR {sortIcon("fdr")}</span>
            </th>
            <th
              onClick={() => onSort("leading_edge")}
              title="Leading edge: the set's genes that drive the score / the set's genes in the ranking."
              className={`${th} whitespace-nowrap text-right`}
            >
              <span className="flex items-center justify-end gap-1">Lead {sortIcon("leading_edge")}</span>
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
                title={
                  `Click to score the leading edge on the scatter\nCollection: ${collectionLabel(r)}\n` +
                  `Nominal p: ${formatPermP(r.pvalue, permutations)}\n` +
                  `Leading edge: ${r.leading_edge.slice(0, 12).join(", ")}${r.leading_edge.length > 12 ? "..." : ""}`
                }
              >
                <td className="max-w-[200px] px-2 py-1.5">
                  <div className="flex items-center gap-1.5">
                    {isScoring && <Loader2 className="h-3 w-3 flex-shrink-0 animate-spin text-blue-500" />}
                    <div className="min-w-0 flex-1 truncate font-medium text-slate-800">{r.term}</div>
                    <PathwaySourceAction
                      sourceId={r.source_id}
                      sourceUrl={r.source_url}
                      onDiagram={onDiagram ? () => onDiagram(r) : undefined}
                    />
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
                  <div className="truncate text-[10px] text-slate-400">
                    {isScoring ? "Scoring..." : `${collectionLabel(r)} · ${r.leading_edge.slice(0, 4).join(", ")}`}
                  </div>
                </td>
                <td className="px-2 py-1.5 text-right tabular-nums">
                  <span className={r.nes > 0 ? "text-emerald-600" : "text-red-600"}>
                    {r.nes > 0 ? "+" : ""}
                    {r.nes.toFixed(2)}
                  </span>
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums">
                  <span className={r.fdr < 0.05 ? "font-medium text-blue-600" : "text-slate-600"}>
                    {formatPermP(r.fdr, permutations)}
                  </span>
                </td>
                <td className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums text-slate-600">
                  {r.leading_edge.length}/{r.set_size}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
