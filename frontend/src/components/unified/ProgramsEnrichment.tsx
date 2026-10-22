import type { ProgramEnrichment } from "@/api/programs";
import { Caption, CapNote } from "@/components/common/Caption";
import { formatPValue } from "@/lib/formatting";
import { prettyTerm } from "@/lib/programs";

const SHOWN = 15;

/** A program's top genes tested for over-representation in the Hallmark sets. */
export function ProgramsEnrichment({ result }: { result: ProgramEnrichment }) {
  const rows = result.results.slice(0, SHOWN);
  return (
    <div className="space-y-1" data-testid="programs-enrichment">
      <div className="text-[11px] font-medium text-slate-600">{result.name}: enriched gene sets</div>
      {result.warning && <p className="text-[11px] text-amber-700">{result.warning}</p>}
      {rows.length === 0 ? (
        <p className="text-[11px] text-slate-500">No gene set shares a gene with the program&apos;s top genes.</p>
      ) : (
        <div className="max-h-64 overflow-auto rounded-lg border border-slate-200">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-slate-50">
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="px-2 py-1 text-left font-medium" title="Gene set (MSigDB Hallmark unless another collection was chosen)">
                  Gene set
                </th>
                <th
                  className="px-2 py-1 text-right font-medium"
                  title="Program top genes in the set / the set's genes among the genes NMF used"
                >
                  Overlap
                </th>
                <th
                  className="px-2 py-1 text-right font-medium"
                  title="Hypergeometric p-value, Benjamini-Hochberg adjusted over the sets tested; universe: the genes NMF used"
                >
                  Adj. p
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={`${r.collection}:${r.term}`}
                  className={`border-b border-slate-50 ${r.adjusted_pvalue < 0.05 ? "" : "text-slate-400"}`}
                  title={r.genes.join(", ")}
                >
                  <td className="px-2 py-1">{prettyTerm(r.term)}</td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {r.overlap_count}/{r.gene_count}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">{formatPValue(r.adjusted_pvalue)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <CapNote shown={SHOWN} total={result.results.length} by="adjusted p" />
      <Caption
        n={[
          { value: result.n_query, unit: "top genes tested" },
          { value: result.background_n, unit: "genes in the universe" },
        ]}
      >
        Over-representation of the program&apos;s top genes (by gene score) among the variable genes NMF used;
        {` ${result.n_significant} of ${result.n_terms} sets with adjusted p < 0.05`}
        {result.n_excluded_ribo_mito > 0 && `; ${result.n_excluded_ribo_mito} ribosomal or mitochondrial genes left out`}.
      </Caption>
    </div>
  );
}
