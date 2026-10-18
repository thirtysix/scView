import { Download } from "lucide-react";
import type { DEMethodsResponse } from "@/api/de";
import { InfoTip } from "@/components/common/InfoTip";
import { downloadCsv } from "@/lib/csv";
import { formatPct } from "@/lib/deDetection";
import { agreementCsv, agreementRows, cellMethodInfo } from "@/lib/deMethods";

interface Props {
  result: DEMethodsResponse;
  /** Display names of the two sides: "up" = higher in `labelA`. */
  labelA: string;
  labelB: string;
  csvName: string;
  onGeneClick: (gene: string) => void;
}

/** Colour for a Jaccard index: green when the tests mostly agree, amber when not. */
function agreementClass(j: number | null): string {
  if (j == null) return "text-slate-400";
  if (j >= 0.7) return "text-emerald-700";
  if (j >= 0.4) return "text-slate-700";
  return "text-amber-700";
}

/**
 * One comparison under every cell-level DE test: how far Wilcoxon, Welch's
 * t-test and logistic regression agree on the top genes and on the ranking,
 * and the genes only one of them puts in its top N.
 */
export function DEMethodsAgreement({ result, labelA, labelB, csvName, onGeneClick }: Props) {
  const rows = agreementRows(result);
  const methods = result.methods;
  const fmt = (n: number) => n.toLocaleString();
  const exportCsv = () => {
    const { headers, rows: data } = agreementCsv(result);
    downloadCsv(csvName, headers, data);
  };
  const unavailable = Object.entries(result.unavailable);

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2.5 text-xs">
      <div className="flex items-center gap-2">
        <span className="font-semibold text-slate-700">Method agreement</span>
        <InfoTip width={320}>
          The same cells and the same {fmt(result.n_genes_tested)} genes under each test. Jaccard: the
          share of genes two tests both put in their top {result.top_n} on a side (1 = the same
          genes), shared genes in brackets. Spearman: the rank correlation of the two tests&apos;
          per-gene statistics over every gene. Wilcoxon and Welch&apos;s t-test ask the same question
          of each gene on its own, so they usually agree; logistic regression fits all genes at once
          and ranks the ones that jointly tell the groups apart, which can differ a lot.
        </InfoTip>
        <span className="flex-1 truncate text-[10px] text-slate-400">
          {labelA} (n={fmt(result.n_selected)}) vs {labelB} (n={fmt(result.n_rest)})
          {result.min_pct ? ` · genes in ≥ ${formatPct(result.min_pct)} of either group` : ""}
        </span>
        <button
          onClick={exportCsv}
          title="Export the agreement as CSV"
          className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
        >
          <Download className="h-3 w-3" />
          CSV
        </button>
      </div>

      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-slate-200 text-slate-500">
            <th className="py-1 text-left font-medium">Test</th>
            <th className="py-1 text-left font-medium">Statistic</th>
            <th className="py-1 text-right font-medium" title="Genes with adjusted p ≤ 0.05">adj. p ≤ 0.05</th>
            <th className="py-1 text-right font-medium">Time</th>
          </tr>
        </thead>
        <tbody>
          {methods.map((m) => (
            <tr key={m.method} className="border-b border-slate-50">
              <td className="py-1 text-slate-700" title={cellMethodInfo(m.method).help}>{m.label}</td>
              <td className="py-1 text-slate-500">{m.stat_label}</td>
              <td className="py-1 text-right tabular-nums text-slate-700">
                {m.n_significant == null ? <span className="text-slate-400">no p-values</span> : fmt(m.n_significant)}
              </td>
              <td className="py-1 text-right tabular-nums text-slate-500">{m.seconds.toFixed(1)} s</td>
            </tr>
          ))}
        </tbody>
      </table>

      <table className="w-full text-[11px]" aria-label="Agreement between tests">
        <thead>
          <tr className="border-b border-slate-200 text-slate-500">
            <th className="py-1 text-left font-medium">Pair</th>
            <th className="py-1 text-right font-medium" title={`Jaccard of the top ${result.top_n} genes higher in ${labelA}`}>
              Top {result.top_n} ↑ {labelA}
            </th>
            <th className="py-1 text-right font-medium" title={`Jaccard of the top ${result.top_n} genes higher in ${labelB}`}>
              Top {result.top_n} ↑ {labelB}
            </th>
            <th className="py-1 text-right font-medium">Spearman</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} className="border-b border-slate-50">
              <td className="py-1 text-slate-700">{r.pair}</td>
              <td className={`py-1 text-right tabular-nums ${agreementClass(r.jaccardUp)}`}>{r.up}</td>
              <td className={`py-1 text-right tabular-nums ${agreementClass(r.jaccardDown)}`}>{r.down}</td>
              <td className="py-1 text-right tabular-nums text-slate-700">{r.spearman}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div>
        <div className="mb-1 font-medium text-slate-600">
          In one test&apos;s top {result.top_n} only
          <span className="ml-1 font-normal text-slate-400">
            (rank on that side in each test; click to overlay)
          </span>
        </div>
        {result.unique.length === 0 ? (
          <p className="text-slate-400">Every top gene of each test is in another test&apos;s top {result.top_n} too.</p>
        ) : (
          <div className="max-h-56 overflow-auto rounded border border-slate-100">
            <table className="w-full text-[11px]" aria-label="Genes only one test ranks top">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="text-slate-500">
                  <th className="px-1.5 py-1 text-left font-medium">Gene</th>
                  <th className="px-1.5 py-1 text-left font-medium">Only in</th>
                  <th className="px-1.5 py-1 text-left font-medium">Higher in</th>
                  {methods.map((m) => (
                    <th key={m.method} className="px-1.5 py-1 text-right font-medium" title={m.label}>
                      {cellMethodInfo(m.method).label.split(" ")[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.unique.map((u) => (
                  <tr
                    key={`${u.method}|${u.side}|${u.gene}`}
                    onClick={() => onGeneClick(u.gene)}
                    className="cursor-pointer border-t border-slate-50 hover:bg-blue-50"
                  >
                    <td className="px-1.5 py-1 font-mono text-slate-800">{u.gene}</td>
                    <td className="px-1.5 py-1 text-slate-600">{cellMethodInfo(u.method).label}</td>
                    <td className="px-1.5 py-1 text-slate-600">{u.side === "up" ? labelA : labelB}</td>
                    {methods.map((m) => (
                      <td
                        key={m.method}
                        className={`px-1.5 py-1 text-right tabular-nums ${
                          m.method === u.method ? "font-semibold text-slate-800" : "text-slate-500"
                        }`}
                      >
                        {u.ranks[m.method]?.toLocaleString() ?? "–"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {unavailable.length > 0 && (
        <p className="text-[10px] text-slate-400">
          {unavailable.map(([name, why]) => `${name} not run: ${why}`).join(" ")}
        </p>
      )}
    </div>
  );
}
