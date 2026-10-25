import { useEffect, useMemo, useState } from "react";
import { ExternalLink, Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { getOffTarget, type OffTargetResult } from "@/api/offtarget";
import { plotConfig } from "@/lib/plotConfig";
import { VERDICT_STYLE, categoryTitle, heatmapHeight, summaryText, topText } from "@/lib/offtarget";

/**
 * "Elsewhere in the body" in the Markers subtab (backend core/offtarget.py): the
 * selected group's markers looked up in the Human Protein Atlas, tissue by tissue
 * and cell type by cell type. A marker specific among these clusters can still be
 * expressed all over the body, which matters for sorting, imaging panels and targets.
 * Off until asked (the HPA tables, ~21 MB, are fetched on first use); then it follows
 * the selected group.
 */
export function OffTargetPanel({
  datasetId,
  column,
  group,
  onGeneClick,
}: {
  datasetId: string;
  column: string;
  group: string;
  onGeneClick: (gene: string) => void;
}) {
  const [on, setOn] = useState(false);
  const [result, setResult] = useState<OffTargetResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!on || !group || !column) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getOffTarget(datasetId, column, group)
      .then((r) => !cancelled && setResult(r))
      .catch((e) => {
        if (cancelled) return;
        setResult(null);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [on, datasetId, column, group]);

  const hm = result?.heatmap;
  const traces = useMemo(
    () =>
      hm && hm.genes.length
        ? [
            {
              type: "heatmap" as const,
              z: hm.log2_ntpm,
              x: hm.tissues,
              y: hm.genes,
              colorscale: "Viridis",
              colorbar: { title: { text: "log2(nTPM+1)", side: "right" as const }, thickness: 10 },
              hovertemplate: "%{y} in %{x}: log2(nTPM+1) %{z:.1f}<extra></extra>",
              name: "HPA tissues",
            },
          ]
        : [],
    [hm],
  );

  return (
    <section aria-label="Elsewhere in the body" className="space-y-1.5 rounded-lg border border-slate-200 p-2">
      <div className="flex items-center gap-1">
        <span className="text-[11px] font-medium text-slate-600">Elsewhere in the body</span>
        <InfoTip label="About the HPA lookup" width={320}>
          Each marker of the group is looked up in the Human Protein Atlas: RNA across human tissues (HPA + GTEx
          consensus, nTPM) and across HPA&rsquo;s single-cell types (nCPM). The category follows HPA&rsquo;s rules:
          enriched = at least 4x the next tissue; group enriched = a group of 2&ndash;5 tissues (2&ndash;10 cell types)
          at least 4x every other; enhanced = at least 4x the mean of the others; low specificity = detected but none of
          these. <b>Restricted</b> means enriched or group enriched in either table; <b>broad</b> means low tissue
          specificity. Tau is 0 for even expression and 1 for one tissue only. A marker is flagged when it is specific
          among these groups (tau &ge; 0.5 here) but broad in the body.
        </InfoTip>
        <span className="ml-auto" />
        {!on && (
          <button
            type="button"
            onClick={() => setOn(true)}
            disabled={!group}
            className="rounded bg-blue-500 px-2 py-0.5 text-xs font-medium text-white hover:bg-blue-600 disabled:opacity-50"
          >
            Look up in HPA
          </button>
        )}
      </div>
      {!on && (
        <p className="text-[10px] text-slate-400">
          Shows where {group ? `${group}'s` : "the group's"} markers are expressed across human tissues and cell
          types. The first lookup downloads the Human Protein Atlas tables (about 21 MB).
        </p>
      )}
      {loading && (
        <div className="flex items-center gap-1.5 text-[11px] text-slate-500" role="status">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Looking up {group}&apos;s markers in the Human Protein Atlas…
        </div>
      )}
      {error && <p className="text-[11px] text-amber-700">{error}</p>}
      {result && !loading && (
        <div className="space-y-1.5">
          <p className="text-[11px] text-slate-600">
            <span className="font-medium">{summaryText(result)}.</span>
          </p>
          {result.note && <p className="text-[10px] text-amber-700">{result.note}</p>}
          {result.genes.length > 0 && (
            <div className="max-h-64 overflow-auto rounded border border-slate-100">
              <table className="w-full border-collapse text-[11px]">
                <thead className="sticky top-0 bg-slate-50">
                  <tr className="border-b border-slate-200 text-left text-slate-500">
                    <th className="px-1.5 py-1 font-medium">Gene</th>
                    <th
                      className="px-1.5 py-1 text-right font-medium"
                      title="Tau over this column's groups (0 even, 1 in one group only)"
                    >
                      Here
                    </th>
                    <th className="px-1.5 py-1 font-medium" title="HPA tissue and cell-type categories (hover a row)">
                      HPA
                    </th>
                    <th className="px-1.5 py-1 font-medium" title="Highest tissues, nTPM">
                      Top tissues
                    </th>
                    <th className="px-1.5 py-1 font-medium" title="Highest HPA single-cell types, nCPM">
                      Top cell types
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.genes.map((g) => (
                    <tr
                      key={g.gene}
                      data-testid={`offtarget-row-${g.gene}`}
                      className={`border-b border-slate-50 align-top ${g.flag ? "bg-orange-50/60" : ""}`}
                    >
                      <td className="whitespace-nowrap px-1.5 py-0.5">
                        <button
                          type="button"
                          onClick={() => onGeneClick(g.gene)}
                          className="font-medium text-slate-700 hover:underline"
                        >
                          {g.gene}
                        </button>
                        {g.url && (
                          <a
                            href={g.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            aria-label={`${g.gene} in the Human Protein Atlas`}
                            title={`${g.hpa_symbol ?? g.gene} (${g.ensg}) in the Human Protein Atlas`}
                            className="ml-1 inline-block align-middle text-slate-400 hover:text-blue-600"
                          >
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        )}
                      </td>
                      <td className="px-1.5 py-0.5 text-right tabular-nums">
                        {g.in_dataset_tau != null ? g.in_dataset_tau.toFixed(2) : "—"}
                      </td>
                      <td className="whitespace-nowrap px-1.5 py-0.5" title={categoryTitle(g)}>
                        <span className={`rounded px-1 ${VERDICT_STYLE[g.verdict]}`}>{g.verdict_text}</span>
                        {g.flag && (
                          <span className="ml-1 font-semibold text-orange-700" title="Specific here, broad in the body">
                            !
                            <span className="sr-only"> specific here but broad in the body</span>
                          </span>
                        )}
                      </td>
                      <td className="px-1.5 py-0.5 text-slate-600">{topText(g.tissue)}</td>
                      <td className="px-1.5 py-0.5 text-slate-600">{topText(g.cell_type)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {traces.length > 0 && hm && (
            <>
              <Plot
                data={traces}
                layout={{
                  height: heatmapHeight(hm.genes.length),
                  margin: { l: 70, r: 10, t: 6, b: 100 },
                  xaxis: { tickangle: -50, tickfont: { size: 9 }, title: { text: "" } },
                  yaxis: { autorange: "reversed" as const, tickfont: { size: 10 }, title: { text: "" } },
                }}
                config={plotConfig(`hpa_tissues_${column}_${group}`)}
                useResizeHandler
                style={{ width: "100%" }}
              />
              <Caption n={[{ value: hm.genes.length, unit: "genes" }, { value: hm.tissues.length, unit: "tissues" }]}>
                RNA across human tissues, log2(nTPM + 1); brain regions merged into one brain.
              </Caption>
            </>
          )}
          <Caption n={{ value: result.summary.n_genes, unit: "markers" }}>
            Markers: detected in at least 20% of the group, adjusted p &lt; 0.05, by fold change, no ribosomal or
            mitochondrial genes. Data:{" "}
            <a
              href="https://www.proteinatlas.org"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-blue-600"
            >
              Human Protein Atlas
            </a>{" "}
            v{result.version} ({result.licence}): tissue RNA consensus (HPA + GTEx) and single-cell type RNA. Categories recomputed with HPA&rsquo;s published cut-offs,
            so a gene near a cut-off can differ from the HPA website.
          </Caption>
        </div>
      )}
    </section>
  );
}
