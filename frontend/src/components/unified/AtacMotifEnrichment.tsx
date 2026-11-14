import { Fragment, useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  enrichmentCsvUrl,
  enrichmentExportUrl,
  getEnrichmentPage,
  getLatestEnrichment,
  getMotifLogo,
  runEnrichment,
  type EnrichmentRow,
  type MotifEnrichment,
  type MotifLogo,
} from "@/api/atacMotifs";
import { CapNote, Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { daTestName, formatP } from "@/lib/atac";
import { formatFold, formatPct } from "@/lib/atacMotifs";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { useAtacRunStore } from "@/stores/atacRunStore";

const PAGE = 100;

/** Enrichment of each group's differential peaks, Signac's FindMotifs. */
export function AtacMotifEnrichment({ datasetId }: { datasetId: string }) {
  const da = useAtacRunStore((s) => s.da);
  const [pColumn, setPColumn] = useState<"pval" | "pval_adj">("pval");
  const [pMax, setPMax] = useState("0.005");
  const [minPct, setMinPct] = useState("5");
  const [result, setResult] = useState<MotifEnrichment | null>(null);
  const [group, setGroup] = useState("");
  const [rows, setRows] = useState<EnrichmentRow[] | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [logo, setLogo] = useState<MotifLogo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getLatestEnrichment(datasetId)
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch(() => {
        if (!cancelled) setResult(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const shown = result && result.groups.includes(group) ? group : result?.groups[0] ?? "";
  const first = useMemo(() => result?.rows[shown] ?? [], [result, shown]);
  useEffect(() => {
    setRows(null);
    setOpen(null);
    setLogo(null);
  }, [result?.id, shown]);
  const table = rows ?? first;
  const info = result?.summary[shown];

  const run = async () => {
    if (!da) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await runEnrichment(datasetId, { da_id: da.id, p_column: pColumn, p_max: Number(pMax), min_pct: Number(minPct) / 100 }));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const more = async () => {
    if (!result) return;
    try {
      const page = await getEnrichmentPage(datasetId, result.id, shown, Math.min(1000, table.length + PAGE * 4));
      setRows(page.rows[shown] ?? []);
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const toggle = async (id: string) => {
    if (open === id) {
      setOpen(null);
      return;
    }
    setOpen(id);
    setLogo(null);
    try {
      setLogo(await getMotifLogo(datasetId, id));
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const numberInput = "w-14 rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700";
  return (
    <div className="space-y-1.5 border-t border-slate-200 pt-2" data-testid="atac-motif-enrichment">
      <h5 className="text-[11px] font-semibold text-slate-600">
        <HoverTip
          width={340}
          tip={<>
            Signac&rsquo;s FindMotifs. The query is what its vignette takes from a differential test: peaks up in the group at p &lt; 0.005 and open in
            over 20% of it. The background is 40,000 other peaks drawn to match the query&rsquo;s GC content; one-sided hypergeometric, BH across the 879
            motifs. A stored differential result keeps each group&rsquo;s 5,000 lowest p, so a query can stop there. Whether the TF itself is expressed in the
            group separates some paralogues. A motif hit is not binding, and GC-rich motifs enrich easily even after matching.
          </>}
        >
          Enriched in differential peaks
        </HoverTip>
      </h5>
      <div className="flex flex-wrap items-center gap-1.5">
        <label className="flex items-center gap-1 text-[11px] text-slate-600">
          <select value={pColumn} onChange={(e) => setPColumn(e.target.value as "pval" | "pval_adj")} aria-label="p value or FDR"
                  className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
            <option value="pval">p</option>
            <option value="pval_adj">FDR</option>
          </select>
          &lt;
          <input value={pMax} onChange={(e) => setPMax(e.target.value)} inputMode="decimal" aria-label="Largest p or FDR" className={numberInput} />
        </label>
        <label className="flex items-center gap-1 text-[11px] text-slate-600">
          <Help id="atac.motif-query">open in over</Help>
          <input value={minPct} onChange={(e) => setMinPct(e.target.value)} inputMode="decimal" aria-label="Smallest % of the group's cells" className={numberInput} />
          % of the group
        </label>
        <button type="button" disabled={!da || busy || !(Number(pMax) > 0) || !(Number(minPct) >= 0)} onClick={() => void run()}
                className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
          Test motifs
        </button>
        {busy && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {da ? (
        <Caption>Uses the differential peaks by {prettyObsLabel(da.column)} ({daTestName(da.test)}). Takes seconds and about 0.3 GB.</Caption>
      ) : (
        <Caption>Find differential peaks first (the card above).</Caption>
      )}
      <RequestError message={error} />
      {result && (
        <div className="space-y-1.5" data-testid="atac-motif-enrichment-result">
          <div className="flex flex-wrap items-center gap-1.5">
            <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
              Group
              <select value={shown} onChange={(e) => setGroup(e.target.value)}
                      className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
                {result.groups.map((g) => (
                  <option key={g} value={g}>{g} ({(result.summary[g]?.n_significant ?? 0).toLocaleString()} at FDR 5%)</option>
                ))}
              </select>
            </label>
            <a href={enrichmentCsvUrl(datasetId, result.id)} download className="text-[11px] text-primary hover:underline">CSV</a>
            <a href={enrichmentExportUrl(datasetId, result.id)} download className="text-[11px] text-primary hover:underline"
               title="The table, the method and the logo of every motif at FDR 5% in some group">Zip with logos</a>
          </div>
          {info?.note && <p className="text-[11px] leading-snug text-amber-700">{info.note}</p>}
          {table.length > 0 && (
            <div className="max-h-80 overflow-auto">
              <table className="w-full min-w-max border-separate border-spacing-x-1 text-[11px] text-slate-600">
                <thead className="sticky top-0 bg-white text-left text-slate-500">
                  <tr>
                    <th className="font-medium" title="JASPAR 2024 motif; click a row for its logo">Motif</th>
                    <th className="text-right font-medium" title="Share of the group's cells with the TF's RNA; for a dimer, its most detected member">TF in group</th>
                    <th className="text-right font-medium" title="Query peaks with a hit">Query</th>
                    <th className="text-right font-medium" title="Background peaks with a hit">Bg</th>
                    <th className="text-right font-medium" title="% of query peaks with a hit / % of background peaks with a hit">Fold</th>
                    <th className="text-right font-medium" title="Benjamini-Hochberg across the 879 motifs, within this group">FDR</th>
                    <th className="font-medium" title="Information-weighted GC over 0.7: enriches easily even after matching">GC</th>
                  </tr>
                </thead>
                <tbody>
                  {table.map((r) => (
                    <Fragment key={r.motif_id}>
                      <tr className="cursor-pointer hover:bg-slate-50" onClick={() => void toggle(r.motif_id)}>
                        <td className="whitespace-nowrap">
                          {r.motif} <span className="text-[10px] text-slate-400">{r.motif_id}</span>
                        </td>
                        <td className={`text-right tabular-nums ${r.tf_expressed ? "" : "text-slate-400"}`}>{formatPct(r.tf_pct)}</td>
                        <td className="text-right tabular-nums">{r.observed.toLocaleString()}</td>
                        <td className="text-right tabular-nums">{r.background.toLocaleString()}</td>
                        <td className="text-right tabular-nums">{formatFold(r.fold_enrichment)}</td>
                        <td className="whitespace-nowrap text-right tabular-nums">{formatP(r.pval_adj)}</td>
                        <td className="text-amber-700">{r.gc_rich ? "GC-rich" : ""}</td>
                      </tr>
                      {open === r.motif_id && (
                        <tr>
                          <td colSpan={7} className="pb-1" data-testid="atac-motif-logo">
                            {logo?.motif_id === r.motif_id ? (
                              <div className="w-0 min-w-full space-y-0.5">
                                <img src={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(logo.svg)}`} alt={`Sequence logo of ${logo.motif}, consensus ${logo.consensus}`} className="max-w-full" />
                                <Caption>
                                  {r.motif}, {logo.length} bases, consensus <span className="font-mono">{logo.consensus}</span>; letter height in bits.
                                  {r.tf_gene ? ` ${r.tf_gene} RNA in ${formatPct(r.tf_pct)} of the group's cells${r.tf_log2fc !== null ? `, log2FC ${r.tf_log2fc.toFixed(2)} against the rest` : ""}.` : " The TF is not measured here."}
                                </Caption>
                              </div>
                            ) : (
                              <Loader2 className="h-3 w-3 animate-spin text-slate-400" />
                            )}
                          </td>
                        </tr>
                      )}
                    </Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {table.length > 0 && <CapNote shown={table.length} total={result.scan.n_motifs} by="p" rest="the CSV has them all" />}
          {table.length > 0 && table.length < result.scan.n_motifs && table.length < 1000 && (
            <button type="button" onClick={() => void more()} className="rounded border border-slate-300 px-2 py-0.5 text-[11px] text-slate-700 hover:border-primary hover:text-primary">
              Show more
            </button>
          )}
          {info && (
            <Caption n={[{ value: info.n_query, unit: "query peaks" }, { value: info.n_background, unit: "background peaks" }]} testId="atac-motif-enrichment-caption">
              {shown} by {prettyObsLabel(result.column)}, {result.params.p_column === "pval_adj" ? "FDR" : "p"} &lt; {result.params.p_max}, open in over{" "}
              {Math.round(100 * (result.params.min_pct ?? 0))}% of the group, up in it, from {daTestName(result.params.da_test as "lr" | "wilcoxon" | undefined)}.{" "}
              {info.capped && "The differential result kept this group's 5,000 lowest p and more passed, so the query stops there. "}
              {info.n_left_out > 0 && `${info.n_left_out} peaks without known bases left out. `}
              {result.method}
            </Caption>
          )}
        </div>
      )}
    </div>
  );
}
