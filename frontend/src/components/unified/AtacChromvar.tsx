import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import { chromVarCsvUrl, getChromVarGroups, getChromVarMotif, type ChromVarGroups, type ChromVarMotif, type MotifState } from "@/api/atacMotifs";
import { CapNote, Caption } from "@/components/common/Caption";
import { Help } from "@/components/common/Help";
import { HoverTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import Plot from "@/components/plots/Plot";
import { formatP } from "@/lib/atac";
import { chromvarEstimateText } from "@/lib/atacMotifs";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";
import { useAtacRunStore } from "@/stores/atacRunStore";
import { AtacRunStatus } from "./AtacRunStatus";

const RANKED = 25;

/** chromVAR per-cell motif activity: motifs by variability, groups x motifs, group against the rest, a motif on the plot. */
export function AtacChromvar({
  datasetId,
  state,
  groupby,
  activeMotif,
  onMotif,
}: {
  datasetId: string;
  state: MotifState;
  groupby: string;
  activeMotif: string | null;
  onMotif: (m: ChromVarMotif) => void;
}) {
  const run = useAtacRunStore();
  const live = run.datasetId === datasetId && run.running;
  const [groups, setGroups] = useState<ChromVarGroups | null>(null);
  const [group, setGroup] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const chromvar = state.chromvar ?? null;
  const runId = chromvar?.run_id;

  useEffect(() => {
    let cancelled = false;
    setGroups(null);
    if (!runId || !groupby) return;
    setBusy("Comparing the groups…");
    getChromVarGroups(datasetId, groupby)
      .then((g) => {
        if (!cancelled) setGroups(g);
      })
      .catch((e) => {
        if (!cancelled) setError(errorMessage(e));
      })
      .finally(() => {
        if (!cancelled) setBusy(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, runId]);

  const pick = async (id: string) => {
    setBusy("Reading its activity…");
    setError(null);
    try {
      onMotif(await getChromVarMotif(datasetId, id, groupby || undefined));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(null);
    }
  };

  const shown = groups && groups.groups.includes(group) ? group : groups?.groups[0] ?? "";
  const rows = groups?.rows[shown] ?? [];
  const heat = groups?.heatmap;
  const zLimit = Math.max(1e-6, ...(heat?.mean_z.flat().map((v) => Math.abs(v ?? 0)) ?? [])); // symmetric: 0 is the midpoint colour
  return (
    <div className="space-y-1.5 border-t border-slate-200 pt-2" data-testid="atac-chromvar">
      <h5 className="text-[11px] font-semibold text-slate-600">
        <HoverTip
          width={340}
          tip={<>
            How far each cell&rsquo;s counts in a motif&rsquo;s peaks are from what its depth predicts, against 50 sets of peaks matched on total counts and GC
            (chromVAR&rsquo;s computeDeviations, the same scores Signac&rsquo;s RunChromVAR gave). The score is relative to those peaks across these cells: it is not a
            measure of binding or of the TF&rsquo;s amount, and paralogues that share a motif move together.
          </>}
        >
          Activity per cell (chromVAR)
        </HoverTip>
      </h5>
      {!live && (
        <div className="flex flex-wrap items-center gap-1.5">
          <button type="button" disabled={run.running || !!state.running} onClick={() => void run.runChromvar(datasetId, {})}
                  className="rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50">
            {chromvar ? "Run chromVAR again" : "Run chromVAR"}
          </button>
          <span className="text-[11px] leading-snug text-slate-500" data-testid="atac-chromvar-estimate">{chromvarEstimateText(state)}</span>
        </div>
      )}
      <AtacRunStatus datasetId={datasetId} kind="chromvar" />
      <RequestError message={error} />
      {chromvar && (
        <div className="space-y-1.5" data-testid="atac-chromvar-result">
          {chromvar.stale && (
            <p className="text-[11px] leading-snug text-amber-700">The dataset&rsquo;s cells changed after this run, so these scores read as stale: run chromVAR again.</p>
          )}
          {busy && (
            <p className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <Loader2 className="h-3 w-3 animate-spin" /> {busy}
            </p>
          )}
          {!groupby && <Caption>Choose Group by at the top of this tab to compare groups.</Caption>}
          {heat && groups && (
            <div data-testid="atac-chromvar-heatmap">
              <Plot
                data={[{
                  type: "heatmap" as const, x: groups.groups, y: heat.motifs, z: heat.mean_z as number[][],
                  colorscale: [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]], zmin: -zLimit, zmax: zLimit, xgap: 1, ygap: 1,
                  colorbar: { title: { text: "mean z", font: { size: 10 } }, thickness: 10 },
                  hovertemplate: "%{y}<br>%{x}: mean z %{z:.2f}<extra></extra>",
                }]}
                layout={{
                  height: 150 + 13 * heat.motifs.length, margin: { t: 6, r: 8, b: 60, l: 110 },
                  xaxis: { type: "category" as const, tickfont: { size: 10, color: "#64748b" }, automargin: true },
                  yaxis: { type: "category" as const, autorange: "reversed" as const, tickfont: { size: 9, color: "#64748b" }, automargin: true },
                  paper_bgcolor: "white", plot_bgcolor: "white", font: { family: "Inter, system-ui, sans-serif" },
                }}
                config={plotConfig("atac_chromvar_groups_by_motifs")}
                useResizeHandler
                style={{ width: "100%" }}
                onClick={(e: { points?: { y?: unknown }[] }) => {
                  const name = e.points?.[0]?.y;
                  const at = heat.motifs.findIndex((m) => m === name);
                  if (at >= 0) void pick(heat.motif_ids[at]!);
                }}
              />
              <Caption n={[{ value: groups.n_cells, unit: "cells" }, { value: heat.motifs.length, unit: "motifs" }, { value: groups.groups.length, unit: "groups" }]}>
                <Help id="atac.chromvar-heatmap">
                  Mean z per {prettyObsLabel(groups.column)} of the {heat.motifs.length} most variable of 879 motifs.
                </Help>{" "}
                Click a motif for its activity on the plot.
              </Caption>
            </div>
          )}
          {groups && (
            <div className="space-y-1">
              <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
                <Help id="atac.chromvar-vs-rest">Group against the rest</Help>
                <select value={shown} onChange={(e) => setGroup(e.target.value)}
                        className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700">
                  {groups.groups.map((g) => <option key={g} value={g}>{g}</option>)}
                </select>
              </label>
              <div className="max-h-64 overflow-auto">
                <table className="w-full min-w-max border-separate border-spacing-x-1 text-[11px] text-slate-600">
                  <thead className="sticky top-0 bg-white text-left text-slate-500">
                    <tr>
                      <th className="font-medium" title="JASPAR 2024 motif; click a row to colour the plot by its activity">Motif</th>
                      <th className="text-right font-medium" title="Mean z in the group">Mean z</th>
                      <th className="text-right font-medium" title="Mean z in the group minus mean z in the other cells">Difference</th>
                      <th className="text-right font-medium" title="Wilcoxon rank-sum, BH across motifs within this group">FDR</th>
                      <th className="font-medium" title="Information-weighted GC over 0.7">GC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.motif_id} className={`cursor-pointer hover:bg-slate-50 ${activeMotif === r.motif_id ? "bg-slate-100" : ""}`} onClick={() => void pick(r.motif_id)}>
                        <td className="whitespace-nowrap">{r.motif} <span className="text-[10px] text-slate-400">{r.motif_id}</span></td>
                        <td className="text-right tabular-nums">{r.mean_z.toFixed(2)}</td>
                        <td className="text-right tabular-nums">{r.mean_diff.toFixed(2)}</td>
                        <td className="whitespace-nowrap text-right tabular-nums">{formatP(r.pval_adj)}</td>
                        <td className="text-amber-700">{r.gc_rich ? "GC-rich" : ""}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <CapNote shown={rows.length} total={879} by="difference" />
            </div>
          )}
          <div className="space-y-1">
            <div className="flex items-center gap-1.5 text-[11px] font-medium text-slate-600">
              <Help id="atac.chromvar-variable">Most variable motifs</Help>
              <a href={chromVarCsvUrl(datasetId)} download className="font-normal text-primary hover:underline">CSV</a>
            </div>
            <div className="max-h-52 overflow-auto">
              <table className="w-full min-w-max border-separate border-spacing-x-1 text-[11px] text-slate-600">
                <thead className="sticky top-0 bg-white text-left text-slate-500">
                  <tr>
                    <th className="font-medium" title="JASPAR 2024 motif; click a row to colour the plot by its activity">Motif</th>
                    <th className="text-right font-medium" title="sd of z over the cells (chromVAR's computeVariability)">Variability</th>
                    <th className="text-right font-medium" title="Peaks with a hit">Peaks</th>
                    <th className="font-medium" title="Information-weighted GC over 0.7">GC</th>
                  </tr>
                </thead>
                <tbody>
                  {chromvar.ranking.slice(0, RANKED).map((r) => (
                    <tr key={r.motif_id} className={`cursor-pointer hover:bg-slate-50 ${activeMotif === r.motif_id ? "bg-slate-100" : ""}`} onClick={() => void pick(r.motif_id)}>
                      <td className="whitespace-nowrap">{r.motif} <span className="text-[10px] text-slate-400">{r.motif_id}</span></td>
                      <td className="text-right tabular-nums">{r.variability.toFixed(2)}</td>
                      <td className="text-right tabular-nums">{r.n_hits.toLocaleString()}</td>
                      <td className="text-amber-700">{r.gc_rich ? "GC-rich" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <CapNote shown={Math.min(RANKED, chromvar.ranking.length)} total={879} by="variability" rest="the CSV has them all" />
            <Caption n={[{ value: chromvar.n_cells, unit: "cells" }, { value: chromvar.n_peaks_used, unit: "peaks" }]}>
              {chromvar.method}
              {chromvar.n_peaks_no_counts > 0 && ` ${chromvar.n_peaks_no_counts.toLocaleString()} peaks with no count in these cells left out.`}
              {chromvar.n_peaks_left_out > 0 && ` ${chromvar.n_peaks_left_out.toLocaleString()} peaks without known bases left out.`}
            </Caption>
          </div>
        </div>
      )}
    </div>
  );
}
