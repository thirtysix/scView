import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import type { PlotMouseEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import {
  enrichProgram,
  getProgramUsage,
  getPrograms,
  type GeneProgram,
  type ProgramEnrichment,
  type ProgramsSummary,
  type ProgramUsage,
} from "@/api/programs";
import { ProgramsEnrichment } from "@/components/unified/ProgramsEnrichment";
import { ProgramsRunForm } from "@/components/unified/ProgramsRunForm";
import {
  errorText,
  heatmapData,
  pctText,
  prettyTerm,
  stabilityBadge,
  stabilityText,
  sweepTraces,
  topGroup,
} from "@/lib/programs";
import { plotConfig } from "@/lib/plotConfig";
import { prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";
import { useProgramsRunStore } from "@/stores/programsRunStore";
import { RequestError } from "@/components/common/RequestError";

interface Props {
  /** Colour the embedding by one program's usage share (and fill the violin). */
  onUsage: (usage: ProgramUsage) => void;
  /** The program on the scatter now (1-based), if any. */
  activeIndex: number | null;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

const GENE_CHIPS = 6;
// Sequential, white at 0: a usage share is never negative.
const USAGE_SCALE: [number, string][] = [
  [0, "#f8fafc"],
  [0.5, "#93c5fd"],
  [1, "#1e3a8a"],
];

/**
 * Gene programs (consensus NMF, backend core/gene_programs.py): run them, list them
 * with their stability and top genes, show one's usage on the embedding, a program x
 * group heatmap of the mean usage share, and a program's Hallmark enrichment.
 */
export function ProgramsSubtab({ onUsage, activeIndex, groupByColumn, setGroupByColumn }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const version = useProgramsRunStore((s) => s.version);
  const liveK = useProgramsRunStore((s) => (s.datasetId === datasetId && s.running ? s.kResults : null));
  const [summary, setSummary] = useState<ProgramsSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showing, setShowing] = useState<number | null>(null);
  const [enriching, setEnriching] = useState<number | null>(null);
  const [enrichment, setEnrichment] = useState<ProgramEnrichment | null>(null);
  const [reload, setReload] = useState(0);

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 1 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const groupby = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : "";

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPrograms(datasetId, groupby || undefined)
      .then((s) => !cancelled && setSummary(s))
      .catch((e) => {
        if (!cancelled) {
          setSummary(null);
          setError(errorText(e));
        }
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, version, reload]);

  // A new run's programs are other programs: their enrichment is not this one.
  useEffect(() => setEnrichment(null), [datasetId, summary?.run_id]);

  const show = useCallback(
    async (index: number) => {
      if (!datasetId || showing !== null) return;
      setShowing(index);
      try {
        onUsage(await getProgramUsage(datasetId, index, groupby || undefined));
      } catch (e) {
        setError(errorText(e));
      } finally {
        setShowing(null);
      }
    },
    [datasetId, groupby, onUsage, showing],
  );

  const enrich = useCallback(
    async (index: number) => {
      if (!datasetId || enriching !== null) return;
      setEnriching(index);
      try {
        setEnrichment(await enrichProgram(datasetId, index));
        setReload((r) => r + 1); // its best term now shows in the list
      } catch (e) {
        setError(errorText(e));
      } finally {
        setEnriching(null);
      }
    },
    [datasetId, enriching],
  );

  const programs: GeneProgram[] = useMemo(() => summary?.programs ?? [], [summary]);
  const nSeeds = summary?.seeds?.length ?? 0;
  const heat = useMemo(
    () => (summary?.by_group && programs.length ? heatmapData(summary.by_group, programs) : null),
    [summary, programs],
  );
  const sweep = liveK && liveK.length > 1 ? liveK : (summary?.sweep ?? []);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
        Gene programs (NMF)
        <InfoTip label="About gene programs" width={300}>
          Clusters put each cell in one group; an interferon response, the cell cycle or stress run across
          groups. NMF writes each cell&apos;s variable-gene expression as a weighted sum of k non-negative
          programs: a program is a set of gene weights, and each cell gets a usage of each. Usage share: the
          part of the cell&apos;s modelled expression a program accounts for (a cell&apos;s shares sum to 1).
          Repeating NMF from several random starts shows which programs come back every time (stable).
          A program can be a cell type as well as a process, and k is a choice, not a fact.
        </InfoTip>
      </div>

      {summary && (
        <ProgramsRunForm
          datasetId={datasetId}
          runnable={summary.runnable}
          defaults={summary.defaults}
          hasResult={summary.available}
        />
      )}
      {loading && !summary && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading the saved programs…
        </div>
      )}
      <RequestError message={error} />

      {sweep.length > 1 && (
        <div data-testid="programs-sweep">
          <div className="mb-0.5 text-[11px] font-medium text-slate-600">Compare k</div>
          <div style={{ height: 170 }}>
            <Plot
              data={sweepTraces(sweep)}
              layout={{
                autosize: true,
                margin: { l: 42, r: 46, t: 6, b: 30 },
                showlegend: true,
                legend: { orientation: "h", y: 1.15, font: { size: 9 } },
                xaxis: { title: { text: "k", font: { size: 10 } }, tickfont: { size: 9 }, dtick: 5 },
                yaxis: { title: { text: "silhouette", font: { size: 9 } }, tickfont: { size: 9 } },
                yaxis2: { overlaying: "y", side: "right", tickfont: { size: 9 }, title: { text: "rel. error", font: { size: 9 } } },
              }}
              config={plotConfig("programs_k_sweep")}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </div>
          <Caption>
            For each k: the silhouette of the k-means grouping of the runs&apos; programs (higher: the runs agree
            on the programs) and the relative reconstruction error, ‖X − WH‖² / ‖X‖² (it always falls as k
            grows). {sweep.map((s) => `k = ${s.k}: ${s.n_stable} of ${s.k} stable`).join("; ")}.
          </Caption>
        </div>
      )}

      {summary?.available && (
        <>
          <Caption n={{ value: summary.n_cells ?? 0, unit: "cells" }}>
            {summary.k} programs from {nSeeds} NMF run{nSeeds === 1 ? "" : "s"} on{" "}
            {summary.n_genes?.toLocaleString()} variable genes ({summary.source});{" "}
            {programs.filter((p) => p.stable).length} stable (found in every run, spectra correlating ≥{" "}
            {summary.defaults.stable_correlation}). Click a program to colour the cells by its usage share.
          </Caption>
          {summary.notes?.map((n) => (
            <p key={n} className="text-[11px] text-slate-500">
              {n}
            </p>
          ))}
          <div className="max-h-80 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-slate-50">
                <tr className="border-b border-slate-200 text-slate-600">
                  <th className="px-2 py-1.5 text-left font-medium" title="P1 is the program with the highest mean usage share">
                    Program
                  </th>
                  <th
                    className="px-2 py-1.5 text-left font-medium"
                    title="Highest gene scores: the least-squares coefficient of the gene's z-scored expression on the usage shares (the z-score it is predicted to have in a cell made entirely of this program)"
                  >
                    Top genes
                  </th>
                  <th
                    className="px-2 py-1.5 text-right font-medium"
                    title="Share of the cells where this program has the largest usage share"
                  >
                    Top in
                  </th>
                  <th className="px-2 py-1.5 text-right font-medium" title="Mean usage share over all cells">
                    Mean
                  </th>
                  <th className="px-2 py-1.5 text-left font-medium" title="Over-representation of its top genes in the MSigDB Hallmark sets">
                    Hallmark
                  </th>
                </tr>
              </thead>
              <tbody>
                {programs.map((p) => {
                  const where = summary.by_group ? topGroup(summary.by_group, p.index) : null;
                  return (
                    <tr
                      key={p.index}
                      onClick={() => show(p.index)}
                      className={`cursor-pointer border-b border-slate-50 align-top hover:bg-blue-50 ${
                        activeIndex === p.index ? "bg-blue-50" : ""
                      }`}
                      data-testid={`program-row-${p.index}`}
                    >
                      <td className="whitespace-nowrap px-2 py-1.5">
                        <span className="font-medium text-slate-800">{p.name}</span>
                        {showing === p.index && <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />}
                        <div
                          className={`mt-0.5 w-fit rounded px-1 text-[10px] ${
                            p.stable ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
                          }`}
                          title={stabilityText(p, nSeeds, summary.defaults.stable_correlation)}
                        >
                          {stabilityBadge(p, nSeeds)}
                        </div>
                      </td>
                      <td className="px-2 py-1.5">
                        <div className="flex flex-wrap gap-0.5">
                          {p.top_genes.slice(0, GENE_CHIPS).map((g) => (
                            <span
                              key={g.gene}
                              className="rounded bg-slate-100 px-1 font-mono text-[10px] text-slate-700"
                              title={`gene score ${g.score.toFixed(2)}`}
                            >
                              {g.gene}
                            </span>
                          ))}
                        </div>
                        {where && (
                          <div className="mt-0.5 text-[10px] text-slate-400">
                            highest in {where.group} ({pctText(where.mean)})
                          </div>
                        )}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">{pctText(p.pct_top)}</td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">{pctText(p.mean_usage)}</td>
                      <td className="px-2 py-1.5">
                        {p.top_term ? (
                          <span className="text-[10px] text-slate-600" title={`adjusted p ${p.top_term.adjusted_pvalue.toExponential(1)}`}>
                            {prettyTerm(p.top_term.term)}
                          </span>
                        ) : (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              enrich(p.index);
                            }}
                            disabled={enriching !== null}
                            className="text-[10px] font-medium text-primary hover:underline disabled:opacity-50"
                          >
                            {enriching === p.index ? "…" : "Enrich"}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {enrichment && <ProgramsEnrichment result={enrichment} />}

          <div>
            <label className="mb-1 block text-[11px] font-medium text-slate-500">Group by (heatmap)</label>
            <select
              value={groupby}
              onChange={(e) => setGroupByColumn(e.target.value)}
              className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
            >
              <option value="">None</option>
              {groupColumns.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)} ({c.n_unique} groups)
                </option>
              ))}
            </select>
          </div>
          {summary.by_group_error && <p className="text-[11px] text-amber-700">{summary.by_group_error}</p>}
          {heat && summary.by_group && (
            <div data-testid="programs-heatmap">
              <div style={{ height: Math.max(220, heat.y.length * 18 + 110) }}>
                <Plot
                  data={[
                    {
                      type: "heatmap",
                      x: heat.x,
                      y: heat.y,
                      z: heat.z,
                      zmin: 0,
                      zmax: heat.zmax,
                      colorscale: USAGE_SCALE,
                      colorbar: { title: { text: "share", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
                      hovertemplate: "%{y} in %{x}<br>mean usage share %{z:.3f}<extra></extra>",
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 120, r: 6, t: 4, b: 90 },
                    xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true },
                    yaxis: { type: "category", tickfont: { size: 9 }, automargin: true, autorange: "reversed" },
                  }}
                  config={plotConfig("programs_by_group")}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                  onClick={(e: Readonly<PlotMouseEvent>) => {
                    const row = e.points[0]?.pointIndex;
                    const r = Array.isArray(row) ? row[0] : undefined;
                    const index = typeof r === "number" ? heat.index[r] : undefined;
                    if (index !== undefined) show(index);
                  }}
                />
              </div>
              <Caption
                n={[
                  { value: summary.by_group.groups.length, unit: "groups" },
                  { value: summary.by_group.n_cells.reduce((a, b) => a + b, 0), unit: "cells with a label" },
                ]}
              >
                Mean usage share of each program in each {prettyObsLabel(summary.by_group.column)} group. Click a
                row to colour the cells by it.
              </Caption>
            </div>
          )}
          {summary.method && <p className="text-[10px] leading-snug text-slate-400">{summary.method}</p>}
        </>
      )}
    </div>
  );
}
