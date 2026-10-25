import { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import { getCnv, getCnvHeatmap, type CnvHeatmap, type CnvSummary } from "@/api/cnv";
import { CnvRunForm } from "@/components/unified/CnvRunForm";
import { DIVERGING_COLORSCALE } from "@/lib/cellHeatmap";
import {
  chromosomeTicks,
  decodeHeatmap,
  groupTicks,
  heatmapShapes,
  patternText,
  referenceColumns,
  scoreBoxTraces,
} from "@/lib/cnv";
import { plotConfig } from "@/lib/plotConfig";
import { prettyObsLabel } from "@/lib/formatting";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { useDatasetStore } from "@/stores/datasetStore";
import { useCnvRunStore } from "@/stores/cnvRunStore";
import { useSettingsStore } from "@/stores/settingsStore";

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Copy number inferred from expression (backend core/cnv.py, inferCNV-style): pick
 * normal reference cells, run, then a cells x genome heatmap, the CNV score per group
 * and the CNV clusters. The scores and clusters are obs columns (cnv_score,
 * cnv_leiden), so the embedding colours by them like any column.
 */
export function CnvSubtab() {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const setColorBy = useSettingsStore((s) => s.setColorBy);
  const colorBy = useSettingsStore((s) => s.colorBy);
  const version = useCnvRunStore((s) => s.version);
  const refresh = useDatasetRefresh();
  const [summary, setSummary] = useState<CnvSummary | null>(null);
  const [heat, setHeat] = useState<CnvHeatmap | null>(null);
  const [rowsBy, setRowsBy] = useState("");
  const [loading, setLoading] = useState(false);
  const [heatLoading, setHeatLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const nCells = dataset?.n_cells ?? 0;
  const groupColumns = useMemo(() => (dataset ? referenceColumns(dataset.obs_columns, nCells) : []), [dataset, nCells]);
  const groupby = rowsBy && groupColumns.some((c) => c.name === rowsBy) ? rowsBy : "";

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getCnv(datasetId, groupby || undefined)
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
  }, [datasetId, groupby, version]);

  // A run saved new obs columns: list them in the colour-by pickers.
  useEffect(() => {
    if (version > 0 && datasetId) refresh(datasetId, "cnv_score").catch(() => undefined);
  }, [version, datasetId, refresh]);

  const available = summary?.available ?? false;
  const runId = summary?.run_id;
  useEffect(() => {
    if (!datasetId || !available) {
      setHeat(null);
      return;
    }
    let cancelled = false;
    setHeatLoading(true);
    getCnvHeatmap(datasetId, groupby || undefined)
      .then((h) => !cancelled && setHeat(h))
      .catch((e) => !cancelled && setError(errorText(e)))
      .finally(() => !cancelled && setHeatLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, available, runId, groupby]);

  const z = useMemo(() => (heat ? decodeHeatmap(heat) : null), [heat]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const refs = summary?.reference_groups ?? [];
  const high = (summary?.clusters ?? []).filter((c) => c.high);
  const xTicks = heat ? chromosomeTicks(heat) : null;
  const yTicks = heat ? groupTicks(heat) : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
        Copy number (inferred from expression)
        <InfoTip label="About copy-number inference" width={300}>
          A gained chromosome arm raises the expression of all its genes a little; a lost one lowers it. Averaged
          over about 100 neighbouring genes and compared with normal cells, this separates malignant cells (many
          changes) from normal ones (none). It is inferred from RNA, not measured on DNA: it needs a true normal
          reference from the same samples, low sequencing depth makes it noisy, and changes smaller than the window
          are not seen.
        </InfoTip>
      </div>

      {summary && (
        <CnvRunForm
          key={`${datasetId}|${summary.run_id ?? "none"}`}
          datasetId={datasetId}
          columns={dataset.obs_columns}
          nCells={nCells}
          runnable={summary.runnable}
          defaults={summary.defaults}
          estimate={summary.estimate}
          saved={summary.available ? summary : null}
        />
      )}
      {loading && !summary && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Loading the saved result…
        </div>
      )}
      <RequestError message={error} />

      {summary?.available && (
        <>
          <Caption n={[{ value: summary.n_cells ?? 0, unit: "cells" }, { value: summary.n_windows ?? 0, unit: "windows" }]}>
            {summary.n_genes?.toLocaleString()} genes on {summary.assembly}, relative to {refs.join(", ")} (
            {summary.n_reference_cells?.toLocaleString()} reference cells in{" "}
            {prettyObsLabel(summary.reference_column ?? "")}). Inferred from expression, not DNA: a true normal
            reference is needed and low depth is noisy.
          </Caption>
          {summary.notes?.map((n) => (
            <p key={n} className="text-[11px] text-slate-500">
              {n}
            </p>
          ))}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-slate-500">Colour the cells by</span>
            {(["cnv_score", "cnv_leiden"] as const).map((c) => (
              <button
                key={c}
                onClick={() => setColorBy(c)}
                aria-pressed={colorBy === c}
                className={`rounded border px-2 py-0.5 text-[11px] ${
                  colorBy === c ? "border-primary bg-blue-50 text-primary" : "border-slate-300 text-slate-700 hover:bg-slate-50"
                }`}
              >
                {c === "cnv_score" ? "CNV score" : "CNV cluster"}
              </button>
            ))}
          </div>

          <label className="block text-[11px] font-medium text-slate-500">
            Group the rows by
            <select
              aria-label="Group the rows by"
              value={groupby}
              onChange={(e) => setRowsBy(e.target.value)}
              className="mt-0.5 w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
            >
              <option value="">{prettyObsLabel(summary.reference_column ?? "")} (the reference column)</option>
              {groupColumns
                .filter((c) => c.name !== summary.reference_column)
                .map((c) => (
                  <option key={c.name} value={c.name}>
                    {prettyObsLabel(c.name)} ({c.n_unique} groups)
                  </option>
                ))}
            </select>
          </label>

          {heatLoading && !heat && (
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Loading the heatmap…
            </div>
          )}
          {heat && z && xTicks && yTicks && (
            <div data-testid="cnv-heatmap">
              <div style={{ height: 380 }}>
                <Plot
                  data={[
                    {
                      type: "heatmap",
                      z,
                      zmin: -heat.vmax,
                      zmax: heat.vmax,
                      colorscale: DIVERGING_COLORSCALE,
                      colorbar: { title: { text: "CNV", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
                      hovertemplate: "%{z:.3f}<extra></extra>",
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 110, r: 6, t: 4, b: 28 },
                    xaxis: { tickvals: xTicks.vals, ticktext: xTicks.text, tickfont: { size: 8 }, showgrid: false, zeroline: false },
                    yaxis: { tickvals: yTicks.vals, ticktext: yTicks.text, tickfont: { size: 9 }, autorange: "reversed", showgrid: false, zeroline: false },
                    shapes: heatmapShapes(heat) as never,
                  }}
                  config={plotConfig("cnv_heatmap")}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
              <Caption
                n={[
                  { value: heat.n_rows, unit: "cells shown", of: heat.n_cells_total },
                  { value: heat.groups.length, unit: "groups" },
                ]}
              >
                Each row a cell, each column a stretch of the genome (chromosomes 1 to X, left to right). Red: higher
                than the reference (a gain), blue: lower (a loss). Cells sampled per {prettyObsLabel(heat.column)} group
                in proportion to its size, ordered by CNV cluster.
              </Caption>
            </div>
          )}

          {summary.by_group_error && <p className="text-[11px] text-amber-700">{summary.by_group_error}</p>}
          {summary.by_group && summary.by_group.groups.length > 0 && (
            <div data-testid="cnv-score-box">
              <div style={{ height: 220 }}>
                <Plot
                  data={scoreBoxTraces(summary.by_group.groups)}
                  layout={{
                    autosize: true,
                    showlegend: false,
                    margin: { l: 46, r: 6, t: 4, b: 90 },
                    xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true },
                    yaxis: { title: { text: "CNV score", font: { size: 9 } }, tickfont: { size: 9 }, rangemode: "tozero" },
                  }}
                  config={plotConfig("cnv_score_by_group")}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
              <Caption n={{ value: summary.by_group.groups.length, unit: "groups" }}>
                CNV score (a cell&apos;s mean absolute value over the genome) per{" "}
                {prettyObsLabel(summary.by_group.column)} group; grey boxes are reference groups. Malignant cells
                score well above the reference.
              </Caption>
            </div>
          )}

          {summary.clusters && summary.clusters.length > 0 && (
            <div>
              <div className="mb-0.5 flex items-center gap-1 text-[11px] font-medium text-slate-600">
                CNV clusters
                <InfoTip label="About CNV clusters" width={260}>
                  Cells clustered on their copy-number profiles (PCA, neighbour graph, Leiden). High: mean score at
                  least {summary.defaults.high_ratio}x the reference cells&apos; and under half reference cells. A
                  chromosome is called gained or lost when its mean value in the cluster is clearly away from 0.
                </InfoTip>
              </div>
              <div className="max-h-64 overflow-auto rounded-lg border border-slate-200">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-50">
                    <tr className="border-b border-slate-200 text-slate-600">
                      <th className="px-2 py-1.5 text-left font-medium">Cluster</th>
                      <th className="px-2 py-1.5 text-right font-medium" title="Cells in the cluster">Cells</th>
                      <th className="px-2 py-1.5 text-right font-medium" title="Mean CNV score / the reference cells' mean score">
                        Score
                      </th>
                      <th className="px-2 py-1.5 text-right font-medium" title="Share of the cluster's cells in the reference groups">
                        Ref
                      </th>
                      <th className="px-2 py-1.5 text-left font-medium" title="Chromosomes whose mean value in the cluster is clearly above (gain) or below (loss) 0">
                        Pattern
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.clusters.map((c) => (
                      <tr key={c.cluster} className={`border-b border-slate-50 ${c.high ? "bg-red-50" : ""}`}>
                        <td className="px-2 py-1 font-medium text-slate-800">
                          {c.cluster}
                          {c.high && <span className="ml-1 rounded bg-red-100 px-1 text-[10px] text-red-700">high</span>}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums text-slate-600">{c.n.toLocaleString()}</td>
                        <td className="px-2 py-1 text-right tabular-nums text-slate-600">
                          {c.ratio !== null ? `${c.ratio.toFixed(1)}x` : c.mean.toFixed(3)}
                        </td>
                        <td className="px-2 py-1 text-right tabular-nums text-slate-600">{Math.round(c.reference_share * 100)}%</td>
                        <td className="px-2 py-1 text-[11px] text-slate-600">{patternText(c.gains, c.losses) || "–"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Caption n={{ value: summary.clusters.length, unit: "CNV clusters" }}>
                {high.length
                  ? `${high.length} cluster${high.length === 1 ? "" : "s"} with a high score: likely malignant cells.`
                  : "No cluster scores well above the reference: no clear sign of malignant cells."}
              </Caption>
            </div>
          )}
          {summary.method && <p className="text-[10px] leading-snug text-slate-400">{summary.method}</p>}
        </>
      )}
    </div>
  );
}
