import { useEffect, useMemo, useState } from "react";
import { plotlyColorscale } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import Plot from "@/components/plots/Plot";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { plotConfig } from "@/lib/plotConfig";
import { Loader2, Sparkles } from "lucide-react";

import { apiFetch } from "@/api/client";
import { getQcByGroup, type QcByGroup } from "@/api/qc";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { groupScatterTraces, qcGroupOptions, qcGroupReason, QC_MAX_GROUPS } from "@/lib/qcGroups";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { Panel } from "@/components/common/Panel";
import { PlotGrid } from "@/components/common/PlotGrid";

interface QcMetric {
  min: number;
  max: number;
  mean: number;
  median: number;
  q1: number;
  q3: number;
  hist: { bin_edges: number[]; counts: number[] };
}

interface QcResponse {
  n_cells: number;
  computed_on_demand: boolean;
  /** False when the dataset has no MT- genes: % mito can't be measured (older backends omit it). */
  has_mt?: boolean;
  metrics: Record<string, QcMetric>;
  scatter: {
    x: number[];
    y: number[];
    color: number[];
    x_label: string;
    y_label: string;
    color_label: string;
    n_shown: number;
  };
}

const METRIC_LABELS: Record<string, string> = {
  n_genes_by_counts: "Genes per cell",
  total_counts: "Counts per cell",
  pct_counts_mt: "% Mitochondrial",
  doublet_score: "Doublet score",
  S_score: "Cell-cycle S score",
  G2M_score: "Cell-cycle G2M score",
  dissociation_stress_score: "Dissociation stress score",
  pct_counts_ribo: "% Ribosomal",
  pct_counts_hb: "% Hemoglobin",
};

const FILL_STYLE = { width: "100%", height: "100%" } as const;

function fmt(v: number): string {
  if (!isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
  if (Math.abs(v) >= 10) return v.toFixed(0);
  return v.toFixed(2);
}

function HistCard({ label, metric }: { label: string; metric: QcMetric }) {
  const edges = metric.hist.bin_edges;
  const counts = metric.hist.counts;
  const mids = counts.map((_, i) => (edges[i]! + edges[i + 1]!) / 2);
  const width = edges.length > 1 ? edges[1]! - edges[0]! : 1;
  return (
    <Panel title={label}>
      <div className="flex h-full flex-col">
        <div className="min-h-0 flex-1">
          <Plot
            data={[
              {
                type: "bar",
                x: mids,
                y: counts,
                width,
                marker: { color: "#60a5fa" },
                hovertemplate: "%{x:.1f}: %{y} cells<extra></extra>",
              },
            ]}
            layout={{
              autosize: true,
              margin: { l: 40, r: 10, t: 8, b: 24 },
              xaxis: { tickfont: { size: 9 } },
              yaxis: { tickfont: { size: 9 } },
              bargap: 0.02,
              showlegend: false,
              shapes: [
                {
                  type: "line",
                  x0: metric.median,
                  x1: metric.median,
                  yref: "paper",
                  y0: 0,
                  y1: 1,
                  line: { color: "#ef4444", width: 1, dash: "dot" },
                },
              ],
            }}
            config={plotConfig(`qc_${label}`)}
            style={FILL_STYLE}
            useResizeHandler
          />
        </div>
        <div className="flex-shrink-0 px-1 pt-1 text-[10px] text-slate-400">
          median {fmt(metric.median)} · IQR {fmt(metric.q1)}–{fmt(metric.q3)} · max {fmt(metric.max)}
        </div>
      </div>
    </Panel>
  );
}

function ScatterCard({ scatter }: { scatter: QcResponse["scatter"] }) {
  const colormapName = useSettingsStore((s) => s.continuousColormap);
  // No colour when the dataset has no MT- genes: every cell would be "0 %".
  const byMito = scatter.color.length > 0;
  return (
    <Panel
      title={byMito ? "Counts vs. genes per cell (colour = % mito)" : "Counts vs. genes per cell"}
      className="h-[380px]"
    >
      <Plot
        data={[
          {
            type: "scattergl",
            mode: "markers",
            x: scatter.x,
            y: scatter.y,
            marker: byMito
              ? {
                  size: 3,
                  color: scatter.color,
                  colorscale: plotlyColorscale(colormapName),
                  showscale: true,
                  opacity: 0.55,
                  colorbar: { title: { text: "% mito", font: { size: 9 } }, thickness: 8, len: 0.85 },
                }
              : { size: 3, color: "#64748b", opacity: 0.55 },
            hovertemplate: "counts %{x}<br>genes %{y}<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 54, r: 10, t: 8, b: 40 },
          xaxis: { title: { text: scatter.x_label, font: { size: 10 } }, tickfont: { size: 9 } },
          yaxis: { title: { text: scatter.y_label, font: { size: 10 } }, tickfont: { size: 9 } },
        }}
        config={plotConfig("qc_counts_vs_genes")}
        style={FILL_STYLE}
        useResizeHandler
      />
    </Panel>
  );
}

/** Violins of each QC metric per group, and the counts-vs-genes scatter coloured by
 *  group (colours shared with the embedding scatter). The backend sends each group's
 *  quantiles and a stratified sample, not every cell, so 200k cells cost no more. */
function GroupedQc({ data }: { data: QcByGroup }) {
  const column = data.column ?? "";
  const colorOf = useCategoryColors(column);
  const order = useCategoryOrder(column);
  const traces = useMemo(
    () => (data.scatter ? groupScatterTraces(data.scatter, data.groups, order, colorOf) : []),
    [data, order, colorOf],
  );
  const smallest = data.groups.reduce<QcByGroup["groups"][number] | null>(
    (min, g) => (min === null || g.n_cells < min.n_cells ? g : min),
    null,
  );
  const why = qcGroupReason(data.reason);
  return (
    <>
      <p className="text-[11px] text-slate-500">
        By <span className="font-semibold text-slate-700">{column}</span>
        {why ? ` (${why})` : ""}: {data.n_groups} groups
        {smallest ? `, smallest ${smallest.name} with ${smallest.n_cells.toLocaleString()} cells` : ""}
        {(data.n_unlabelled ?? 0) > 0
          ? `; ${data.n_unlabelled!.toLocaleString()} cells without a value are left out`
          : ""}
        .
      </p>
      {Object.entries(data.metrics).map(([k, groups]) => (
        <Panel key={k} title={`${METRIC_LABELS[k] ?? k} by ${column}`} className="h-[280px]">
          <ViolinPlot data={groups} groupColumn={column} yLabel={METRIC_LABELS[k] ?? k} height={240} />
        </Panel>
      ))}
      {data.scatter && (
        <Panel title={`Counts vs. genes per cell (colour = ${column})`} className="h-[380px]">
          <Plot
            data={traces}
            layout={{
              autosize: true,
              margin: { l: 54, r: 10, t: 8, b: 40 },
              xaxis: { title: { text: data.scatter.x_label, font: { size: 10 } }, tickfont: { size: 9 } },
              yaxis: { title: { text: data.scatter.y_label, font: { size: 10 } }, tickfont: { size: 9 } },
              showlegend: true,
              legend: { font: { size: 9 }, itemsizing: "constant" },
            }}
            config={plotConfig(`qc_counts_vs_genes_by_${column}`)}
            style={FILL_STYLE}
            useResizeHandler
          />
        </Panel>
      )}
    </>
  );
}

export function QcPlots({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [data, setData] = useState<QcResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const askCopilot = useViewStore((s) => s.askCopilot);
  const obsColumns = useDatasetStore((s) => s.currentDataset?.obs_columns);

  // Group by: null until the user picks (the backend then picks the donor, batch
  // or clustering column); "" = all cells pooled. Kept per dataset.
  const [choice, setChoice] = useState<{ datasetId: string; column: string | null }>({
    datasetId,
    column: null,
  });
  const groupBy = choice.datasetId === datasetId ? choice.column : null;
  const [grouped, setGrouped] = useState<QcByGroup | null>(null);
  const [groupLoading, setGroupLoading] = useState(false);
  const [groupError, setGroupError] = useState<string | null>(null);

  const askAboutQc = () => {
    if (!data) return;
    const parts = Object.entries(data.metrics).map(
      ([k, m]) => `${METRIC_LABELS[k] ?? k} median ${fmt(m.median)} (IQR ${fmt(m.q1)}–${fmt(m.q3)})`,
    );
    askCopilot(
      `Are these QC distributions healthy for ${data.n_cells.toLocaleString()} cells? ${parts.join(
        "; ",
      )}. What thresholds should I consider for filtering low-quality cells or doublets?`,
    );
  };

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    apiFetch<QcResponse>(`/datasets/${datasetId}/qc`)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e?.message ?? e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, refreshKey]);

  useEffect(() => {
    setGrouped(null);
    setGroupError(null);
    if (!datasetId || groupBy === "") return;
    let cancelled = false;
    setGroupLoading(true);
    getQcByGroup(datasetId, groupBy ?? undefined)
      .then((d) => {
        if (!cancelled) setGrouped(d);
      })
      .catch((e) => {
        if (!cancelled) setGroupError(String(e?.message ?? e));
      })
      .finally(() => {
        if (!cancelled) setGroupLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupBy, refreshKey]);

  const groupOptions = useMemo(() => {
    const opts = qcGroupOptions(obsColumns);
    // The backend's pick is listed even when its dtype is not one the list takes.
    const picked = grouped?.column;
    if (picked && !opts.some((o) => o.name === picked)) {
      opts.unshift({ name: picked, n: grouped!.n_groups, disabled: false });
    }
    return opts;
  }, [obsColumns, grouped]);

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white p-8 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Computing QC distributions…</span>
      </div>
    );
  }
  if (error || !data) {
    return error ? (
      <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-400">
        QC metrics unavailable: {error}
      </div>
    ) : null;
  }

  const metricKeys = Object.keys(data.metrics);
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold text-slate-700">QC Distributions</h3>
          <button
            onClick={askAboutQc}
            title="Ask the co-pilot about these QC metrics"
            className="inline-flex items-center gap-1 rounded-full border border-primary/30 px-2 py-0.5 text-[11px] font-medium text-primary hover:bg-primary/10"
          >
            <Sparkles className="h-3 w-3" />
            Ask about QC
          </button>
        </div>
        <span className="text-xs text-slate-400">
          {data.n_cells.toLocaleString()} cells
          {data.computed_on_demand ? " · computed on the fly" : ""}
          {data.scatter.n_shown < data.n_cells
            ? ` · scatter shows ${data.scatter.n_shown.toLocaleString()} sampled`
            : ""}
        </span>
      </div>
      <div className="flex items-center gap-2 text-[11px] text-slate-500">
        <label htmlFor="qc-group-by" className="font-medium">
          Group by
        </label>
        <select
          id="qc-group-by"
          value={groupBy ?? grouped?.column ?? ""}
          onChange={(e) => setChoice({ datasetId, column: e.target.value })}
          disabled={groupBy === null && groupLoading}
          className="min-w-0 max-w-[60%] rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600 disabled:opacity-50"
        >
          <option value="">None (all cells)</option>
          {groupOptions.map((o) => (
            <option key={o.name} value={o.name} disabled={o.disabled}>
              {o.name} ({o.n}
              {o.disabled ? ` groups: over ${QC_MAX_GROUPS}, pick another` : ""})
            </option>
          ))}
        </select>
        {groupLoading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      {groupError && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {groupError}
        </p>
      )}
      {data.has_mt === false && (
        <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          No mitochondrial (MT-) genes in this dataset, so % mito can&apos;t be measured: a 0 %
          reading would mean nothing, and a mito filter has nothing to remove.
        </p>
      )}
      <PlotGrid minTile={300} rowHeight={240}>
        {metricKeys.map((k) => (
          <HistCard key={k} label={METRIC_LABELS[k] ?? k} metric={data.metrics[k]!} />
        ))}
      </PlotGrid>
      {grouped?.column ? <GroupedQc data={grouped} /> : <ScatterCard scatter={data.scatter} />}
    </div>
  );
}
