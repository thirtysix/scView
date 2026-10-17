import { useCallback, useEffect, useMemo, useState } from "react";
import { plotlyColorscale } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import Plot from "@/components/plots/Plot";
import { ViolinPlot } from "@/components/plots/ViolinPlot";
import { plotConfig } from "@/lib/plotConfig";
import { Loader2, Sparkles } from "lucide-react";

import { apiFetch } from "@/api/client";
import { getQcByGroup, type FilterThresholds, type QcByGroup } from "@/api/qc";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { groupScatterTraces, qcGroupOptions, qcGroupReason, QC_MAX_GROUPS } from "@/lib/qcGroups";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";
import { Panel } from "@/components/common/Panel";
import { PlotGrid } from "@/components/common/PlotGrid";
import { DoubletCard } from "@/components/panels/DoubletCard";
import { QcFilterSummary } from "@/components/panels/QcFilterSummary";
import { ThresholdHistogram, type RefLine, type ThresholdLine } from "@/components/plots/ThresholdHistogram";
import { roundTo, thresholdMode } from "@/lib/qcThresholds";

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

/** A filtering-step threshold set on a QC histogram. */
type ThresholdParam = "min_genes" | "max_genes" | "min_counts" | "max_pct_mt";

/** Which thresholds each histogram carries. */
export const HIST_THRESHOLDS: Record<string, { param: ThresholdParam; label: string }[]> = {
  n_genes_by_counts: [
    { param: "min_genes", label: "Min" },
    { param: "max_genes", label: "Max" },
  ],
  total_counts: [{ param: "min_counts", label: "Min" }],
  pct_counts_mt: [{ param: "max_pct_mt", label: "Max" }],
};

/** A histogram's thresholds with their values. In MAD mode only the % mito ceiling:
 *  the genes / counts bounds are each sample's own (listed under the loss summary). */
export function histThresholds(metric: string, t: FilterThresholds | null) {
  const mad = t ? thresholdMode(t) === "mad" : false;
  const list = (t && HIST_THRESHOLDS[metric]) || [];
  const shown = list
    .filter((h) => !mad || h.param === "max_pct_mt")
    .map((h) => ({ ...h, label: mad ? "Ceiling" : h.label, value: t![h.param] }));
  return shown.length ? shown : undefined;
}

const THRESHOLD_STEP: Record<ThresholdParam, number> = { min_genes: 1, max_genes: 1, min_counts: 1, max_pct_mt: 0.1 };

/** 0 turns min / max genes and min counts off; max % mito has no off (100 keeps every cell). */
export function thresholdOn(param: ThresholdParam, value: number): boolean {
  return param === "max_pct_mt" ? value < 100 : value > 0;
}

/** A number input for one threshold: empty = off (min/max genes, min counts). The text
 *  is the user's while they type, and follows the value again when they leave. */
function ThresholdInput({
  param,
  label,
  value,
  onCommit,
}: {
  param: ThresholdParam;
  label: string;
  value: number;
  onCommit: (param: ThresholdParam, value: number) => void;
}) {
  const shown = thresholdOn(param, value) || param === "max_pct_mt" ? String(value) : "";
  const [text, setText] = useState<string | null>(null);
  // What was typed stays while it still says the value; a value set elsewhere (a drag
  // on the plot, which keeps the input focused) replaces it.
  const keep = text !== null && (text.trim() === "" || Number(text) === value);
  return (
    <label className="flex items-center gap-1">
      {label}
      <input
        type="number"
        min={0}
        step={THRESHOLD_STEP[param]}
        value={keep ? text! : shown}
        placeholder="off"
        aria-label={`${label} ${param}`}
        onChange={(e) => {
          setText(e.target.value);
          const raw = e.target.value.trim();
          const v = Number(raw);
          if (raw === "" && param !== "max_pct_mt") onCommit(param, 0);
          else if (raw !== "" && isFinite(v) && v >= 0) onCommit(param, v);
        }}
        onBlur={() => setText(null)}
        className="w-16 rounded border border-slate-200 px-1 py-0 text-[10px] text-slate-600"
      />
    </label>
  );
}

function HistCard({
  label,
  metric,
  thresholds,
  onDrag,
  onCommit,
}: {
  label: string;
  metric: QcMetric;
  /** The filtering thresholds this histogram sets, with their current values. */
  thresholds?: { param: ThresholdParam; label: string; value: number }[];
  onDrag?: (param: ThresholdParam, value: number) => void;
  onCommit?: (param: ThresholdParam, value: number) => void;
}) {
  const refLines = useMemo<RefLine[]>(
    () => [{ value: metric.median, label: "median", color: "#94a3b8", dash: "dot" }],
    [metric.median],
  );
  const lines: ThresholdLine[] = (thresholds ?? [])
    .filter((t) => thresholdOn(t.param, t.value))
    .map((t) => ({ id: t.param, value: t.value, label: `${t.label.toLowerCase()} ${fmt(t.value)}` }));
  return (
    <Panel title={label}>
      <div className="flex h-full flex-col">
        <div className="min-h-0 flex-1">
          <ThresholdHistogram
            edges={metric.hist.bin_edges}
            counts={metric.hist.counts}
            lines={lines}
            refLines={refLines}
            plotName={`qc_${label}`}
            onDrag={onDrag ? (id, v) => onDrag(id as ThresholdParam, v) : undefined}
            onCommit={onCommit ? (id, v) => onCommit(id as ThresholdParam, v) : undefined}
          />
        </div>
        <div className="flex flex-shrink-0 flex-wrap items-center gap-x-2 px-1 pt-1 text-[10px] text-slate-400">
          <span>
            median {fmt(metric.median)} · IQR {fmt(metric.q1)}–{fmt(metric.q3)} · max {fmt(metric.max)}
          </span>
          {thresholds && onCommit && (
            <span className="ml-auto flex items-center gap-2 text-slate-500">
              {thresholds.map((t) => (
                <ThresholdInput key={t.param} param={t.param} label={t.label} value={t.value} onCommit={onCommit} />
              ))}
            </span>
          )}
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

/** The filtering step's thresholds, owned by the caller (they are its parameters). */
export interface QcFilterControl {
  values: FilterThresholds;
  onChange: (patch: Partial<FilterThresholds>) => void;
}

export function QcPlots({
  datasetId,
  refreshKey = 0,
  onDoubletsApplied,
  filter,
}: {
  datasetId: string;
  refreshKey?: number;
  /** After new doublet calls were saved (other QC views count them). */
  onDoubletsApplied?: () => void;
  /** Set the filtering thresholds on the histograms, with a live cell-loss summary. */
  filter?: QcFilterControl;
}) {
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

  // A threshold being dragged, before the drag ends and it is handed to `filter`.
  const [live, setLive] = useState<Partial<FilterThresholds>>({});
  const thresholds = useMemo(() => (filter ? { ...filter.values, ...live } : null), [filter, live]);
  const onFilterChange = filter?.onChange;
  const dragThreshold = useCallback(
    (param: ThresholdParam, v: number) => setLive((l) => ({ ...l, [param]: roundTo(v, THRESHOLD_STEP[param]) })),
    [],
  );
  const commitThreshold = useCallback(
    (param: ThresholdParam, v: number) => {
      setLive((l) => {
        const next = { ...l };
        delete next[param];
        return next;
      });
      onFilterChange?.({ [param]: roundTo(v, THRESHOLD_STEP[param]) });
    },
    [onFilterChange],
  );

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

  // The doublet score has a card of its own, with its threshold.
  const metricKeys = Object.keys(data.metrics).filter((k) => k !== "doublet_score");
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
          <HistCard
            key={k}
            label={METRIC_LABELS[k] ?? k}
            metric={data.metrics[k]!}
            thresholds={histThresholds(k, thresholds)}
            onDrag={thresholds ? dragThreshold : undefined}
            onCommit={thresholds ? commitThreshold : undefined}
          />
        ))}
      </PlotGrid>
      {filter && thresholds && (
        <QcFilterSummary
          datasetId={datasetId}
          thresholds={thresholds}
          refreshKey={refreshKey}
          onChange={filter.onChange}
        />
      )}
      {data.metrics.doublet_score && (
        <DoubletCard datasetId={datasetId} refreshKey={refreshKey} onApplied={onDoubletsApplied} />
      )}
      {grouped?.column ? <GroupedQc data={grouped} /> : <ScatterCard scatter={data.scatter} />}
    </div>
  );
}
