import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { Loader2 } from "lucide-react";

import { apiFetch } from "@/api/client";

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
  pct_counts_ribo: "% Ribosomal",
  pct_counts_hb: "% Hemoglobin",
};

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
    <div className="rounded-lg border border-slate-100">
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
          title: { text: label, font: { size: 11 } },
          height: 160,
          margin: { l: 36, r: 8, t: 22, b: 22 },
          xaxis: { tickfont: { size: 8 } },
          yaxis: { tickfont: { size: 8 } },
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
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
      <div className="px-2 pb-1 text-[10px] text-slate-400">
        median {fmt(metric.median)} · IQR {fmt(metric.q1)}–{fmt(metric.q3)} · max {fmt(metric.max)}
      </div>
    </div>
  );
}

function ScatterCard({ scatter }: { scatter: QcResponse["scatter"] }) {
  return (
    <div className="rounded-lg border border-slate-100">
      <Plot
        data={[
          {
            type: "scattergl",
            mode: "markers",
            x: scatter.x,
            y: scatter.y,
            marker: {
              size: 3,
              color: scatter.color,
              colorscale: "Viridis",
              showscale: true,
              opacity: 0.55,
              colorbar: { title: { text: "% mito", font: { size: 9 } }, thickness: 8, len: 0.85 },
            },
            hovertemplate: "counts %{x}<br>genes %{y}<extra></extra>",
          },
        ]}
        layout={{
          title: { text: "Counts vs. genes per cell (colour = % mito)", font: { size: 11 } },
          height: 300,
          margin: { l: 52, r: 8, t: 24, b: 38 },
          xaxis: { title: { text: scatter.x_label, font: { size: 9 } }, tickfont: { size: 8 } },
          yaxis: { title: { text: scatter.y_label, font: { size: 9 } }, tickfont: { size: 8 } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

export function QcPlots({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [data, setData] = useState<QcResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">QC Distributions</h3>
        <span className="text-xs text-slate-400">
          {data.n_cells.toLocaleString()} cells
          {data.computed_on_demand ? " · computed on the fly" : ""}
          {data.scatter.n_shown < data.n_cells
            ? ` · scatter shows ${data.scatter.n_shown.toLocaleString()} sampled`
            : ""}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        {metricKeys.map((k) => (
          <HistCard key={k} label={METRIC_LABELS[k] ?? k} metric={data.metrics[k]!} />
        ))}
      </div>
      <ScatterCard scatter={data.scatter} />
    </div>
  );
}
