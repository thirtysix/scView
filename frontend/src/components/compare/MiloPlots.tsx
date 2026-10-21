import { useMemo } from "react";
import type { PlotMouseEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import type { MiloResult } from "@/api/milo";
import { DIVERGING_COLORSCALE } from "@/lib/cellHeatmap";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import {
  beeswarmOffsets,
  edgeSegments,
  labelOrder,
  logFcRange,
  nodeSize,
  type NhoodRow,
} from "@/lib/milo";
import { plotConfig } from "@/lib/plotConfig";

const GREY = "#cbd5e1";
const ROW_PX = 34;

interface PlotProps {
  result: MiloResult;
  rows: NhoodRow[];
  datasetName: string;
  /** A neighbourhood was clicked (its row id). */
  onPick?: (id: number) => void;
}

function hover(r: MiloResult, n: NhoodRow): string {
  const fc = n.logFC == null ? "–" : n.logFC.toFixed(2);
  const fdr = n.spatialFdr == null ? "not tested" : formatPValue(n.spatialFdr);
  const cells = r.conditions.map((k, i) => `${k} ${n.counts[i]}`).join(", ");
  const label = r.label_key
    ? `${n.label}${n.labelFraction != null ? ` (${Math.round(n.labelFraction * 100)}%)` : ""}<br>`
    : "";
  return `${label}${n.size} cells (${cells})<br>log2FC ${fc}, spatial FDR ${fdr}`;
}

function pickHandler(onPick?: (id: number) => void) {
  return (e: Readonly<PlotMouseEvent>) => {
    const id = e.points?.[0]?.customdata;
    if (onPick && typeof id === "number") onPick(id);
  };
}

/** Neighbourhood log fold changes by label (Milo's DA beeswarm): coloured by
 *  logFC where the spatial FDR passes, grey otherwise. */
export function MiloBeeswarm({ result, rows, datasetName, onPick }: PlotProps) {
  const labels = useMemo(() => labelOrder(rows), [rows]);
  const range = useMemo(() => logFcRange(rows), [rows]);
  const traces = useMemo(() => {
    const tested = rows.filter((r) => r.logFC != null);
    const xs = tested.map((r) => r.logFC!);
    const span = xs.length ? Math.max(...xs) - Math.min(...xs) : 1;
    const radius = Math.max(span, 1e-6) / 90;
    const y = new Map<number, number>();
    labels.forEach((label, li) => {
      const inRow = tested.filter((r) => r.label === label);
      const offsets = beeswarmOffsets(inRow.map((r) => r.logFC!), radius, 0.06);
      inRow.forEach((r, k) => y.set(r.id, li + offsets[k]!));
    });
    const part = (sig: boolean) => {
      const pts = tested.filter((r) => r.significant === sig);
      return {
        type: "scatter" as const,
        mode: "markers" as const,
        name: sig ? `spatial FDR < ${result.alpha}` : "not significant",
        x: pts.map((r) => r.logFC!),
        y: pts.map((r) => y.get(r.id)!),
        customdata: pts.map((r) => r.id),
        text: pts.map((r) => hover(result, r)),
        hovertemplate: "%{text}<extra></extra>",
        marker: sig
          ? {
              size: 6, color: pts.map((r) => r.logFC!), colorscale: DIVERGING_COLORSCALE,
              cmin: -range, cmax: range, line: { width: 0.3, color: "#475569" },
              colorbar: { title: { text: "log2FC", side: "right" as const }, thickness: 10, len: 0.6 },
            }
          : { size: 5, color: GREY },
      };
    };
    return [part(false), part(true)];
  }, [rows, labels, range, result]);

  const height = Math.max(260, labels.length * ROW_PX + 90);
  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height,
        margin: { t: 34, r: 16, b: 48, l: 150 },
        title: {
          text: `Neighbourhoods by ${result.label_key ? prettyObsLabel(result.label_key) : "label"}: ${result.conditions[1]} vs ${result.conditions[0]}`,
          font: { size: 13, color: "#334155" },
        },
        xaxis: {
          title: { text: `log2 fold change (${result.conditions[1]} / ${result.conditions[0]})`, font: { size: 11 } },
          zeroline: true, zerolinecolor: "#94a3b8", gridcolor: "#f1f5f9",
        },
        yaxis: {
          tickmode: "array", tickvals: labels.map((_, i) => i), ticktext: labels,
          range: [-0.6, labels.length - 0.4], gridcolor: "#f1f5f9", automargin: true,
          zeroline: false,
        },
        showlegend: false,
        hovermode: "closest",
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }}
      config={plotConfig(`milo_beeswarm_${result.label_key ?? "nhoods"}`, datasetName)}
      onClick={pickHandler(onPick)}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}

/** Milo's neighbourhood graph: a node per neighbourhood at its index cell on
 *  the embedding, sized by cells, coloured by logFC where significant; edges
 *  join neighbourhoods that share cells. */
export function MiloNhoodGraph({ result, rows, datasetName, onPick, layout }: PlotProps & { layout: string }) {
  const range = useMemo(() => logFcRange(rows), [rows]);
  const traces = useMemo(() => {
    const pos = result.layouts[layout] ?? [];
    const maxSize = Math.max(1, ...rows.map((r) => r.size));
    const edges = edgeSegments(result, layout);
    const gl = rows.length > 3000;
    // Grey first, then the significant ones with the strongest on top.
    const ordered = [...rows].sort(
      (a, b) =>
        Number(a.significant) - Number(b.significant) || Math.abs(a.logFC ?? 0) - Math.abs(b.logFC ?? 0),
    );
    const placed = ordered.filter((r) => pos[r.id]?.[0] != null && pos[r.id]?.[1] != null);
    return [
      {
        type: (gl ? "scattergl" : "scatter") as "scatter",
        mode: "lines" as const,
        x: edges.x,
        y: edges.y,
        line: { color: "rgba(100,116,139,0.18)", width: 0.6 },
        hoverinfo: "skip" as const,
        showlegend: false,
      },
      {
        type: (gl ? "scattergl" : "scatter") as "scatter",
        mode: "markers" as const,
        x: placed.map((r) => pos[r.id]![0]!),
        y: placed.map((r) => pos[r.id]![1]!),
        customdata: placed.map((r) => r.id),
        text: placed.map((r) => hover(result, r)),
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          size: placed.map((r) => nodeSize(r.size, maxSize)),
          color: placed.map((r) => (r.significant && r.logFC != null ? r.logFC : 0)),
          colorscale: DIVERGING_COLORSCALE,
          cmin: -range,
          cmax: range,
          line: { width: placed.map((r) => (r.significant ? 0.6 : 0.3)), color: placed.map((r) => (r.significant ? "#334155" : "#94a3b8")) },
          colorbar: { title: { text: "log2FC", side: "right" as const }, thickness: 10, len: 0.6 },
        },
        showlegend: false,
      },
    ];
  }, [result, rows, layout, range]);

  const axis = { showgrid: false, zeroline: false, showticklabels: false, title: { text: "" } };
  return (
    <Plot
      data={traces}
      layout={{
        autosize: true,
        height: 520,
        margin: { t: 34, r: 16, b: 16, l: 16 },
        title: {
          text: `Neighbourhood graph on ${layout.replace(/^X_/, "").toUpperCase()} (white: not significant)`,
          font: { size: 13, color: "#334155" },
        },
        xaxis: axis,
        yaxis: { ...axis, scaleanchor: "x" },
        hovermode: "closest",
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }}
      config={plotConfig(`milo_nhood_graph_${layout}`, datasetName)}
      onClick={pickHandler(onPick)}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
