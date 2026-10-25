/**
 * Tracks plot (C121, scanpy's `sc.pl.tracksplot`): each gene's per-cell
 * expression as a filled line, one row per gene on a shared x of cells, the
 * cells ordered by group (the column's stored order) and the group's colour
 * under each stretch, with a band of group colours at the bottom. Data: the
 * per-cell heatmap's seeded, stratified sample (GET /expression/heatmap-cells,
 * scale "none"), so a large group does not fill the width.
 */
import { discreteColorscale, groupSpans, type HeatmapCells } from "@/lib/cellHeatmap";

/** Most genes drawn as tracks (rows get too thin past this). */
export const TRACKS_MAX_GENES = 30;

/** How cells sit within their group: as in the dataset, or by one gene's value (rising). */
export type TrackOrder = { by: "dataset" } | { by: "gene"; gene: number };

const TRACK_PX = 44;
const BAND_PX = 14;
const GAP_PX = 4;
const MARGIN = { l: 90, r: 50, t: 20, b: 90 };

/**
 * The plot order of the sample's columns: grouped as served, and within each
 * group in dataset order or by the chosen gene's value, lowest first (ties keep
 * dataset order). Returns indices into the sample.
 */
export function trackOrder(data: HeatmapCells, order: TrackOrder): number[] {
  const idx = Array.from({ length: data.codes.length }, (_, i) => i);
  if (order.by === "dataset") return idx;
  const v = data.values[order.gene];
  if (!v) return idx;
  const out: number[] = [];
  for (const span of groupSpans(data.codes)) {
    const part = idx.slice(span.start, span.end);
    part.sort((a, b) => v[a]! - v[b]! || a - b);
    out.push(...part);
  }
  return out;
}

/** "1.2", "35" or "1,250": the highest value, for a track's scale. */
function fmtMax(v: number): string {
  return v >= 100 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v.toFixed(1);
}

/** A colour with alpha, from "rgb(r, g, b)" or "#rrggbb". */
export function withAlpha(color: string, alpha: number): string {
  const m = color.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
  if (m) return `rgba(${m[1]}, ${m[2]}, ${m[3]}, ${alpha})`;
  const h = color.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
  if (h) return `rgba(${parseInt(h[1]!, 16)}, ${parseInt(h[2]!, 16)}, ${parseInt(h[3]!, 16)}, ${alpha})`;
  return color;
}

/**
 * Plotly traces + layout: per gene, one filled trace per group stretch (in the
 * group's colour) on its own y axis, 0 to the gene's highest value in the
 * sample (each gene on its own scale, as scanpy); the group band underneath
 * on the last axis; white lines at the group boundaries; group names under
 * their cells. `groupColors` is in meta.groups order.
 */
export function tracksFigure(
  data: HeatmapCells,
  groupColors: readonly string[],
  order: TrackOrder = { by: "dataset" },
): { data: object[]; layout: Record<string, unknown> } {
  const { meta, codes } = data;
  const genes = meta.genes.slice(0, TRACKS_MAX_GENES);
  const cols = trackOrder(data, order);
  const n = cols.length;
  const spans = groupSpans(cols.map((c) => codes[c]!));
  const plotPx = genes.length * TRACK_PX + GAP_PX + BAND_PX;
  const bandTop = BAND_PX / plotPx;
  const rowH = (1 - bandTop - GAP_PX / plotPx) / Math.max(genes.length, 1);
  const traces: object[] = [];
  const layout: Record<string, unknown> = {};
  genes.forEach((gene, i) => {
    const v = data.values[i]!;
    const axis = i === 0 ? "" : String(i + 1);
    let hi = 0;
    for (const c of cols) hi = Math.max(hi, v[c]!);
    // One filled line per group stretch, in the group's colour (scanpy's
    // fill_between): a few SVG paths per gene, where a bar per cell would be
    // thousands of shapes.
    for (const s of spans) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (let k = s.start; k < s.end; k++) {
        xs.push(k);
        ys.push(v[cols[k]!]!);
      }
      const color = groupColors[s.group] ?? "#94a3b8";
      traces.push({
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        fill: "tozeroy",
        line: { color, width: 0.5 },
        fillcolor: withAlpha(color, 0.8),
        name: meta.groups[s.group] ?? "",
        showlegend: false,
        hovertemplate: `<b>${gene}</b> %{y:.2f}<br>${meta.groupby}: ${meta.groups[s.group] ?? ""}<extra></extra>`,
        xaxis: "x",
        yaxis: `y${axis}`,
      });
    }
    // Rows from the top: gene 0 highest.
    const top = 1 - i * rowH;
    layout[`yaxis${axis}`] = {
      domain: [top - rowH * 0.92, top],
      range: [0, hi > 0 ? hi * 1.05 : 1],
      fixedrange: true,
      showgrid: false,
      zeroline: false,
      side: "right",
      tickvals: [hi > 0 ? hi : 1],
      ticktext: [hi > 0 ? fmtMax(hi) : "0"],
      tickfont: { size: 9 },
      title: { text: "" },
    };
  });
  const bandAxis = String(genes.length + 1);
  traces.push({
    type: "heatmap",
    x: Array.from({ length: n }, (_, k) => k),
    y: [meta.groupby],
    z: [cols.map((c) => codes[c]!)],
    zmin: -0.5,
    zmax: meta.groups.length - 0.5,
    colorscale: discreteColorscale(groupColors),
    showscale: false,
    text: [cols.map((c) => meta.groups[codes[c]!] ?? "")],
    hovertemplate: `${meta.groupby}: %{text}<extra></extra>`,
    xaxis: "x",
    yaxis: `y${bandAxis}`,
  });
  layout[`yaxis${bandAxis}`] = {
    domain: [0, bandTop],
    type: "category",
    fixedrange: true,
    showticklabels: false,
  };
  const annotations = genes.map((gene, i) => ({
    xref: "paper",
    yref: "paper",
    x: 0,
    xanchor: "right",
    y: 1 - i * rowH - (rowH * 0.92) / 2,
    yanchor: "middle",
    text: gene,
    showarrow: false,
    xshift: -6,
    font: { size: 11 },
  }));
  const boundaries = spans.slice(1).map((s) => ({
    type: "line",
    xref: "x",
    yref: "paper",
    x0: s.start - 0.5,
    x1: s.start - 0.5,
    y0: 0,
    y1: 1,
    line: { color: "#ffffff", width: 1.5 },
  }));
  return {
    data: traces,
    layout: {
      ...layout,
      height: plotPx + MARGIN.t + MARGIN.b,
      margin: MARGIN,
      xaxis: {
        // The group names sit under the band, not under the first track.
        anchor: `y${bandAxis}`,
        range: [-0.5, n - 0.5],
        tickmode: "array",
        tickvals: spans.map((s) => (s.start + s.end - 1) / 2),
        ticktext: spans.map((s) => meta.groups[s.group] ?? ""),
        ticks: "",
        showgrid: false,
        zeroline: false,
        automargin: true,
      },
      annotations,
      shapes: boundaries,
      showlegend: false,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    },
  };
}

/** The scale caption: each gene's track from 0 to its highest value in the sample. */
export function tracksCaption(order: TrackOrder, genes: string[]): string {
  const within = order.by === "gene" ? `by ${genes[order.gene] ?? "the gene"} (lowest first)` : "in dataset order";
  return `Each track: one gene's value per cell, 0 to its highest in these cells (the number on the right). Cells within a group ${within}.`;
}
