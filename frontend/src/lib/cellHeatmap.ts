/**
 * The marker heatmap's "Per cell" view (C034): genes x a stratified, seeded
 * sample of cells from GET /expression/heatmap-cells, grouped in the column's
 * stored order, with a strip of group colours above.
 */
import { tableFromIPC } from "apache-arrow";

/** Most genes the endpoint takes. */
export const HEATMAP_MAX_GENES = 100;
/** Choices for "cells per group". */
export const CELLS_PER_GROUP = [50, 100, 200, 500] as const;
export const DEFAULT_CELLS_PER_GROUP = 200;
/** Seurat DoHeatmap's clip for scaled data; the server clips to it. */
export const ZSCORE_CLIP = 2.5;

export type CellScale = "zscore" | "none";

export interface HeatmapCellsMeta {
  groupby: string;
  layer: string;
  genes: string[];
  missing_genes: string[];
  groups: string[];
  /** Cells drawn per group, in `groups` order */
  n_sampled: number[];
  /** Cells the group has */
  n_total: number[];
  /** Cells in the dataset */
  n_cells: number;
  scale: CellScale;
  value_range: [number, number];
  seed: number;
  per_group: number;
  max_cells: number;
}

export interface HeatmapCells {
  meta: HeatmapCellsMeta;
  /** Row of each sampled cell in the dataset, in plot order */
  cells: Int32Array;
  /** Index into meta.groups of each sampled cell */
  codes: Int32Array;
  /** One array per gene (meta.genes order), one value per sampled cell */
  values: Float32Array[];
}

/** Decode the endpoint's Arrow stream: column 0 cell rows, 1 group codes, then
 *  one column per gene, read by position; the facts are in schema metadata. */
export function decodeHeatmapCells(buffer: ArrayBuffer | Uint8Array): HeatmapCells {
  const table = tableFromIPC(buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer));
  const raw = table.schema.metadata.get("scview");
  if (!raw) throw new Error("The heatmap response has no description of its cells.");
  const meta = JSON.parse(raw) as HeatmapCellsMeta;
  const col = (i: number) => table.getChildAt(i)?.toArray();
  const asInt = (a: unknown) => (a instanceof Int32Array ? a : Int32Array.from((a ?? []) as ArrayLike<number>));
  const asFloat = (a: unknown) => (a instanceof Float32Array ? a : Float32Array.from((a ?? []) as ArrayLike<number>));
  return {
    meta,
    cells: asInt(col(0)),
    codes: asInt(col(1)),
    values: meta.genes.map((_, j) => asFloat(col(j + 2))),
  };
}

export interface GroupSpan {
  group: number;
  /** First column of the run */
  start: number;
  /** One past the last column */
  end: number;
}

/** Runs of equal group codes, left to right (one per group when sorted). */
export function groupSpans(codes: ArrayLike<number>): GroupSpan[] {
  const spans: GroupSpan[] = [];
  for (let i = 0; i < codes.length; i++) {
    const last = spans[spans.length - 1];
    if (last && last.group === codes[i]) last.end = i + 1;
    else spans.push({ group: codes[i]!, start: i, end: i + 1 });
  }
  return spans;
}

/** A Plotly colourscale that gives integer codes 0..n-1 their own colour,
 *  for z from -0.5 to n - 0.5 (code k sits mid-band at (k + 0.5) / n). */
export function discreteColorscale(colors: readonly string[]): [number, string][] {
  const n = colors.length;
  if (n === 0) return [[0, "#cbd5e1"], [1, "#cbd5e1"]];
  return colors.flatMap((c, k) => [
    [k / n, c],
    [(k + 1) / n, c],
  ] as [number, string][]);
}

/** Blue - white - red (ColorBrewer RdBu, reversed so high is red), for z-scores
 *  centred at 0. */
export const DIVERGING_COLORSCALE: [number, string][] = [
  [0, "#2166ac"],
  [0.17, "#67a9cf"],
  [0.33, "#d1e5f0"],
  [0.5, "#f7f7f7"],
  [0.67, "#fddbc7"],
  [0.83, "#ef8a62"],
  [1, "#b2182b"],
];

/** Height of one gene row (px): smaller with more genes, so 100 still fit. */
export function geneRowPx(nGenes: number): number {
  return nGenes > 60 ? 10 : nGenes > 30 ? 13 : 16;
}

const STRIP_PX = 16;
/** Least height of the heatmap rows (px): a few genes still leave room for the colour bar and its title. */
export const MIN_HEAT_PX = 180;
const GAP_PX = 4;
const MARGIN = { l: 90, r: 20, t: 30, b: 90 };

/**
 * Plotly traces and layout: genes (rows, top to bottom) x sampled cells
 * (columns), a strip of group colours on its own y axis above, a white line
 * at each group boundary and the group names centred under their cells.
 * `groupColors` is in meta.groups order; `sequential` is the colourscale for
 * scale "none" (z-scores use the diverging one, centred at 0).
 */
export function cellHeatmapFigure(
  data: HeatmapCells,
  groupColors: readonly string[],
  sequential: [number, string][],
): { data: object[]; layout: Record<string, unknown> } {
  const { meta, codes, values } = data;
  const nCells = codes.length;
  const genes = meta.genes;
  const rowPx = geneRowPx(genes.length);
  const heatPx = Math.max(genes.length * rowPx, MIN_HEAT_PX);
  const plotPx = heatPx + GAP_PX + STRIP_PX;
  const heatTop = heatPx / plotPx;
  const spans = groupSpans(codes);
  const x = Array.from({ length: nCells }, (_, i) => i);
  const zscore = meta.scale === "zscore";
  const [lo, hi] = meta.value_range;
  const strip = {
    type: "heatmap",
    x,
    y: [meta.groupby],
    z: [Array.from(codes)],
    zmin: -0.5,
    zmax: meta.groups.length - 0.5,
    colorscale: discreteColorscale(groupColors),
    showscale: false,
    text: [Array.from(codes, (k) => meta.groups[k] ?? "")],
    hovertemplate: `${meta.groupby}: %{text}<extra></extra>`,
    xaxis: "x",
    yaxis: "y2",
  };
  const heat = {
    type: "heatmap",
    x,
    y: genes,
    z: values.map((v) => Array.from(v)),
    zmin: lo,
    zmax: hi > lo ? hi : lo + 1,
    ...(zscore ? { zmid: 0 } : {}),
    colorscale: zscore ? DIVERGING_COLORSCALE : sequential,
    colorbar: {
      title: { text: zscore ? `z-score<br>(clipped ±${ZSCORE_CLIP})` : "Expression" },
      thickness: 12,
      len: heatTop,
      y: 0,
      yanchor: "bottom",
    },
    hovertemplate: `<b>%{y}</b><br>${zscore ? "z-score" : "value"} %{z:.2f}<extra></extra>`,
    xaxis: "x",
    yaxis: "y",
  };
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
    data: [heat, strip],
    layout: {
      height: plotPx + MARGIN.t + MARGIN.b,
      margin: MARGIN,
      xaxis: {
        range: [-0.5, nCells - 0.5],
        tickmode: "array",
        tickvals: spans.map((s) => (s.start + s.end - 1) / 2),
        ticktext: spans.map((s) => meta.groups[s.group] ?? ""),
        ticks: "",
        showgrid: false,
        zeroline: false,
        automargin: true,
      },
      yaxis: {
        domain: [0, heatTop],
        type: "category",
        categoryarray: genes,
        categoryorder: "array",
        autorange: "reversed",
        tickfont: { size: rowPx >= 13 ? 11 : 9 },
        automargin: true,
      },
      yaxis2: {
        domain: [heatTop + GAP_PX / plotPx, 1],
        type: "category",
        fixedrange: true,
        tickfont: { size: 10 },
        automargin: true,
      },
      shapes: boundaries,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    },
  };
}

const n = (v: number) => v.toLocaleString("en-US");

/** "200 cells per group (sampled), 2,368 of 13,836 cells, seed 0", plus which
 *  groups were smaller than that and whether the total cap cut the shares. */
export function cellHeatmapCaption(meta: HeatmapCellsMeta): string {
  const sampled = meta.n_sampled.reduce((a, b) => a + b, 0);
  const wanted = meta.n_total.reduce((a, t) => a + Math.min(t, meta.per_group), 0);
  const capped = sampled < wanted;
  const small = meta.groups.filter((_, k) => meta.n_total[k]! <= meta.per_group && !capped);
  let text = `${meta.per_group} cells per group (sampled), ${n(sampled)} of ${n(meta.n_cells)} cells, seed ${meta.seed}.`;
  if (capped) {
    text += ` Each group's share was scaled down to fit ${n(meta.max_cells)} cells.`;
  } else if (small.length > 0) {
    const list = small.slice(0, 4).map((g) => `${g} (${meta.n_total[meta.groups.indexOf(g)]})`);
    text += ` All cells of the smaller group${small.length === 1 ? "" : "s"} ${list.join(", ")}${small.length > 4 ? ` and ${small.length - 4} more` : ""}.`;
  }
  return text;
}

/** The colour caption for the scale. */
export function cellScaleCaption(scale: CellScale, range: [number, number]): string {
  return scale === "zscore"
    ? `Colour: each gene's z-score over the sampled cells, clipped to ±${ZSCORE_CLIP} (as Seurat's DoHeatmap).`
    : `Colour: expression on the layer's fixed ${range[0]} to ${range[1]} scale (higher values show as ${range[1]}), as the dot plot's fixed range.`;
}
