import type { Data, Layout } from "plotly.js";
import type { Coverage, FragmentLengths, FragmentQc, QcValuesGroup } from "@/api/atacFragments";

// Figures of the ATAC fragment cards (components/unified/AtacFragmentQc.tsx, AtacCoverage.tsx).

const AXIS = { tickfont: { size: 10, color: "#64748b" }, gridcolor: "#f1f5f9", zeroline: false };
const TITLE = (text: string) => ({ text, font: { size: 11, color: "#64748b" } });
const BASE = {
  paper_bgcolor: "white",
  plot_bgcolor: "white",
  font: { family: "Inter, system-ui, sans-serif" },
};
const NEUTRAL = "#2563eb";

export type ColorOf = (group: string) => string;

/** Column values that are numbers. */
const numbers = (v: (number | null)[]) => v.filter((x): x is number => x !== null);

/** Equal bins from the lowest value to the 99.5th percentile, the rest in the last bin: the
 *  app's Plotly bundle has no histogram trace, and one odd cell would squash the others. */
export function binned(values: number[], n = 60): { centers: number[]; counts: number[]; width: number } {
  if (!values.length) return { centers: [], counts: [], width: 1 };
  const sorted = [...values].sort((a, b) => a - b);
  const low = sorted[0]!;
  const high = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.995))]!;
  const width = high > low ? (high - low) / n : 1;
  const counts = new Array<number>(n).fill(0);
  for (const v of sorted) {
    const i = Math.min(n - 1, Math.floor((v - low) / width));
    counts[i] = (counts[i] ?? 0) + 1;
  }
  return { centers: counts.map((_, i) => low + (i + 0.5) * width), counts, width };
}

export function histogramFigure(
  title: string,
  values: (number | null)[],
  note?: { low?: number; high?: number },
): { data: Data[]; layout: Partial<Layout> } {
  const shapes: Partial<Layout["shapes"][number]>[] = [];
  for (const x of [note?.low, note?.high]) {
    if (x !== undefined) {
      shapes.push({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: "#94a3b8", width: 1, dash: "dot" } });
    }
  }
  const bins = binned(numbers(values));
  return {
    data: [{ type: "bar", x: bins.centers, y: bins.counts, width: bins.width, marker: { color: NEUTRAL },
             hovertemplate: `${title} %{x:.2f}<br>%{y} cells<extra></extra>` }],
    layout: {
      ...BASE,
      height: 170,
      margin: { t: 6, r: 10, b: 36, l: 44 },
      xaxis: { ...AXIS, title: TITLE(title) },
      yaxis: { ...AXIS, title: TITLE("cells") },
      shapes,
      showlegend: false,
    },
  };
}

/** One violin per group. Traces are named by group, so the group axis is a category axis:
 *  cluster numbers ("0", "1", ...) would make it numeric and label every other row. */
export function violinFigure(
  title: string,
  groups: QcValuesGroup[],
  key: "tss_enrichment" | "nucleosome_signal",
  colorOf: ColorOf,
): { data: Data[]; layout: Partial<Layout> } {
  const names = groups.map((g) => g.name);
  return {
    data: groups.map((g) => ({
      type: "violin" as const,
      name: g.name,
      x: numbers(g[key]),
      y0: g.name,
      orientation: "h" as const,
      points: false as const,
      spanmode: "hard" as const,
      box: { visible: true, width: 0.2 },
      meanline: { visible: false },
      line: { color: colorOf(g.name), width: 1 },
      fillcolor: colorOf(g.name),
      opacity: 0.7,
      hoveron: "violins" as const,
    })) as Data[],
    layout: {
      ...BASE,
      height: Math.max(150, 26 * groups.length + 50),
      margin: { t: 6, r: 10, b: 36, l: 110 },
      xaxis: { ...AXIS, title: TITLE(title) },
      yaxis: { ...AXIS, type: "category", tickmode: "array", tickvals: names, ticktext: names, autorange: "reversed",
               automargin: true },
      showlegend: false,
    },
  };
}

/** Length histogram, linear y: the ladder at about 200 bp steps is the point. */
export function lengthFigure(hist: FragmentQc["length_histogram"]): { data: Data[]; layout: Partial<Layout> } {
  const counts = hist.counts.slice(0, hist.max);
  return {
    data: [{ type: "scatter", mode: "lines", x: counts.map((_, i) => i), y: counts, line: { color: NEUTRAL, width: 1 },
             fill: "tozeroy", fillcolor: "rgba(37,99,235,0.15)",
             hovertemplate: "%{x} bp<br>%{y:,} fragments<extra></extra>" }],
    layout: {
      ...BASE,
      height: 200,
      margin: { t: 6, r: 10, b: 36, l: 60 },
      xaxis: { ...AXIS, title: TITLE("Fragment length (bp)"), range: [0, 800] },
      yaxis: { ...AXIS, title: TITLE("fragments in cells") },
      showlegend: false,
    },
  };
}

export function lengthByGroupFigure(lengths: FragmentLengths, colorOf: ColorOf): { data: Data[]; layout: Partial<Layout> } {
  return {
    data: lengths.groups.map((g) => ({
      type: "scatter" as const,
      mode: "lines" as const,
      name: g.name,
      x: lengths.x,
      y: g.share,
      line: { color: colorOf(g.name), width: 1 },
      hovertemplate: `${g.name}<br>%{x} bp: %{y:.2%}<extra></extra>`,
    })) as Data[],
    layout: {
      ...BASE,
      height: 220,
      margin: { t: 6, r: 10, b: 36, l: 56 },
      xaxis: { ...AXIS, title: TITLE("Fragment length (bp)"), range: [0, 800] },
      yaxis: { ...AXIS, title: TITLE(`share per ${lengths.bin} bp`), tickformat: ".1%" },
      showlegend: lengths.groups.length > 1,
      legend: { font: { size: 10 } },
    },
  };
}

export function tssProfileFigure(profile: FragmentQc["tss_profile"]): { data: Data[]; layout: Partial<Layout> } {
  const x = profile.fold.map((_, i) => profile.start + (i + 0.5) * profile.bin);
  return {
    data: [{ type: "scatter", mode: "lines", x, y: profile.fold, line: { color: NEUTRAL, width: 1.5 },
             hovertemplate: "%{x} bp from the TSS<br>%{y:.2f} x the flanks<extra></extra>" }],
    layout: {
      ...BASE,
      height: 200,
      margin: { t: 6, r: 10, b: 36, l: 50 },
      xaxis: { ...AXIS, title: TITLE("Distance from the TSS (bp)") },
      yaxis: { ...AXIS, title: TITLE("cuts, fold over the flanks"), rangemode: "tozero" },
      showlegend: false,
    },
  };
}

const TRACK_HEIGHT = 56;
const ANNOTATION_HEIGHT = 56;

/** The y axis key of row ``i`` (0-based): "yaxis", "yaxis2", ... */
function yKey(i: number): string {
  return i === 0 ? "yaxis" : `yaxis${i + 1}`;
}

/** Trace reference of row ``i``: "y", "y2", ... */
function yRef(i: number): string {
  return i === 0 ? "y" : `y${i + 1}`;
}

/** One filled track per group sharing x and y, peaks and genes in a row beneath. */
export function coverageFigure(cov: Coverage, colorOf: ColorOf): { data: Data[]; layout: Partial<Layout> } {
  const n = cov.tracks.length;
  const height = n * TRACK_HEIGHT + ANNOTATION_HEIGHT + 44;
  const plotHeight = height - 44;
  const annotationShare = ANNOTATION_HEIGHT / plotHeight;
  const rowShare = (1 - annotationShare) / Math.max(n, 1);
  const data: Data[] = cov.tracks.map((t, i) => ({
    type: "scatter" as const,
    mode: "lines" as const,
    name: t.name,
    x: cov.x,
    y: t.values,
    xaxis: "x",
    yaxis: yRef(i),
    line: { color: colorOf(t.name), width: 1, shape: "hvh" as const },
    fill: "tozeroy" as const,
    fillcolor: colorOf(t.name),
    hovertemplate: `${t.name}<br>%{x:,.0f}: %{y:.2f}<extra></extra>`,
  })) as Data[];
  const layout: Record<string, unknown> = {
    ...BASE,
    height,
    margin: { t: 6, r: 12, b: 38, l: 48 },
    showlegend: false,
    hovermode: "closest",
    xaxis: {
      ...AXIS,
      range: [cov.region.start, cov.region.end],
      anchor: yRef(n),
      title: TITLE(`${cov.region.chrom} position (bp)`),
      tickformat: ",d",
    },
  };
  const ymax = cov.ymax > 0 ? cov.ymax * 1.05 : 1;
  const annotations: Partial<Layout["annotations"][number]>[] = [];
  cov.tracks.forEach((t, i) => {
    const top = 1 - i * rowShare;
    layout[yKey(i)] = {
      ...AXIS,
      domain: [top - rowShare + 0.004, top],
      range: [0, ymax],
      nticks: 2,
      anchor: "x",
    };
    annotations.push({
      xref: "paper", yref: "paper", x: 1, y: top, xanchor: "right", yanchor: "top", showarrow: false,
      text: `${t.name} (n = ${t.n_cells.toLocaleString()})`,
      font: { size: 10, color: "#475569" },
    });
  });
  // The row beneath: peaks along its top, genes under them, with an arrow for the strand.
  layout[yKey(n)] = {
    domain: [0, annotationShare - 0.004],
    range: [0, 1],
    visible: false,
    fixedrange: true,
    anchor: "x",
  };
  const row = yRef(n);
  const shapes: Partial<Layout["shapes"][number]>[] = cov.peaks.map((p) => ({
    type: "rect", xref: "x", yref: row as "y", x0: p.start, x1: p.end, y0: 0.82, y1: 1,
    fillcolor: "#475569", line: { width: 0 },
  }));
  const genes = laneGenes(cov);
  genes.forEach(({ gene, lane }) => {
    const y = 0.62 - lane * 0.3;
    shapes.push({
      type: "line", xref: "x", yref: row as "y", x0: gene.start, x1: gene.end, y0: y, y1: y,
      line: { color: "#334155", width: 2 },
    });
    const forward = gene.strand !== "-";
    const left = Math.max(gene.start, cov.region.start);
    const right = Math.min(gene.end, cov.region.end);
    annotations.push({
      xref: "x", yref: row as "y", axref: "x", ayref: row as "y",
      x: forward ? right : left, y,
      ax: forward ? left : right, ay: y,
      showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 1, arrowcolor: "#334155", text: "",
    });
    annotations.push({
      xref: "x", yref: row as "y", x: (left + right) / 2, y: y - 0.02, yanchor: "top", showarrow: false,
      text: gene.name, font: { size: 10, color: "#334155" },
    });
  });
  layout.shapes = shapes;
  layout.annotations = annotations;
  return { data, layout: layout as Partial<Layout> };
}

/** Each gene on the first of two lanes that its neighbour's label leaves free. */
function laneGenes(cov: Coverage): { gene: Coverage["genes"][number]; lane: number }[] {
  const span = cov.region.end - cov.region.start;
  const room = span * 0.08; // about the width a name takes
  const ends = [-Infinity, -Infinity];
  return cov.genes.map((gene) => {
    const lane = gene.start >= (ends[0] ?? -Infinity) ? 0 : 1;
    ends[lane] = gene.end + room;
    return { gene, lane };
  });
}
