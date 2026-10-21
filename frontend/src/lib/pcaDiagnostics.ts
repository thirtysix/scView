import type { Data, Layout, Shape } from "plotly.js";
import type { PcaDiagnostics, PcLoadings } from "@/api/pca";

type Available = Extract<PcaDiagnostics, { available: true }>;

export const POSITIVE_COLOR = "#2563eb"; // blue-600
export const NEGATIVE_COLOR = "#ea580c"; // orange-600 (blue/orange reads for everyone)
const BAR_COLOR = "#94a3b8"; // slate-400, as the QC plots
const ELBOW_COLOR = "#0f172a"; // slate-900
const NEIGHBORS_COLOR = "#64748b"; // slate-500

/** "PC3 (0.9%)" */
export function pcLabel(d: Available, pc: number): string {
  const v = d.variance_ratio[pc - 1];
  return v === undefined ? `PC${pc}` : `PC${pc} (${fmtPct(v)})`;
}

export function fmtPct(share: number): string {
  const p = 100 * share;
  return `${p >= 10 ? p.toFixed(0) : p >= 1 ? p.toFixed(1) : p.toFixed(2)}%`;
}

/** Bars of each PC's variance share, with the cumulative curve on a right-hand axis. */
export function elbowTraces(d: Available): Data[] {
  const x = d.variance_ratio.map((_, i) => i + 1);
  return [
    {
      type: "bar",
      name: "variance ratio",
      x,
      y: d.variance_ratio.map((v) => 100 * v),
      marker: { color: x.map((pc) => (pc === d.elbow ? ELBOW_COLOR : BAR_COLOR)) },
      hovertemplate: "PC%{x}: %{y:.2f}% of the variance<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "cumulative",
      x,
      y: d.cumulative.map((v) => 100 * v),
      yaxis: "y2",
      line: { color: POSITIVE_COLOR, width: 1.5 },
      marker: { size: 3 },
      hovertemplate: "PC1..%{x}: %{y:.1f}% together<extra></extra>",
    },
  ];
}

/** The elbow (dashed) and, when the neighbour graph was built from the PCA, the
 *  number of PCs it used (dotted), as vertical lines with labels. */
export function elbowLayout(d: Available): Partial<Layout> {
  const shapes: Partial<Shape>[] = [];
  const annotations: NonNullable<Layout["annotations"]> = [];
  const vline = (x: number, color: string, dash: "dash" | "dot", text: string, y: number) => {
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color, width: 1, dash } });
    annotations.push({ x, y, xref: "x", yref: "paper", text, showarrow: false, xanchor: "left", xshift: 3, font: { size: 9, color } });
  };
  vline(d.elbow, ELBOW_COLOR, "dash", `elbow PC${d.elbow}`, 0.98);
  const nb = d.neighbors;
  if (nb && nb.from_pca && nb.n_pcs !== null && nb.n_pcs <= d.n_pcs) {
    vline(nb.n_pcs, NEIGHBORS_COLOR, "dot", `neighbours: ${nb.n_pcs} PCs`, 0.88);
  }
  return {
    autosize: true,
    margin: { l: 44, r: 44, t: 8, b: 36 },
    showlegend: false,
    bargap: 0.25,
    xaxis: { title: { text: "PC", font: { size: 10 } }, tickfont: { size: 9 }, dtick: d.n_pcs > 20 ? 5 : 1 },
    yaxis: { title: { text: "% of variance", font: { size: 10 } }, tickfont: { size: 9 }, rangemode: "tozero" },
    yaxis2: {
      title: { text: "cumulative %", font: { size: 10, color: POSITIVE_COLOR } },
      tickfont: { size: 9, color: POSITIVE_COLOR },
      overlaying: "y",
      side: "right",
      rangemode: "tozero",
      showgrid: false,
    },
    shapes,
    annotations,
  };
}

/** One horizontal bar per gene: negatives at the bottom, positives at the top, the
 *  largest at the outer ends. */
export function loadingTraces(entry: PcLoadings): Data[] {
  const neg = [...entry.negative].reverse(); // most negative nearest the bottom
  const pos = [...entry.positive].reverse(); // largest at the top
  const genes = [...neg, ...pos];
  return [
    {
      type: "bar",
      orientation: "h",
      name: `PC${entry.pc}`,
      y: genes.map((g) => g.gene),
      x: genes.map((g) => g.loading),
      marker: { color: genes.map((g) => (g.loading < 0 ? NEGATIVE_COLOR : POSITIVE_COLOR)) },
      hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
    },
  ];
}

export function loadingLayout(entry: PcLoadings): Partial<Layout> {
  const n = entry.positive.length + entry.negative.length;
  return {
    autosize: true,
    margin: { l: 72, r: 8, t: 8, b: 36 },
    showlegend: false,
    bargap: 0.2,
    xaxis: { title: { text: `loading on PC${entry.pc}`, font: { size: 10 } }, tickfont: { size: 9 }, zeroline: true },
    yaxis: { tickfont: { size: n > 24 ? 8 : 9 }, automargin: true, type: "category" },
  };
}
