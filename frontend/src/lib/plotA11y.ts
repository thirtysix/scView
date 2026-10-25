/**
 * Screen-reader text for Plotly figures (components/plots/Plot): a name built
 * from the figure's title, trace kinds and axis titles, and, for a small
 * figure, its numbers as a table.
 */
import type { Data, Layout } from "plotly.js";

/** Figures with at most this many points get a data table for screen readers. */
export const TABLE_MAX_POINTS = 200;

type Titled = string | { text?: string } | undefined;

function titleText(t: unknown): string {
  if (typeof t === "string") return t;
  if (t && typeof t === "object" && typeof (t as { text?: unknown }).text === "string") return (t as { text: string }).text;
  return "";
}

/** Plotly titles may carry HTML (<b>, <br>, <sup>): plain text for reading out. */
function plain(text: string): string {
  return text.replace(/<br\s*\/?>/gi, " ").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
}

const KIND: Record<string, string> = {
  bar: "bar chart",
  scatter: "scatter plot",
  scattergl: "scatter plot",
  box: "box plot",
  violin: "violin plot",
  histogram: "histogram",
  heatmap: "heatmap",
  pie: "pie chart",
  sankey: "flow diagram",
};

/** "Bar chart: Cells per group. x axis: group, y axis: cells." */
export function plotAriaLabel(data: readonly Data[] | undefined, layout: Partial<Layout> | undefined): string {
  const traces = (data ?? []) as { type?: string; mode?: string }[];
  const kinds = [...new Set(traces.map((t) => KIND[t.type ?? "scatter"] ?? "chart"))];
  const kind = kinds.length === 1 ? kinds[0]! : kinds.length > 1 ? "chart" : "empty chart";
  const title = plain(titleText((layout as { title?: Titled } | undefined)?.title));
  const axes: string[] = [];
  for (const axis of ["xaxis", "yaxis"] as const) {
    const t = plain(titleText((layout?.[axis] as { title?: Titled } | undefined)?.title));
    if (t) axes.push(`${axis[0]} axis: ${t}`);
  }
  const head = kind.charAt(0).toUpperCase() + kind.slice(1);
  return `${head}${title ? `: ${title}` : ""}${axes.length ? `. ${axes.join(", ")}` : ""}.`;
}

export interface PlotTable {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
}

const fmt = (v: unknown): string | number =>
  typeof v === "number" ? (Number.isInteger(v) ? v : Number(v.toPrecision(4))) : v == null ? "" : String(v);

/**
 * The figure's numbers as one table (trace, x, y), when every trace is a
 * plain x/y trace (bar, scatter) and there are at most TABLE_MAX_POINTS
 * points in all; null otherwise.
 */
export function plotTable(data: readonly Data[] | undefined, layout: Partial<Layout> | undefined): PlotTable | null {
  const traces = (data ?? []) as { type?: string; name?: string; x?: unknown; y?: unknown }[];
  if (traces.length === 0) return null;
  let total = 0;
  for (const t of traces) {
    const type = t.type ?? "scatter";
    if (type !== "bar" && type !== "scatter" && type !== "scattergl") return null;
    if (!isArrayLike(t.x) || !isArrayLike(t.y) || t.x.length !== t.y.length) return null;
    total += t.x.length;
    if (total > TABLE_MAX_POINTS) return null;
  }
  if (total === 0) return null;
  const xTitle = plain(titleText((layout?.xaxis as { title?: Titled } | undefined)?.title)) || "x";
  const yTitle = plain(titleText((layout?.yaxis as { title?: Titled } | undefined)?.title)) || "y";
  const several = traces.length > 1;
  const rows: (string | number)[][] = [];
  traces.forEach((t, i) => {
    const xs = t.x as ArrayLike<unknown>;
    const ys = t.y as ArrayLike<unknown>;
    for (let j = 0; j < xs.length; j++) {
      const row = [fmt(xs[j]), fmt(ys[j])];
      rows.push(several ? [plain(t.name ?? `Trace ${i + 1}`), ...row] : row);
    }
  });
  return {
    caption: plotAriaLabel(data, layout).replace(/\.$/, "") + ", as a table",
    columns: several ? ["Series", xTitle, yTitle] : [xTitle, yTitle],
    rows,
  };
}

function isArrayLike(v: unknown): v is ArrayLike<unknown> {
  return Array.isArray(v) || ArrayBuffer.isView(v);
}
