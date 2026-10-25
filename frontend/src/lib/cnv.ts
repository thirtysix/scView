import type { Data } from "plotly.js";
import type { CnvGroupScore, CnvHeatmap, CnvProgress } from "@/api/cnv";

/** The heatmap's values as rows of numbers (cells x display columns). */
export function decodeHeatmap(h: Pick<CnvHeatmap, "values" | "n_rows" | "n_cols" | "scale">): number[][] {
  const bin = atob(h.values);
  const rows: number[][] = [];
  for (let r = 0; r < h.n_rows; r++) {
    const row = new Array<number>(h.n_cols);
    for (let c = 0; c < h.n_cols; c++) {
      let v = bin.charCodeAt(r * h.n_cols + c);
      if (v > 127) v -= 256; // int8
      row[c] = v * h.scale;
    }
    rows.push(row);
  }
  return rows;
}

/** "chr7" -> "7"; the x-axis labels, one at each chromosome's middle. */
export function chromosomeTicks(h: Pick<CnvHeatmap, "chromosomes">): { vals: number[]; text: string[] } {
  return {
    vals: h.chromosomes.map((c) => (c.start + c.end - 1) / 2),
    text: h.chromosomes.map((c) => c.chromosome.replace(/^chr/, "")),
  };
}

/** Thin lines at each chromosome boundary and between row groups. */
export function heatmapShapes(h: Pick<CnvHeatmap, "chromosomes" | "groups">) {
  const lines: Record<string, unknown>[] = [];
  for (const c of h.chromosomes.slice(1)) {
    lines.push({ type: "line", xref: "x", yref: "paper", x0: c.start - 0.5, x1: c.start - 0.5, y0: 0, y1: 1,
      line: { color: "#334155", width: 0.6 } });
  }
  let row = 0;
  for (const g of h.groups.slice(0, -1)) {
    row += g.n_sampled;
    lines.push({ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: row - 0.5, y1: row - 0.5,
      line: { color: "#0f172a", width: 1 } });
  }
  return lines;
}

/** Each row group's label at its middle row (y-axis ticks). */
export function groupTicks(h: Pick<CnvHeatmap, "groups">): { vals: number[]; text: string[] } {
  const vals: number[] = [];
  const text: string[] = [];
  let row = 0;
  for (const g of h.groups) {
    vals.push(row + (g.n_sampled - 1) / 2);
    text.push(g.reference ? `${g.group} (ref)` : g.group);
    row += g.n_sampled;
  }
  return { vals, text };
}

/** Box traces from the server's quartiles (one per group; reference groups grey). */
export function scoreBoxTraces(groups: CnvGroupScore[]): Data[] {
  return groups.map(
    (g) =>
      ({
        type: "box",
        name: g.group,
        x: [g.group],
        q1: [g.q1],
        median: [g.median],
        q3: [g.q3],
        lowerfence: [g.lo],
        upperfence: [g.hi],
        mean: [g.mean],
        boxpoints: false,
        marker: { color: g.reference ? "#94a3b8" : "#dc2626" },
        line: { width: 1 },
        hovertemplate: `${g.group}${g.reference ? " (reference)" : ""}<br>n = ${g.n.toLocaleString()}<br>median ${g.median.toFixed(4)}${
          g.ratio !== null ? `<br>mean ${g.ratio.toFixed(1)}x the reference cells` : ""
        }<extra></extra>`,
      }) as unknown as Data,
  );
}

/** The progress line of a run. */
export function progressText(p: CnvProgress | null): string {
  if (!p) return "Placing the genes on the genome…";
  if (p.stage === "reference") return "Averaging the reference cells…";
  if (p.stage === "clusters") return "Clustering the profiles…";
  return `${p.done.toLocaleString()} of ${p.total.toLocaleString()} cells smoothed`;
}

/** "chr7, chr8 gain; chr10 loss", or "" when none was called. */
export function patternText(gains: string[], losses: string[]): string {
  const parts = [];
  if (gains.length) parts.push(`${gains.slice(0, 4).join(", ")} gain`);
  if (losses.length) parts.push(`${losses.slice(0, 4).join(", ")} loss`);
  return parts.join("; ");
}

/** Categorical obs columns that can hold the reference groups (2-100 values). */
export function referenceColumns(
  columns: { name: string; dtype: string; n_unique: number }[],
  nCells: number,
): { name: string; n_unique: number }[] {
  return columns.filter(
    (c) =>
      (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
      c.n_unique >= 2 &&
      c.n_unique <= 100 &&
      !(nCells > 0 && c.n_unique / nCells >= 0.9) &&
      c.name !== "cnv_leiden",
  );
}

/** The reference column to suggest: one named like a cell-type annotation. */
export function suggestReferenceColumn(cols: { name: string }[]): string {
  const re = /cell.?type|celltype|annotation|label|cluster/i;
  return (cols.find((c) => /cell.?type/i.test(c.name)) ?? cols.find((c) => re.test(c.name)) ?? cols[0])?.name ?? "";
}

/** Groups that look like normal immune or stromal cells (the usual reference). */
export function suggestReferenceGroups(values: string[]): string[] {
  const re = /\b(t|b|nk)\b|t[ _-]?cell|b[ _-]?cell|nk|lymph|myeloid|mono|macro|dc\b|mast|plasma|neutro|immune|fibro|endo|stromal|pericyte/i;
  return values.filter((v) => re.test(v));
}
