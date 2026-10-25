import type { Data, Layout } from "plotly.js";
import type { BaselinePoint, PanelResult, PanelStep } from "@/api/markerPanel";
import { NEGATIVE_COLOR, POSITIVE_COLOR } from "@/lib/pcaDiagnostics";

/** Genes typed or pasted into a field: split on commas, spaces, semicolons, new lines. */
export function parseGenes(text: string): string[] {
  return [...new Set(text.split(/[\s,;]+/).map((g) => g.trim()).filter(Boolean))];
}

/** Accuracy vs genes: the selected panel (solid) and the top markers in turn (dashed). */
export function curveTraces(path: PanelStep[], baseline: BaselinePoint[]): Data[] {
  const out: Data[] = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "selected panel",
      x: path.map((p) => p.n_genes),
      y: path.map((p) => p.balanced_accuracy),
      text: path.map((p) => p.gene),
      line: { color: POSITIVE_COLOR, width: 2 },
      marker: { size: 5 },
      hovertemplate: "%{x} genes (+%{text}): %{y:.3f}<extra>selected</extra>",
    },
  ];
  if (baseline.length) {
    out.push({
      type: "scatter",
      mode: "lines+markers",
      name: "top markers in turn",
      x: baseline.map((p) => p.n_genes),
      y: baseline.map((p) => p.balanced_accuracy),
      text: baseline.map((p) => p.gene),
      line: { color: NEGATIVE_COLOR, width: 1.5, dash: "dash" },
      marker: { size: 4, symbol: "square" },
      hovertemplate: "%{x} genes (+%{text}): %{y:.3f}<extra>top markers</extra>",
    });
  }
  return out;
}

export function curveLayout(size: number): Partial<Layout> {
  return {
    autosize: true,
    height: 220,
    margin: { l: 48, r: 10, t: 28, b: 40 },
    xaxis: { title: { text: "genes in the panel" }, range: [0.5, Math.max(size, 2) + 0.5], dtick: size > 20 ? 5 : 2 },
    yaxis: { title: { text: "balanced accuracy" }, range: [0, 1.02] },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 10 } },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
  };
}

/** Row-normalised confusion (share of each true group predicted as each group). */
export function confusionShares(counts: number[][]): number[][] {
  return counts.map((row) => {
    const n = row.reduce((a, b) => a + b, 0);
    return row.map((v) => (n > 0 ? v / n : 0));
  });
}

/** The panel as plain text: one gene per line with what it helps, then the scores. */
export function panelText(r: PanelResult): string {
  const lines = [
    `Minimal marker panel for '${r.groupby}' (${r.panel.length} genes)`,
    `Balanced accuracy ${r.balanced_accuracy.toFixed(3)} (${r.cv}, up to ${r.cells_per_group} cells per group, ${r.n_cells} cells)`,
  ];
  if (r.baseline_accuracy != null) {
    lines.push(`Top ${r.baseline.length} markers taken in turn from each group: ${r.baseline_accuracy.toFixed(3)}`);
  }
  lines.push("");
  for (const g of r.genes) {
    lines.push(`${g.gene}\t${g.required ? "required" : g.helps.join(", ")}`);
  }
  return lines.join("\n");
}

/** The genes alone, one per line, for the clipboard. */
export function panelGenesText(r: PanelResult): string {
  return r.panel.join("\n");
}

/** The groups the final panel tells apart worst, by recall (lowest first). */
export function hardestGroups(r: PanelResult, n = 3): { group: string; recall: number }[] {
  const last = r.path[r.path.length - 1];
  if (!last) return [];
  return Object.entries(last.recall)
    .map(([group, recall]) => ({ group, recall }))
    .sort((a, b) => a.recall - b.recall)
    .slice(0, n);
}
