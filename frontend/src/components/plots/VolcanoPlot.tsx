import { useMemo } from "react";
import Plot from "@/components/plots/Plot";
import type { DEGene } from "@/api/de";

interface VolcanoPlotProps {
  genes: DEGene[];
  /** |log2FC| significance threshold (vertical guides) */
  fcThreshold: number;
  /** adjusted p-value threshold (horizontal guide) */
  pThreshold: number;
  /** Fired when a point is clicked, with the gene symbol */
  onGeneClick?: (gene: string) => void;
}

// -log10, with p=0 floored so it plots at the top instead of +Infinity.
const NEG_LOG10_FLOOR = 1e-300;
const negLog10 = (p: number) => -Math.log10(Math.max(p, NEG_LOG10_FLOOR));

/**
 * Volcano plot for a differential-expression result: log2 fold-change (x) vs
 * -log10 adjusted p-value (y). Up/down/non-significant genes are colored by the
 * fold-change + p thresholds; click a point to overlay that gene's expression.
 */
export function VolcanoPlot({ genes, fcThreshold, pThreshold, onGeneClick }: VolcanoPlotProps) {
  const { traces, yGuide, xMax, yMax } = useMemo(() => {
    const up: DEGene[] = [];
    const down: DEGene[] = [];
    const ns: DEGene[] = [];
    let xm = 1;
    let ym = negLog10(pThreshold) * 1.2;
    for (const g of genes) {
      const sig = g.pval_adj <= pThreshold;
      const y = negLog10(g.pval_adj);
      if (Number.isFinite(g.logfoldchange)) xm = Math.max(xm, Math.abs(g.logfoldchange));
      if (Number.isFinite(y)) ym = Math.max(ym, y);
      if (sig && g.logfoldchange >= fcThreshold) up.push(g);
      else if (sig && g.logfoldchange <= -fcThreshold) down.push(g);
      else ns.push(g);
    }

    const mk = (arr: DEGene[], color: string, name: string) => ({
      type: "scattergl" as const,
      mode: "markers" as const,
      name: `${name} (${arr.length})`,
      x: arr.map((g) => g.logfoldchange),
      y: arr.map((g) => negLog10(g.pval_adj)),
      customdata: arr.map((g) => g.gene),
      text: arr.map((g) => g.gene),
      hovertemplate: "<b>%{text}</b><br>log2FC=%{x:.2f}<br>-log10(adj.P)=%{y:.1f}<extra></extra>",
      marker: { color, size: 5, opacity: 0.6 },
    });

    return {
      traces: [
        mk(ns, "#cbd5e1", "Not sig."),
        mk(down, "#3b82f6", "Down"),
        mk(up, "#ef4444", "Up"),
      ],
      yGuide: negLog10(pThreshold),
      xMax: Math.ceil(xm) + 0.5,
      yMax: ym * 1.05,
    };
  }, [genes, fcThreshold, pThreshold]);

  const layout = useMemo(
    () => ({
      title: { text: "", font: { size: 12 } },
      xaxis: {
        title: { text: "log2 fold-change", font: { size: 11, color: "#64748b" } },
        range: [-xMax, xMax],
        zeroline: true,
        zerolinecolor: "#e2e8f0",
        tickfont: { size: 10, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      yaxis: {
        title: { text: "-log10(adj. P)", font: { size: 11, color: "#64748b" } },
        range: [0, yMax],
        tickfont: { size: 10, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      shapes: [
        { type: "line" as const, x0: fcThreshold, x1: fcThreshold, y0: 0, y1: yMax, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
        { type: "line" as const, x0: -fcThreshold, x1: -fcThreshold, y0: 0, y1: yMax, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
        { type: "line" as const, x0: -xMax, x1: xMax, y0: yGuide, y1: yGuide, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
      ],
      legend: { font: { size: 10 }, orientation: "h" as const, y: 1.12, x: 0 },
      margin: { t: 20, r: 16, b: 44, l: 52 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height: 360,
    }),
    [xMax, yMax, yGuide, fcThreshold],
  );

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{ responsive: true, displayModeBar: false }}
      useResizeHandler
      className="w-full"
      style={{ width: "100%" }}
      onClick={(e) => {
        const pt = e.points?.[0] as { customdata?: string } | undefined;
        if (pt?.customdata && onGeneClick) onGeneClick(pt.customdata);
      }}
    />
  );
}
