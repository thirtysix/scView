import { useMemo } from "react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import type { DEGene } from "@/api/de";

interface VolcanoPlotProps {
  genes: DEGene[];
  /** |log2FC| significance threshold (vertical guides) */
  fcThreshold: number;
  /** adjusted p-value threshold (horizontal guide) */
  pThreshold: number;
  /** Fired when a point is clicked, with the gene symbol */
  onGeneClick?: (gene: string) => void;
  /** "coefficient": y is each gene's statistic (logistic regression gives no
   *  p-values), up / down by |log2FC| and the coefficient's sign alone. */
  yMode?: "p" | "coefficient";
}

// -log10, with p=0 floored so it plots at the top instead of +Infinity.
const NEG_LOG10_FLOOR = 1e-300;
const negLog10 = (p: number) => -Math.log10(Math.max(p, NEG_LOG10_FLOOR));

/**
 * Volcano plot for a differential-expression result: log2 fold-change (x) vs
 * -log10 adjusted p-value (y). Up/down/non-significant genes are colored by the
 * fold-change + p thresholds; click a point to overlay that gene's expression.
 * A result without p-values (logistic regression) plots its coefficient instead.
 */
export function VolcanoPlot({ genes, fcThreshold, pThreshold, onGeneClick, yMode = "p" }: VolcanoPlotProps) {
  const byCoef = yMode === "coefficient";
  const { traces, yGuide, xMax, yMin, yMax } = useMemo(() => {
    const up: DEGene[] = [];
    const down: DEGene[] = [];
    const ns: DEGene[] = [];
    const yOf = (g: DEGene) => (byCoef ? (g.stat ?? 0) : negLog10(g.pval_adj ?? 1));
    let xm = 1;
    let ym = byCoef ? 0 : negLog10(pThreshold) * 1.2;
    for (const g of genes) {
      const y = yOf(g);
      if (Number.isFinite(g.logfoldchange)) xm = Math.max(xm, Math.abs(g.logfoldchange));
      if (Number.isFinite(y)) ym = Math.max(ym, Math.abs(y));
      const sig = (g.pval_adj ?? 1) <= pThreshold;
      const upSide = byCoef ? y > 0 : sig;
      const downSide = byCoef ? y < 0 : sig;
      if (upSide && g.logfoldchange >= fcThreshold) up.push(g);
      else if (downSide && g.logfoldchange <= -fcThreshold) down.push(g);
      else ns.push(g);
    }

    const mk = (arr: DEGene[], color: string, name: string) => ({
      type: "scattergl" as const,
      mode: "markers" as const,
      name: `${name} (${arr.length})`,
      x: arr.map((g) => g.logfoldchange),
      y: arr.map(yOf),
      customdata: arr.map((g) => g.gene),
      text: arr.map((g) => g.gene),
      hovertemplate: byCoef
        ? "<b>%{text}</b><br>log2FC=%{x:.2f}<br>coefficient=%{y:.3f}<extra></extra>"
        : "<b>%{text}</b><br>log2FC=%{x:.2f}<br>-log10(adj.P)=%{y:.1f}<extra></extra>",
      marker: { color, size: 5, opacity: 0.6 },
    });

    const top = ym * 1.05 || 1;
    return {
      traces: [
        mk(ns, "#cbd5e1", byCoef ? "Other" : "Not sig."),
        mk(down, "#3b82f6", "Down"),
        mk(up, "#ef4444", "Up"),
      ],
      yGuide: byCoef ? 0 : negLog10(pThreshold),
      xMax: Math.ceil(xm) + 0.5,
      yMin: byCoef ? -top : 0,
      yMax: top,
    };
  }, [genes, fcThreshold, pThreshold, byCoef]);

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
        title: {
          text: byCoef ? "logistic regression coefficient" : "-log10(adj. P)",
          font: { size: 11, color: "#64748b" },
        },
        range: [yMin, yMax],
        tickfont: { size: 10, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      shapes: [
        { type: "line" as const, x0: fcThreshold, x1: fcThreshold, y0: yMin, y1: yMax, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
        { type: "line" as const, x0: -fcThreshold, x1: -fcThreshold, y0: yMin, y1: yMax, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
        { type: "line" as const, x0: -xMax, x1: xMax, y0: yGuide, y1: yGuide, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
      ],
      legend: { font: { size: 10 }, orientation: "h" as const, y: 1.12, x: 0 },
      margin: { t: 20, r: 16, b: 44, l: 52 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height: 360,
    }),
    [xMax, yMin, yMax, yGuide, fcThreshold, byCoef],
  );

  return (
    <Plot
      data={traces}
      layout={layout}
      config={plotConfig("volcano")}
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
