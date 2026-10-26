import { useMemo, useState } from "react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { useResolvedTheme } from "@/lib/theme";
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

const LABEL_CHOICES = [0, 10, 20] as const;
// Up / down are the two ends of a red-blue scale (safe for red-green colour
// blindness); not significant is a grey that recedes on either background.
const UP = "#ef4444";
const DOWN = "#3b82f6";
const NOT_SIG = { light: "#cbd5e1", dark: "#475569" } as const;

/**
 * The genes to name on the plot: half from the up side and half from the down
 * side (the rest from the other side when one runs short), each the most
 * significant (by -log10 adj. P, or |coefficient|), ties by |log2FC|. Split,
 * because many genes share the floored top p-value and the larger fold changes
 * of one side would otherwise take every name.
 */
export function volcanoLabelGenes(up: DEGene[], down: DEGene[], n: number, yOf: (g: DEGene) => number): DEGene[] {
  if (n <= 0) return [];
  const rank = (gs: DEGene[]) =>
    gs
      .filter((g) => Number.isFinite(g.logfoldchange) && Number.isFinite(yOf(g)))
      .sort((a, b) => Math.abs(yOf(b)) - Math.abs(yOf(a)) || Math.abs(b.logfoldchange) - Math.abs(a.logfoldchange));
  const u = rank(up);
  const d = rank(down);
  const nUp = Math.min(u.length, Math.max(Math.ceil(n / 2), n - d.length));
  return rank([...u.slice(0, nUp), ...d.slice(0, n - nUp)]);
}

const HEIGHT = 360;
const MARGIN = { t: 20, r: 16, b: 44, l: 52 };
const LINE_PX = 13; // one label line at 10 px type
const ARM_PX = 26; // horizontal reach of a label's leader line

export interface LabelSpot {
  /** pixel offset of the label from its point (Plotly annotation ax / ay) */
  ax: number;
  ay: number;
  xanchor: "left" | "right";
}

/**
 * Where to put each name so none overlap: names go to the right of their point
 * (to the left on the negative side, or near the plot's outer edge, inwards),
 * and each side's names are stacked downwards at least one line apart. Many
 * genes share the floored top p-value, so their names would otherwise sit on
 * one spot.
 */
export function labelSpots(
  points: { x: number; y: number }[],
  xRange: [number, number],
  yRange: [number, number],
  plotHeightPx = HEIGHT - MARGIN.t - MARGIN.b,
): LabelSpot[] {
  const [x0, x1] = xRange;
  const [y0, y1] = yRange;
  const toPx = (y: number) => (1 - (y - y0) / (y1 - y0 || 1)) * plotHeightPx;
  const sideOf = (x: number) => {
    const f = (x - x0) / (x1 - x0 || 1);
    const right = x >= 0;
    return f > 0.8 ? "left" : f < 0.2 ? "right" : right ? "right" : "left";
  };
  const out: LabelSpot[] = points.map(() => ({ ax: 0, ay: 0, xanchor: "left" }));
  for (const side of ["left", "right"] as const) {
    const idx = points
      .map((p, i) => ({ i, px: toPx(p.y) }))
      .filter(({ i }) => sideOf(points[i]!.x) === side)
      .sort((a, b) => a.px - b.px);
    let next = -Infinity;
    for (const { i, px } of idx) {
      const at = Math.min(Math.max(px, next), plotHeightPx);
      next = at + LINE_PX;
      out[i] = { ax: side === "right" ? ARM_PX : -ARM_PX, ay: at - px, xanchor: side === "right" ? "left" : "right" };
    }
  }
  return out;
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
  const theme = useResolvedTheme();
  const [labelTop, setLabelTop] = useState<number>(10);
  const { traces, labelled, yGuide, xMax, yMin, yMax } = useMemo(() => {
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
        mk(ns, NOT_SIG[theme], byCoef ? "Other" : "Not sig."),
        mk(down, DOWN, "Down"),
        mk(up, UP, "Up"),
      ],
      labelled: volcanoLabelGenes(up, down, labelTop, yOf),
      yGuide: byCoef ? 0 : negLog10(pThreshold),
      xMax: Math.ceil(xm) + 0.5,
      yMin: byCoef ? -top : 0,
      yMax: top,
    };
  }, [genes, fcThreshold, pThreshold, byCoef, theme, labelTop]);

  const annotations = useMemo(() => {
    const at = (g: DEGene) => ({ x: g.logfoldchange, y: byCoef ? (g.stat ?? 0) : negLog10(g.pval_adj ?? 1) });
    const spots = labelSpots(labelled.map(at), [-xMax, xMax], [yMin, yMax]);
    return labelled.map((g, i) => ({
      ...at(g),
      ...spots[i]!,
      text: g.gene,
      showarrow: true,
      arrowhead: 0,
      arrowwidth: 1,
      arrowcolor: "#94a3b8",
      font: { size: 10, color: "#334155" },
      bgcolor: "rgba(255, 255, 255, 0.75)",
      borderpad: 1,
    }));
  }, [labelled, byCoef, xMax, yMin, yMax]);

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
      annotations,
      legend: { font: { size: 10 }, orientation: "h" as const, y: 1.12, x: 0 },
      margin: MARGIN,
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height: HEIGHT,
    }),
    [xMax, yMin, yMax, yGuide, fcThreshold, byCoef, annotations],
  );

  return (
    <div>
      <div className="flex justify-end px-2 pt-1.5 text-[11px] text-slate-500">
        <label className="flex items-center gap-1" title="Write the names of the most significant up and down genes on the plot">
          Name genes
          <select
            className="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]"
            value={labelTop}
            onChange={(e) => setLabelTop(Number(e.target.value))}
          >
            {LABEL_CHOICES.map((n) => (
              <option key={n} value={n}>
                {n === 0 ? "none" : `top ${n}`}
              </option>
            ))}
          </select>
        </label>
      </div>
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
    </div>
  );
}
