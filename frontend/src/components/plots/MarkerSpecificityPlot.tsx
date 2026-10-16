import { useMemo } from "react";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import type { MarkerGene } from "@/api/markers";
import {
  SPECIFIC_TAU,
  STRONG_LOG2FC,
  detectionRatio,
  formatPct,
  formatRatio,
} from "@/lib/markerSpecificity";

interface Props {
  markers: MarkerGene[];
  /** Names the plot's download, e.g. the group. */
  name: string;
  onGeneClick?: (gene: string) => void;
  height?: number;
  /** Genes in the strong-and-specific corner that get a text label. */
  maxLabels?: number;
}

/** Plotted genes: both values known. */
export function specificityPoints(markers: readonly MarkerGene[]) {
  return markers.filter(
    (m) => m.logfoldchange != null && Number.isFinite(m.logfoldchange) && m.tau != null && Number.isFinite(m.tau),
  ) as (MarkerGene & { logfoldchange: number; tau: number })[];
}

/** Strong (log2FC past the guide) and specific (tau past the guide). */
export function isStrongAndSpecific(m: { logfoldchange: number; tau: number }): boolean {
  return m.logfoldchange >= STRONG_LOG2FC && m.tau >= SPECIFIC_TAU;
}

/**
 * Effect size against specificity for one group's markers: log2 fold change (x)
 * vs tau (y). The top-right corner holds markers that are both strong and found
 * in this group only; a strong marker low on the plot is shared with other groups.
 */
export function MarkerSpecificityPlot({ markers, name, onGeneClick, height = 240, maxLabels = 6 }: Props) {
  const { traces, annotations, xMin, xMax } = useMemo(() => {
    const pts = specificityPoints(markers);
    const strong = pts.filter(isStrongAndSpecific);
    const rest = pts.filter((m) => !isStrongAndSpecific(m));
    const mk = (arr: typeof pts, color: string, label: string) => ({
      type: "scatter" as const,
      mode: "markers" as const,
      name: `${label} (${arr.length})`,
      x: arr.map((m) => m.logfoldchange),
      y: arr.map((m) => m.tau),
      customdata: arr.map((m) => m.gene),
      text: arr.map(
        (m) =>
          `<b>${m.gene}</b><br>log2FC ${m.logfoldchange.toFixed(2)} · tau ${m.tau.toFixed(2)}` +
          `<br>detected ${formatPct(m.pct_in)} here, ${formatPct(m.pct_max_other)} in ` +
          `${m.max_other_group ?? "next group"} (ratio ${formatRatio(detectionRatio(m))})`,
      ),
      hovertemplate: "%{text}<extra></extra>",
      marker: { color, size: 6, opacity: 0.75 },
    });
    const labelled = [...strong].sort((a, b) => b.logfoldchange * b.tau - a.logfoldchange * a.tau)
      .slice(0, maxLabels);
    let lo = 0;
    let hi = STRONG_LOG2FC + 1;
    for (const m of pts) {
      lo = Math.min(lo, m.logfoldchange);
      hi = Math.max(hi, m.logfoldchange);
    }
    return {
      traces: [mk(rest, "#94a3b8", "Other markers"), mk(strong, "#dc2626", "Strong and specific")],
      // Labels alternate above and below their point: the corner they mark is crowded.
      annotations: labelled.map((m, i) => ({
        x: m.logfoldchange,
        y: m.tau,
        text: m.gene,
        showarrow: false,
        yshift: i % 2 === 0 ? 9 : -9,
        font: { size: 9, color: "#7f1d1d" },
      })),
      xMin: Math.floor(lo) - 0.25,
      xMax: Math.ceil(hi) + 0.25,
    };
  }, [markers, maxLabels]);

  const layout = useMemo(
    () => ({
      xaxis: {
        title: { text: "log2 fold change", font: { size: 10, color: "#64748b" } },
        range: [xMin, xMax],
        zeroline: true,
        zerolinecolor: "#e2e8f0",
        tickfont: { size: 9, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      yaxis: {
        title: { text: "tau (specificity)", font: { size: 10, color: "#64748b" } },
        range: [-0.03, 1.05],
        tickfont: { size: 9, color: "#64748b" },
        gridcolor: "#f1f5f9",
      },
      shapes: [
        { type: "line" as const, x0: STRONG_LOG2FC, x1: STRONG_LOG2FC, y0: 0, y1: 1, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
        { type: "line" as const, x0: xMin, x1: xMax, y0: SPECIFIC_TAU, y1: SPECIFIC_TAU, line: { color: "#e2e8f0", width: 1, dash: "dash" as const } },
      ],
      annotations,
      legend: { font: { size: 9 }, orientation: "h" as const, y: 1.16, x: 0 },
      margin: { t: 24, r: 10, b: 36, l: 44 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      autosize: true,
      height,
    }),
    [xMin, xMax, annotations, height],
  );

  return (
    <Plot
      data={traces}
      layout={layout}
      config={plotConfig(`marker_specificity_${name}`)}
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
