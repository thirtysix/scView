import { useMemo } from "react";
import type { Data, Layout } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { plotConfig } from "@/lib/plotConfig";
import { detectionBins, formatPct } from "@/lib/deDetection";

interface Props {
  /** The DE response's detection histogram (0.01 % grid). */
  detection: number[];
  /** The cut-off the result was computed with (0: every gene tested). */
  minPct: number;
  /** The cut-off now picked, when it differs: drawn dotted. */
  pending?: number | null;
  height?: number;
}

const TESTED = "#60a5fa";
const SKIPPED = "#cbd5e1";
const UNDER_GAP = 0.35; // log10 units between the "<0.1%" bar and the 0.1% edge
const STYLE = { width: "100%" } as const;

/**
 * Genes by the larger share of cells expressing them in the two DE groups, on a
 * log axis from 0.1 % to all cells, with the detection cut-off marked: grey bars
 * were skipped, blue ones tested. Genes in under 0.1 % sit in one bar at the left.
 */
export function DetectionHistogram({ detection, minPct, pending = null, height = 110 }: Props) {
  const bins = useMemo(() => detectionBins(detection), [detection]);

  const data = useMemo<Data[]>(() => {
    const cut = minPct > 0 ? Math.log10(minPct) : -Infinity;
    const label = (lo: number) =>
      `${formatPct(10 ** lo)}–${formatPct(10 ** (lo + bins.width))}`;
    return [
      {
        type: "bar",
        x: [bins.minLog - UNDER_GAP, ...bins.lo.map((lo) => lo + bins.width / 2)],
        y: [bins.below, ...bins.counts],
        width: [0.3, ...bins.lo.map(() => bins.width)],
        marker: {
          color: [
            minPct > 0 ? SKIPPED : TESTED,
            ...bins.lo.map((lo) => (lo + 1e-9 >= cut ? TESTED : SKIPPED)),
          ],
        },
        customdata: [`under ${formatPct(10 ** bins.minLog)}`, ...bins.lo.map(label)],
        hovertemplate: "%{customdata}: %{y} genes<extra></extra>",
      },
    ];
  }, [bins, minPct]);

  const layout = useMemo<Partial<Layout>>(() => {
    const line = (v: number, dash: "dash" | "dot") => ({
      type: "line" as const,
      xref: "x" as const,
      yref: "paper" as const,
      x0: Math.log10(v),
      x1: Math.log10(v),
      y0: 0,
      y1: 1,
      line: { color: "#dc2626", width: dash === "dash" ? 2 : 1.5, dash },
    });
    return {
      autosize: true,
      height,
      margin: { l: 40, r: 10, t: 6, b: 26 },
      bargap: 0,
      showlegend: false,
      xaxis: {
        range: [bins.minLog - UNDER_GAP - 0.2, 0.05],
        // No "0.1%" tick: beside the "<0.1%" bar's label the two would overprint.
        tickvals: [bins.minLog - UNDER_GAP, -2, -1, 0],
        ticktext: ["<0.1%", "1%", "10%", "100%"],
        tickfont: { size: 9 },
        fixedrange: true,
      },
      yaxis: { tickfont: { size: 9 }, fixedrange: true },
      shapes: [
        ...(minPct > 0 ? [line(minPct, "dash")] : []),
        ...(pending != null && pending > 0 && pending !== minPct ? [line(pending, "dot")] : []),
      ],
    };
  }, [bins, minPct, pending, height]);

  return (
    <Plot
      data={data}
      layout={layout}
      config={plotConfig("de_detection")}
      useResizeHandler
      style={STYLE}
    />
  );
}
