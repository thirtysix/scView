import { useMemo } from "react";
import Plot from "@/components/plots/Plot";
import { getPalette } from "@/lib/colors";
import { plotConfig } from "@/lib/plotConfig";
import type { Curve } from "@/lib/studyPlanner";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Power (y, 0-100%) against donors per group (x), one line per cells-per-donor
 * value, the pessimistic curve dashed, the target power dotted.
 */
export function PowerCurves({
  curves,
  target,
  title,
  unit,
  filename,
}: {
  curves: Curve[];
  target: number;
  title: string;
  /** What a curve's cells value counts ("cells per donor"). */
  unit: string;
  filename: string;
}) {
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));
  const main = curves.filter((c) => !c.pessimistic);
  const data = useMemo(
    () =>
      curves.map((c) => {
        const i = main.findIndex((m) => m.cells === c.cells);
        const rgb = palette[(i < 0 ? 0 : i) % palette.length]!;
        const color = c.pessimistic ? "#94a3b8" : `rgb(${rgb.join(",")})`;
        const name = c.pessimistic
          ? `${c.cells.toLocaleString()} (upper spread)`
          : `${c.cells.toLocaleString()}`;
        return {
          type: "scatter" as const,
          mode: "lines+markers" as const,
          name,
          x: c.donors,
          y: c.power.map((p) => (p == null ? null : p * 100)),
          customdata: c.se.map((s) => (s == null ? 0 : s * 100)),
          line: { color, width: c.pessimistic ? 1.5 : 2, dash: c.pessimistic ? ("dash" as const) : ("solid" as const) },
          marker: { color, size: 5 },
          hovertemplate: `%{x} donors per group, ${c.cells.toLocaleString()} ${unit}${c.pessimistic ? " (upper spread)" : ""}: %{y:.0f}% (±%{customdata:.0f})<extra></extra>`,
        };
      }),
    [curves, main, palette, unit],
  );
  const donors = curves.flatMap((c) => c.donors);
  const x0 = Math.min(...donors);
  const x1 = Math.max(...donors);
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        height: 260,
        margin: { t: 26, r: 10, b: 40, l: 46 },
        title: { text: title, font: { size: 12, color: "#334155" } },
        xaxis: {
          title: { text: "Donors per group", font: { size: 11, color: "#64748b" } },
          tickvals: [...new Set(donors)].sort((a, b) => a - b),
          tickfont: { size: 10 },
          gridcolor: "#f1f5f9",
        },
        yaxis: {
          title: { text: "Power (%)", font: { size: 11, color: "#64748b" } },
          range: [0, 102],
          tickfont: { size: 10 },
          gridcolor: "#f1f5f9",
        },
        shapes: [
          {
            type: "line",
            x0,
            x1,
            y0: target * 100,
            y1: target * 100,
            line: { color: "#cbd5e1", width: 1, dash: "dot" },
          },
        ],
        legend: { orientation: "h", y: -0.28, font: { size: 10 }, title: { text: `${unit}: `, font: { size: 10 } } },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
      }}
      config={plotConfig(filename)}
      useResizeHandler
      style={{ width: "100%" }}
    />
  );
}
