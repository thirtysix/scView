import { useMemo, useState } from "react";
import Plot from "@/components/plots/Plot";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";
import { pseudoLog, pseudoLogTicks, summary } from "@/lib/logAxis";
import { plotConfig } from "@/lib/plotConfig";

interface ViolinPlotProps {
  data: Record<string, number[]>; // group name -> values
  /** Obs column the groups come from: colours and order then match the scatter. */
  groupColumn?: string;
  title?: string;
  xLabel?: string;
  yLabel?: string;
  height?: number;
}

const STATS = [
  ["max", "max"],
  ["q3", "upper quartile"],
  ["median", "median"],
  ["q1", "lower quartile"],
  ["min", "min"],
] as const;

export function ViolinPlot({
  data,
  groupColumn,
  title,
  xLabel,
  yLabel,
  height = 400,
}: ViolinPlotProps) {
  const colorOf = useCategoryColors(groupColumn);
  const order = useCategoryOrder(groupColumn);
  // A log value axis, per plot (lib/logAxis.ts: zeros and negatives stay).
  const [log, setLog] = useState(false);
  const groups = useMemo(() => sortByCategoryOrder(Object.keys(data), order), [data, order]);
  const traces = useMemo(() => {
    const violins = groups.map((group) => {
      const color = colorOf(group);
      const rgbStr = rgbCss(color);
      const rgbaStr = rgbCss(color, 0.5);
      const values = data[group] ?? [];

      return {
        type: "violin" as const,
        y: log ? values.map(pseudoLog) : values,
        name: group,
        box: { visible: true },
        meanline: { visible: true },
        line: { color: rgbStr },
        fillcolor: rgbaStr,
        spanmode: "soft" as const,
        scalemode: "width" as const,
        points: false as const,
        // On the log axis Plotly's hover would give the box on that scale: the
        // markers below give it in the original units instead.
        ...(log ? { hoverinfo: "skip" as const } : {}),
      };
    });
    if (!log) return violins;
    // Invisible markers at each group's quartiles and range, for the hover.
    const x: string[] = [];
    const y: number[] = [];
    const text: string[] = [];
    for (const group of groups) {
      const s = summary(data[group] ?? []);
      if (!s) continue;
      for (const [key, label] of STATS) {
        x.push(group);
        y.push(pseudoLog(s[key]));
        text.push(`${label} ${statText(s[key])}`);
      }
    }
    const stats = {
      type: "scatter" as const,
      mode: "markers" as const,
      x,
      y,
      customdata: text,
      marker: { size: 14, opacity: 0 },
      hovertemplate: "<b>%{x}</b><br>%{customdata}<extra></extra>",
      showlegend: false,
      name: "summary",
    };
    return [...violins, stats];
  }, [data, groups, colorOf, log]);

  const ticks = useMemo(() => {
    if (!log) return null;
    let min = Infinity;
    let max = -Infinity;
    for (const values of Object.values(data)) {
      for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return pseudoLogTicks(min, max);
  }, [data, log]);

  const layout = useMemo(
    () => ({
      title: title
        ? {
            text: title,
            font: { size: 14, color: "#334155" },
          }
        : undefined,
      xaxis: {
        title: xLabel ? { text: xLabel, font: { size: 12, color: "#64748b" } } : undefined,
        tickfont: { size: 11, color: "#64748b" },
        automargin: true,
      },
      yaxis: {
        title: yLabel
          ? { text: log ? `${yLabel} (log scale)` : yLabel, font: { size: 12, color: "#64748b" } }
          : undefined,
        tickfont: { size: 11, color: "#64748b" },
        gridcolor: "#f1f5f9",
        zeroline: false,
        automargin: true,
        ...(ticks ? { tickmode: "array" as const, tickvals: ticks.tickvals, ticktext: ticks.ticktext } : {}),
      },
      showlegend: false,
      margin: { t: title ? 40 : 20, r: 20, b: xLabel ? 50 : 30, l: yLabel ? 60 : 45 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      font: { family: "Inter, system-ui, sans-serif" },
      autosize: true,
      height,
    }),
    [title, xLabel, yLabel, height, log, ticks],
  );

  if (Object.keys(data).length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-sm text-slate-400" style={{ height }}>
        No data to display
      </div>
    );
  }

  return (
    <div className="relative">
      <label
        className="absolute left-1 top-1 z-10 flex cursor-pointer items-center gap-1 text-[10px] text-slate-500"
        title="Log value axis: sign(v) x log10(1 + |v|), so zeros and negative values stay; ticks in the original units"
      >
        <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} className="h-3 w-3" />
        log scale
      </label>
      <Plot
        data={traces}
        layout={layout}
        config={plotConfig("violin")}
        useResizeHandler
        className="w-full"
        style={{ width: "100%", height }}
      />
    </div>
  );
}

/** A summary value for the hover: three significant digits, whole from 1,000. */
function statText(v: number): string {
  return Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : String(Number(v.toPrecision(3)));
}
