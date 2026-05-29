import { useMemo } from "react";
import Plot from "react-plotly.js";
import { CATEGORICAL_COLORS } from "@/lib/colors";

interface ViolinPlotProps {
  data: Record<string, number[]>; // group name -> values
  title?: string;
  xLabel?: string;
  yLabel?: string;
  height?: number;
}

export function ViolinPlot({
  data,
  title,
  xLabel,
  yLabel,
  height = 400,
}: ViolinPlotProps) {
  const traces = useMemo(() => {
    const groups = Object.keys(data);
    return groups.map((group, i) => {
      const color = CATEGORICAL_COLORS[i % CATEGORICAL_COLORS.length]!;
      const rgbStr = `rgb(${color[0]},${color[1]},${color[2]})`;
      const rgbaStr = `rgba(${color[0]},${color[1]},${color[2]},0.5)`;

      return {
        type: "violin" as const,
        y: data[group] ?? [],
        name: group,
        box: { visible: true },
        meanline: { visible: true },
        line: { color: rgbStr },
        fillcolor: rgbaStr,
        spanmode: "soft" as const,
        scalemode: "width" as const,
        points: false as const,
      };
    });
  }, [data]);

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
        title: yLabel ? { text: yLabel, font: { size: 12, color: "#64748b" } } : undefined,
        tickfont: { size: 11, color: "#64748b" },
        gridcolor: "#f1f5f9",
        zeroline: false,
        automargin: true,
      },
      showlegend: false,
      margin: { t: title ? 40 : 20, r: 20, b: xLabel ? 50 : 30, l: yLabel ? 60 : 45 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      font: { family: "Inter, system-ui, sans-serif" },
      autosize: true,
      height,
    }),
    [title, xLabel, yLabel, height],
  );

  if (Object.keys(data).length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-sm text-slate-400" style={{ height }}>
        No data to display
      </div>
    );
  }

  return (
    <Plot
      data={traces}
      layout={layout}
      config={{
        responsive: true,
        displayModeBar: false,
      }}
      useResizeHandler
      className="w-full"
      style={{ width: "100%", height }}
    />
  );
}
