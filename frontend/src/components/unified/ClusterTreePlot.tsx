import { useMemo } from "react";
import type { Layout } from "plotly.js";
import type { ClusterTree } from "@/api/clustering";
import Plot from "@/components/plots/Plot";
import { layoutTree, treeTraces } from "@/lib/clusterTree";
import { plotConfig } from "@/lib/plotConfig";

const ROW_PX = 56;

/**
 * The clustering tree across a sweep's resolutions (clustree-style): one row
 * per resolution, nodes sized by cells and coloured by SC3 stability, edges by
 * shared cells. Clicking a node picks its resolution; `selected` is shaded.
 */
export function ClusterTreePlot({
  tree,
  selected,
  onSelect,
}: {
  tree: ClusterTree;
  selected: number | null;
  onSelect: (resolution: number) => void;
}) {
  const layout = useMemo(() => layoutTree(tree), [tree]);
  const data = useMemo(() => treeTraces(tree, layout), [tree, layout]);
  const level = selected == null ? -1 : layout.resolutions.indexOf(selected);
  const n = layout.resolutions.length;
  const height = Math.max(160, n * ROW_PX + 48);
  const plotLayout = useMemo<Partial<Layout>>(
    () => ({
      height,
      margin: { l: 44, r: 8, t: 28, b: 12 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      hovermode: "closest",
      xaxis: { visible: false, range: [-layout.halfWidth - 0.7, layout.halfWidth + 0.7], fixedrange: true },
      yaxis: {
        range: [n - 0.5, -0.5],
        tickvals: layout.resolutions.map((_, i) => i),
        ticktext: layout.resolutions.map(String),
        title: { text: "Resolution", font: { size: 10 } },
        tickfont: { size: 10 },
        zeroline: false,
        showgrid: false,
        fixedrange: true,
      },
      shapes:
        level < 0
          ? []
          : [
              {
                type: "rect",
                xref: "paper",
                x0: 0,
                x1: 1,
                yref: "y",
                y0: level - 0.45,
                y1: level + 0.45,
                fillcolor: "rgba(59, 130, 246, 0.12)",
                line: { width: 0 },
                layer: "below",
              },
            ],
    }),
    [height, layout, level, n],
  );
  return (
    <Plot
      data={data}
      layout={plotLayout}
      config={plotConfig("cluster_tree")}
      style={{ width: "100%", height }}
      useResizeHandler
      onClick={(e) => {
        const r = e.points?.[0]?.customdata;
        if (typeof r === "number") onSelect(r);
      }}
    />
  );
}
