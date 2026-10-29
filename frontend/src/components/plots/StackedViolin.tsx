import { useMemo } from "react";
import Plot from "@/components/plots/Plot";
import { useCategoryColors, useCategoryOrder } from "@/hooks/useCategoryColors";
import { rgbCss, sortByCategoryOrder } from "@/lib/categoryColors";
import { valuesByGroup } from "@/lib/geneCoexpression";
import { plotConfig } from "@/lib/plotConfig";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { SELECTION_COLOR, SELECTION_FILL, selectedValues } from "@/lib/linkedSelection";
import { selectionName } from "@/lib/linkedTraces";

interface StackedViolinProps {
  /** Genes, one row each, top to bottom */
  genes: string[];
  values: Record<string, Float32Array>;
  /** Each cell's group code in `groupColumn` */
  codes: Int32Array;
  groupColumn: string;
  /** Cells drawn (a sample on large data); all when null */
  cells?: Uint32Array | null;
  /** Height of one gene's row, pixels */
  rowHeight?: number;
  /** File name stem for the PNG / SVG export */
  exportName?: string;
}

const AXIS_FONT = { size: 10, color: "#64748b" };

/** Plotly's id of row `i`'s y axis: "y", "y2", ... and its layout key. */
const yId = (i: number) => (i === 0 ? "y" : `y${i + 1}`) as "y";
const yKey = (i: number) => (i === 0 ? "yaxis" : `yaxis${i + 1}`);

/**
 * Stacked violin (scanpy `stacked_violin`): one row per gene, one violin per
 * group, every row on its own value axis so a lowly expressed gene keeps its
 * shape. Groups keep the scatter's colours and order.
 */
export function StackedViolin({
  genes,
  values,
  codes,
  groupColumn,
  cells = null,
  rowHeight = 64,
  exportName = "stacked_violin",
}: StackedViolinProps) {
  const colorOf = useCategoryColors(groupColumn);
  const order = useCategoryOrder(groupColumn);
  // C035: the selection as one more column, from its cells among those drawn.
  const { selected, mask } = useLinkedSelection(codes.length);
  const selName = selected ? selectionName(selected.size) : null;
  const selCells = useMemo(() => {
    if (!mask) return null;
    if (!cells) return null;
    const out: number[] = [];
    for (let k = 0; k < cells.length; k++) if (mask[cells[k]!]) out.push(cells[k]!);
    return Uint32Array.from(out);
  }, [mask, cells]);

  const byGene = useMemo(
    () =>
      genes.map((g) => {
        const d: Record<string, number[] | Float32Array> = valuesByGroup(values[g]!, codes, order, cells);
        if (!selected || !selName) return d;
        const v = values[g]!;
        return { ...d, [selName]: selCells ? Float32Array.from(selCells, (i) => v[i]!) : selectedValues(v, selected) };
      }),
    [genes, values, codes, order, cells, selected, selName, selCells],
  );
  const groups = useMemo(() => {
    const sorted = sortByCategoryOrder(
      [...new Set(byGene.flatMap((d) => Object.keys(d)))].filter((g) => g !== selName),
      order,
    );
    return selName ? [...sorted, selName] : sorted;
  }, [byGene, order, selName]);

  const traces = useMemo(
    () =>
      genes.flatMap((gene, i) =>
        groups.map((group) => {
          const isSel = group === selName;
          const line = isSel ? SELECTION_COLOR : rgbCss(colorOf(group));
          return {
            type: "violin" as const,
            y: byGene[i]![group] ?? [],
            name: group,
            xaxis: "x",
            yaxis: yId(i),
            line: { color: line, width: 0.8 },
            fillcolor: isSel ? SELECTION_FILL : rgbCss(colorOf(group), 0.6),
            spanmode: "hard" as const,
            scalemode: "width" as const,
            width: 0.85,
            points: false as const,
            box: { visible: false },
            meanline: { visible: false },
            hoveron: "violins" as const,
            hovertemplate: `<b>${gene}</b> in ${group}<extra></extra>`,
            showlegend: false,
          };
        }),
      ),
    [genes, groups, byGene, colorOf, selName],
  );

  const height = genes.length * rowHeight + 110;
  const layout = useMemo(() => {
    const n = genes.length;
    const plotH = n * rowHeight;
    const gap = Math.min(0.3, 6 / Math.max(rowHeight, 1)); // ~6 px between rows, as a share of a row
    const axes: Record<string, unknown> = {};
    genes.forEach((gene, i) => {
      const top = 1 - i / n;
      const bottom = 1 - (i + 1) / n + gap / n;
      axes[yKey(i)] = {
        domain: [bottom, top],
        title: { text: gene, font: { size: 11, color: "#334155" }, standoff: 4 },
        tickfont: AXIS_FONT,
        nticks: 3,
        gridcolor: "#f1f5f9",
        zeroline: false,
        rangemode: "tozero" as const,
        automargin: true,
      };
    });
    return {
      ...axes,
      xaxis: {
        // Category: cluster numbers as trace names would make plotly draw a numeric axis.
        type: "category" as const,
        anchor: yId(n - 1),
        tickfont: { size: 10, color: "#475569" },
        tickangle: groups.length > 6 ? -45 : 0,
        automargin: true,
      },
      showlegend: false,
      margin: { t: 8, r: 8, b: 20, l: 60 },
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      font: { family: "Inter, system-ui, sans-serif" },
      autosize: true,
      height: Math.max(height, plotH),
    };
  }, [genes, groups.length, rowHeight, height]);

  return (
    <Plot
      data={traces}
      layout={layout}
      config={plotConfig(exportName)}
      useResizeHandler
      className="w-full"
      style={{ width: "100%", height }}
    />
  );
}
