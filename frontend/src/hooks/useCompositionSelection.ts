import { useMemo } from "react";
import { useCategoryOrder } from "@/hooks/useCategoryColors";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import { useObsCodes } from "@/hooks/useObsCodes";
import { barLabel, type Bar } from "@/lib/compositionBars";
import { SELECTION_COLOR, countSelectedByCode, countSelectedByPair, selectionMask } from "@/lib/linkedSelection";
import { compositionSelection, selectionName } from "@/lib/linkedTraces";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * C035 for the composition bars: one marker per bar at the selected cells'
 * share of it (or their number, with counts on the axis). The bar's group codes
 * come from the embedding buffer (fetched only while a selection exists).
 */
export function useCompositionSelection(
  datasetId: string | null,
  rowColumn: string,
  facetColumn: string | null,
  bars: readonly Bar[],
  percent: boolean,
  /** x of each bar as the bar traces have it (two arrays on a split axis) */
  x: string[] | string[][],
) {
  const embedding = useSettingsStore((s) => s.embedding);
  const { selected } = useLinkedSelection();
  const { codes: rowCodes } = useObsCodes(datasetId, embedding, selected && rowColumn ? rowColumn : null);
  const { codes: facetCodes } = useObsCodes(datasetId, embedding, selected && facetColumn ? facetColumn : null);
  const rowOrder = useCategoryOrder(rowColumn);
  const facetOrder = useCategoryOrder(facetColumn ?? undefined);

  return useMemo(() => {
    if (!selected || !rowCodes || bars.length === 0) return null;
    const mask = selectionMask(selected, rowCodes.length);
    let selectedOf: (row: string, facet: string | null) => number;
    if (facetColumn) {
      if (!facetCodes || facetCodes.length !== rowCodes.length) return null;
      const pair = countSelectedByPair(rowCodes, facetCodes, mask, rowOrder.length, facetOrder.length);
      selectedOf = (row, facet) => {
        const r = rowOrder.indexOf(row);
        const f = facet == null ? -1 : facetOrder.indexOf(facet);
        return r < 0 || f < 0 ? 0 : pair[f * rowOrder.length + r]!;
      };
    } else {
      const { selected: perCode } = countSelectedByCode(rowCodes, mask, rowOrder.length);
      selectedOf = (row) => {
        const r = rowOrder.indexOf(row);
        return r < 0 ? 0 : perCode[r]!;
      };
    }
    const s = compositionSelection(bars, selectedOf, percent);
    let n = 0;
    for (let i = 0; i < mask.length; i++) n += mask[i]!;
    return {
      type: "scatter" as const,
      mode: "markers" as const,
      x,
      y: s.y,
      name: selectionName(n),
      customdata: bars.map((b, i) => [s.counts[i]!, s.shares[i]!, barLabel(b)]),
      hovertemplate: "%{customdata[2]}<br>selected: %{customdata[0]:,} cells (%{customdata[1]:.1f}% of the bar)<extra></extra>",
      marker: { symbol: "diamond", size: 10, color: SELECTION_COLOR, line: { color: "white", width: 1 } },
    };
  }, [selected, rowCodes, facetCodes, bars, facetColumn, rowOrder, facetOrder, percent, x]);
}
