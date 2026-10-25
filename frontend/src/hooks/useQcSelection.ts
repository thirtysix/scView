import { useCallback, useEffect, useMemo, useState } from "react";
import { useLinkedObsValues } from "@/hooks/useLinkedObsValues";
import { useLinkedSelection } from "@/hooks/useLinkedSelection";
import {
  SELECTION_COLOR,
  binSelected,
  cellsInPlotlySelection,
  selectionMask,
  type PlotlySelection,
} from "@/lib/linkedSelection";
import { selectedPoints, selectionName } from "@/lib/linkedTraces";
import { useSettingsStore } from "@/stores/settingsStore";

/** Selected points drawn on a QC scatter at most (the server's own scatter is a sample too). */
const MAX_SELECTED_POINTS = 10000;

/**
 * C035 for a QC histogram: the selected cells' counts in the histogram's own
 * bins, from the metric's per-cell values (an obs column; none when the metric
 * is only computed on the fly, so nothing is overlaid).
 */
export function useQcHistogramSelection(datasetId: string | undefined, metric: string | undefined, edges: number[]) {
  const embedding = useSettingsStore((s) => s.embedding);
  const { selected } = useLinkedSelection();
  const values = useLinkedObsValues(datasetId ?? null, embedding, selected && metric ? metric : null);
  return useMemo(() => {
    if (!selected || !values) return null;
    const mask = selectionMask(selected, values.length);
    let n = 0;
    for (let i = 0; i < mask.length; i++) n += mask[i]!;
    return { counts: binSelected(values, mask, edges), name: selectionName(n), color: SELECTION_COLOR };
  }, [selected, values, edges]);
}

/**
 * C035 for a QC per-cell scatter (x and y are obs columns): the selected cells
 * drawn on top, and the reverse: a box or lasso drawn on the plot selects every
 * cell inside it, from the full per-cell values (fetched when first needed).
 */
export function useQcScatterSelection(datasetId: string | undefined, xColumn: string, yColumn: string) {
  const embedding = useSettingsStore((s) => s.embedding);
  const { selected, select } = useLinkedSelection();
  const [pending, setPending] = useState<PlotlySelection | null>(null);
  const want = !!selected || !!pending;
  const x = useLinkedObsValues(datasetId ?? null, embedding, want ? xColumn : null);
  const y = useLinkedObsValues(datasetId ?? null, embedding, want ? yColumn : null);

  useEffect(() => {
    if (!pending || !x || !y) return;
    const cells = cellsInPlotlySelection(x, y, pending);
    setPending(null);
    if (cells) select(cells);
  }, [pending, x, y, select]);

  const trace = useMemo(() => {
    if (!selected || !x || !y) return null;
    const p = selectedPoints(x, y, selected, MAX_SELECTED_POINTS);
    return {
      type: "scattergl" as const,
      mode: "markers" as const,
      x: p.x,
      y: p.y,
      name: selectionName(p.n),
      marker: { size: 4, color: SELECTION_COLOR, opacity: 0.85 },
      hovertemplate: `selected<br>${xColumn} %{x}<br>${yColumn} %{y}<extra></extra>`,
      showlegend: false,
    };
  }, [selected, x, y, xColumn, yColumn]);

  const onSelected = useCallback((e: Readonly<PlotlySelection> | undefined) => {
    if (e && (e.range || e.lassoPoints)) setPending({ range: e.range, lassoPoints: e.lassoPoints });
  }, []);

  return { trace, onSelected, dim: !!trace };
}
