import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { exportFilename } from "@/lib/exportFilename";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import type { ScatterLegend } from "@/lib/scatterExport";
import {
  buildSplitPanels,
  effectiveColorType,
  gatherPositions,
  panelColors,
  panelTitle,
  sampleCellIndices,
  valueRange,
} from "@/lib/splitView";
import { useCategoryOrder } from "@/hooks/useCategoryColors";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { MultiPanelScatter, type GridPanel } from "@/components/unified/MultiPanelScatter";
import { groupCentroids } from "@/lib/centroidLabels";

interface SplitScatterGridProps {
  datasetId: string;
  embedding: string;
  /** Obs column to split by */
  column: string;
  positions: Float32Array;
  dimensions: 2 | 3;
  /** What the single plot is coloured by: obs codes, or a gene/score overlay */
  colorValues: Float32Array | Int32Array | null;
  colorType: "categorical" | "continuous";
  selectedIndices?: Set<number> | null;
  /** Hovered cell (original index) with x/y in this grid's (= the scatter area's) pixels */
  onHover?: (info: { index: number; x: number; y: number } | null) => void;
  exportLegend?: ScatterLegend | null;
  exportName?: string;
  /** Names of the colour codes, for the group labels */
  labelCategories?: readonly string[] | null;
  /** Draw each colour group's label at its centroid within each panel */
  showLabels?: boolean;
  highlightCategory?: string | null;
}

/**
 * Small multiples: one panel per value of `column`, each drawing that value's
 * cells over the rest in faint grey, all on one camera (MultiPanelScatter: one
 * Deck, one view per panel, the coordinates uploaded once for the grey context).
 *
 * It covers the single plot rather than replacing it: UnifiedViewPanel keeps
 * that plot mounted underneath, so leaving split mode shows it exactly as it
 * was. Lasso selection is off while split (one polygon would have to be mapped
 * through six viewports); a selection made before still fades the other cells.
 */
export function SplitScatterGrid({
  datasetId,
  embedding,
  column,
  positions,
  dimensions,
  colorValues,
  colorType,
  selectedIndices = null,
  onHover,
  exportLegend = null,
  exportName = "embedding",
  labelCategories = null,
  showLabels = false,
  highlightCategory = null,
}: SplitScatterGridProps) {
  const opacity = useSettingsStore((s) => s.opacity);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const showContext = useSplitViewStore((s) => s.showContext);
  const setShowContext = useSplitViewStore((s) => s.setShowContext);
  const setSplitBy = useSplitViewStore((s) => s.setSplitBy);
  // The column's values in its order (memoised on the dataset, so the panel
  // arrays below are not rebuilt on every parent render, e.g. each hover).
  const categories = useCategoryOrder(column);

  // The split column's value codes. Same query as useEmbedding's, so splitting
  // by the colour-by column reuses that fetch.
  const { data: buffer, error } = useQuery({
    queryKey: ["embedding", datasetId, embedding, column],
    queryFn: () => fetchEmbeddingBinary(datasetId, embedding, column),
    enabled: !!datasetId && !!embedding && !!column && !pipelineRunning,
    staleTime: 5 * 60 * 1000,
    gcTime: 60 * 1000,
  });
  const codes = useMemo(() => {
    if (!buffer) return null;
    const v = decodeArrowBuffer(buffer)[column];
    return v ? Int32Array.from(v as ArrayLike<number>) : null;
  }, [buffer, column]);

  const stride = dimensions === 3 ? 3 : 2;
  const total = positions.length / stride;
  const ready = codes != null && codes.length === total && total > 0;

  // --- Cells per panel (recomputed only when the cells or the split change) ---
  const sample = useMemo(() => sampleCellIndices(total, maxRenderedCells), [total, maxRenderedCells]);
  const { panels, missing } = useMemo(
    () => (ready ? buildSplitPanels(codes, categories, sample) : { panels: [], missing: 0 }),
    [ready, codes, categories, sample],
  );
  const panelPositions = useMemo(
    () => panels.map((p) => gatherPositions(positions, p.indices, stride)),
    [panels, positions, stride],
  );
  const contextPositions = useMemo(
    () => (sample ? gatherPositions(positions, sample, stride) : positions),
    [sample, positions, stride],
  );

  // --- Colours: one mapping and ONE range for all panels ---
  const range = useMemo(
    () =>
      effectiveColorType(colorValues, colorType) === "continuous"
        ? valueRange(colorValues)
        : { min: 0, max: 1 },
    [colorValues, colorType],
  );
  const panelRgba = useMemo(
    () =>
      panels.map((p) =>
        panelColors(p.indices, {
          colorValues,
          colorType,
          range,
          opacity,
          selected: selectedIndices,
          palette,
          colormap,
        }),
      ),
    [panels, colorValues, colorType, range, opacity, selectedIndices, palette, colormap],
  );

  const drawnCells = sample ? sample.length : total;
  const sampledNote = sample
    ? `Showing ${drawnCells.toLocaleString()} of ${total.toLocaleString()} cells (sampled)`
    : undefined;

  // Each panel labels the colour groups it has cells of, where most of them are in it.
  const panelLabels = useMemo(
    () =>
      showLabels && labelCategories && colorValues instanceof Int32Array && colorType === "categorical"
        ? panels.map((p) => groupCentroids(positions, stride as 2 | 3, colorValues, labelCategories, p.indices))
        : null,
    [showLabels, labelCategories, colorValues, colorType, panels, positions, stride],
  );

  const gridPanels = useMemo<GridPanel[]>(
    () =>
      panels.map((p, i) => {
        const { label, count } = panelTitle(p);
        return {
          key: p.code,
          title: label,
          subtitle: count,
          indices: p.indices,
          positions: panelPositions[i]!,
          colors: panelRgba[i]!,
          labels: panelLabels?.[i] ?? null,
          highlight: highlightCategory,
        };
      }),
    [panels, panelPositions, panelRgba, panelLabels, highlightCategory],
  );

  return (
    <MultiPanelScatter
      positions={positions}
      dimensions={dimensions}
      panels={gridPanels}
      ready={ready}
      loadingText={`Loading ${prettyObsLabel(column)}…`}
      errorText={error ? `Could not load ${prettyObsLabel(column)} to split by.` : null}
      context={showContext ? contextPositions : null}
      header={
        <>
          Split by <span className="font-semibold text-slate-700">{prettyObsLabel(column)}</span>
          {ready && ` · ${panels.length} panels`}
          {missing > 0 && ` · ${formatNumber(missing)} cells without a value not shown`}
        </>
      }
      headerControls={
        <label
          className="flex flex-shrink-0 cursor-pointer items-center gap-1"
          title="Draw the other panels' cells as faint grey behind each panel"
        >
          <input
            type="checkbox"
            checked={showContext}
            onChange={(e) => setShowContext(e.target.checked)}
            className="h-3 w-3 accent-blue-500"
          />
          Other cells
        </label>
      }
      sampledNote={sampledNote}
      sampledBadge={
        sample ? `${(drawnCells / 1000).toFixed(0)}K of ${(total / 1000).toFixed(0)}K cells (sampled)` : undefined
      }
      onClose={() => setSplitBy(null)}
      closeTitle="Stop splitting"
      onHover={onHover}
      exportLegend={exportLegend}
      exportFilename={exportFilename(useDatasetStore.getState().currentDataset?.name, `${exportName}_split_${column}`)}
    />
  );
}
