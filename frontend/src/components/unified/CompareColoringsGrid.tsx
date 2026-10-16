import { useCallback, useMemo, useRef } from "react";
import { useQueries } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { categoryColor } from "@/lib/categoryColors";
import { mapCategoryToColor } from "@/lib/colors";
import { exportFilename } from "@/lib/exportFilename";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { colorByLine, type HoverLine } from "@/lib/hoverInfo";
import type { ScatterLegend } from "@/lib/scatterExport";
import { drawnIndices, gatherPositions, panelColors, sampleCellIndices } from "@/lib/splitView";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { MultiPanelScatter, type GridHover, type GridPanel } from "@/components/unified/MultiPanelScatter";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { compareLabelKey, groupCentroids, labelsShown } from "@/lib/centroidLabels";

type RGB = [number, number, number];

interface CompareColoringsGridProps {
  datasetId: string;
  embedding: string;
  /** The obs columns to show side by side, one panel each */
  columns: string[];
  positions: Float32Array;
  dimensions: 2 | 3;
  selectedIndices?: Set<number> | null;
  /** Hovered cell with x/y in the scatter area's pixels, and its value in every column */
  onHover?: (info: { index: number; x: number; y: number; lines: HoverLine[] } | null) => void;
}

// Each embedding buffer carries one column's codes; decode it once.
const decoded = new WeakMap<ArrayBuffer, Int32Array | null>();
function codesOf(buffer: ArrayBuffer, column: string): Int32Array | null {
  if (!decoded.has(buffer)) {
    const v = decodeArrowBuffer(buffer)[column];
    decoded.set(buffer, v ? Int32Array.from(v as ArrayLike<number>) : null);
  }
  return decoded.get(buffer)!;
}

/** The same list as last render when every entry is the same, so memos keep. */
function useStableList<T>(list: T[]): T[] {
  const ref = useRef(list);
  const prev = ref.current;
  if (list.length !== prev.length || list.some((v, i) => v !== prev[i])) ref.current = list;
  return ref.current;
}

/**
 * The same cells coloured by 2–4 columns side by side (two annotations, or an
 * annotation and a clustering), on one camera, with each column's legend in the
 * legend column. Built on MultiPanelScatter like the split-by grid, so it costs
 * one WebGL context; lasso is off while it shows.
 */
export function CompareColoringsGrid({
  datasetId,
  embedding,
  columns,
  positions,
  dimensions,
  selectedIndices = null,
  onHover,
}: CompareColoringsGridProps) {
  const opacity = useSettingsStore((s) => s.opacity);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setCompareColumns = useSplitViewStore((s) => s.setCompareColumns);
  const labelChoice = useUnifiedViewStore((s) => s.centroidLabels);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);

  // Each column's codes: the query useEmbedding and the split grid use, so the
  // colour-by column is not fetched twice.
  const results = useQueries({
    queries: columns.map((column) => ({
      queryKey: ["embedding", datasetId, embedding, column],
      queryFn: () => fetchEmbeddingBinary(datasetId, embedding, column),
      enabled: !!datasetId && !!embedding && !pipelineRunning,
      staleTime: 5 * 60 * 1000,
      gcTime: 60 * 1000,
    })),
  });
  const codes = useStableList(results.map((r, i) => (r.data ? codesOf(r.data, columns[i]!) : null)));
  const failed = columns.filter((_, i) => results[i]?.error || (results[i]?.data && !codes[i]));

  const categories = useMemo(
    () => columns.map((c) => dataset?.obs_columns.find((o) => o.name === c)?.values ?? []),
    [columns, dataset],
  );

  const stride = dimensions === 3 ? 3 : 2;
  const total = positions.length / stride;
  const ready = total > 0 && codes.length > 0 && codes.every((c) => c != null && c.length === total);

  // --- Every panel draws the same cells (the single plot's sample, if any) ---
  const sample = useMemo(() => sampleCellIndices(total, maxRenderedCells), [total, maxRenderedCells]);
  const indices = useMemo(() => drawnIndices(total, sample), [total, sample]);
  const drawnPositions = useMemo(
    () => (sample ? gatherPositions(positions, sample, stride) : positions),
    [sample, positions, stride],
  );
  const panelRgba = useMemo(
    () =>
      ready
        ? codes.map((c) =>
            panelColors(indices, {
              colorValues: c,
              colorType: "categorical",
              range: { min: 0, max: 1 },
              opacity,
              selected: selectedIndices,
              palette,
            }),
          )
        : [],
    [ready, codes, indices, opacity, selectedIndices, palette],
  );
  // Group labels per panel: on by default for a column with few groups, or as
  // the user toggled them for this set of columns.
  const labelKey = compareLabelKey(columns);
  const panelLabels = useMemo(
    () =>
      ready
        ? codes.map((c, i) =>
            labelsShown(labelChoice, labelKey, categories[i]!.length)
              ? groupCentroids(positions, stride as 2 | 3, c!, categories[i]!, sample)
              : null,
          )
        : [],
    [ready, codes, labelChoice, labelKey, categories, positions, stride, sample],
  );
  const panels = useMemo<GridPanel[]>(
    () =>
      ready
        ? columns.map((column, i) => ({
            key: column,
            title: prettyObsLabel(column),
            subtitle: `${categories[i]!.length} groups`,
            indices,
            positions: drawnPositions,
            colors: panelRgba[i]!,
            labels: panelLabels[i] ?? null,
            highlight: highlighted?.column === column ? highlighted.value : null,
          }))
        : [],
    [ready, columns, categories, indices, drawnPositions, panelRgba, panelLabels, highlighted],
  );

  // Legend swatches as the panels draw them (category code -> palette slot).
  const swatches = useMemo(
    () => categories.map((cats) => cats.map((_, k) => mapCategoryToColor(k, palette).slice(0, 3) as RGB)),
    [categories, palette],
  );
  const exportLegend = useMemo<ScatterLegend[]>(
    () =>
      columns.map((column, i) => ({
        type: "categorical",
        title: prettyObsLabel(column),
        categories: categories[i]!,
        colors: swatches[i]!,
      })),
    [columns, categories, swatches],
  );

  // The tooltip leads with the cell's value in every compared column.
  const handleHover = useCallback(
    (info: GridHover | null) => {
      if (!info) {
        onHover?.(null);
        return;
      }
      const lines = columns
        .map((column, i) =>
          colorByLine({
            label: prettyObsLabel(column),
            values: codes[i] ?? null,
            index: info.index,
            type: "categorical",
            categories: categories[i],
            colorOf: (_code, cat) => categoryColor(datasetId, column, categories[i]!, cat, palette),
          }),
        )
        .filter((l): l is HoverLine => l != null);
      onHover?.({ index: info.index, x: info.x, y: info.y, lines });
    },
    [columns, codes, categories, datasetId, palette, onHover],
  );

  const sampledNote = sample
    ? `Showing ${sample.length.toLocaleString()} of ${total.toLocaleString()} cells (sampled)`
    : undefined;

  return (
    <MultiPanelScatter
      positions={positions}
      dimensions={dimensions}
      panels={panels}
      ready={ready}
      loadingText={`Loading ${columns.map(prettyObsLabel).join(", ")}…`}
      errorText={failed.length > 0 ? `Could not load ${failed.map(prettyObsLabel).join(", ")} to compare.` : null}
      header={
        <>
          Compare colourings <span className="font-semibold text-slate-700">{columns.map(prettyObsLabel).join(" · ")}</span>
        </>
      }
      sampledNote={sampledNote}
      sampledBadge={
        sample ? `${(sample.length / 1000).toFixed(0)}K of ${(total / 1000).toFixed(0)}K cells (sampled)` : undefined
      }
      onClose={() => setCompareColumns(null)}
      closeTitle="Stop comparing"
      onHover={handleHover}
      exportLegend={exportLegend}
      exportFilename={exportFilename(
        dataset?.name,
        `${prettyEmbeddingLabel(embedding || "embedding")}_compare_${columns.join("_")}`,
      )}
      legend={
        <div className="flex flex-col gap-3 p-2">
          {columns.map((column, i) => (
            <ColorLegend
              key={column}
              type="categorical"
              categories={categories[i]}
              categoryColors={swatches[i]}
              label={prettyObsLabel(column)}
            />
          ))}
        </div>
      }
    />
  );
}
