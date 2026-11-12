import { useCallback, useMemo, useRef } from "react";
import { useQueries } from "@tanstack/react-query";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import type { ObsColumnInfo } from "@/api/types";
import { useEmbeddingReady } from "@/hooks/useEmbeddingReady";
import { decodeArrowBuffer } from "@/lib/arrow";
import { categoryColor } from "@/lib/categoryColors";
import { mapCategoryToColor } from "@/lib/colors";
import { overlayStats, resolveColorScale, type ResolvedColorScale } from "@/lib/colorScale";
import { exportFilename } from "@/lib/exportFilename";
import { prettyEmbeddingLabel, prettyObsLabel } from "@/lib/formatting";
import { colorByLine, type HoverLine } from "@/lib/hoverInfo";
import { isCategoricalDtype } from "@/lib/obsColumns";
import { isScoreColumn, scoreDetail, scoreTitle } from "@/lib/savedScores";
import type { ScatterLegend } from "@/lib/scatterExport";
import { drawnIndices, gatherPositions, panelColors, sampleCellIndices } from "@/lib/splitView";
import { selectedLast, shuffledOrder } from "@/lib/drawOrder";
import { ColorLegend } from "@/components/plots/ColorLegend";
import { MultiPanelScatter, type GridHover, type GridPanel } from "@/components/unified/MultiPanelScatter";
import { useSameCellsGridLasso } from "@/components/unified/GridLasso";
import { SelectionChip } from "@/components/common/SelectionChip";
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
  /** A lasso closed in any panel: the cells under it */
  onLassoComplete?: (cells: Set<number>) => void;
}

// Each embedding buffer carries one column's values: category codes, or a score's
// values (lib/savedScores.ts) for a continuous column. Decode it once.
const decoded = new WeakMap<ArrayBuffer, Int32Array | Float32Array | null>();
function valuesOf(buffer: ArrayBuffer, column: string, continuous: boolean): Int32Array | Float32Array | null {
  if (!decoded.has(buffer)) {
    const v = decodeArrowBuffer(buffer)[column];
    const out = !v ? null : continuous ? Float32Array.from(v as ArrayLike<number>) : Int32Array.from(v as ArrayLike<number>);
    decoded.set(buffer, out);
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

/** A column drawn by value with a colour bar (a score), not by category. */
const isContinuous = (info: ObsColumnInfo | undefined): boolean => !!info && !isCategoricalDtype(info.dtype);

/**
 * The same cells coloured by 2–12 columns side by side (two annotations, an
 * annotation and a clustering, or a grouping and saved scores), on one camera,
 * with each column's legend in the legend column: a category list, or a colour
 * bar for a score (the scatter's own scale decision, lib/colorScale.ts). Built on
 * MultiPanelScatter like the split-by grid, so it costs one WebGL context; a lasso
 * in any panel selects the cells under it.
 */
export function CompareColoringsGrid({
  datasetId,
  embedding,
  columns,
  positions,
  dimensions,
  selectedIndices = null,
  onHover,
  onLassoComplete,
}: CompareColoringsGridProps) {
  const lasso = useSameCellsGridLasso(positions, dimensions, onLassoComplete);
  const opacity = useSettingsStore((s) => s.opacity);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const maxRenderedCells = useSettingsStore((s) => s.maxRenderedCells);
  const pipelineRunning = useSettingsStore((s) => s.pipelineRunning);
  const sequentialColormap = useSettingsStore((s) => s.continuousColormap);
  const divergingColormap = useSettingsStore((s) => s.divergingColormap);
  const signedColormapMode = useSettingsStore((s) => s.signedColormapMode);
  const defaultClip = useSettingsStore((s) => s.colorClip);
  const viewClip = useUnifiedViewStore((s) => s.colorClip);
  const setViewClip = useUnifiedViewStore((s) => s.setColorClip);
  const clipSetting = viewClip ?? defaultClip;
  const embeddingOk = useEmbeddingReady(datasetId, embedding);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const setCompareColumns = useSplitViewStore((s) => s.setCompareColumns);
  const labelChoice = useUnifiedViewStore((s) => s.centroidLabels);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);

  const infos = useMemo(
    () => columns.map((c) => dataset?.obs_columns.find((o) => o.name === c)),
    [columns, dataset],
  );
  const continuous = useStableList(infos.map(isContinuous));

  // Each column's codes: the query useEmbedding and the split grid use, so the
  // colour-by column is not fetched twice.
  const results = useQueries({
    queries: columns.map((column) => ({
      queryKey: ["embedding", datasetId, embedding, column],
      queryFn: () => fetchEmbeddingBinary(datasetId, embedding, column),
      enabled: embeddingOk && !pipelineRunning,
      staleTime: 5 * 60 * 1000,
      gcTime: 60 * 1000,
    })),
  });
  const codes = useStableList(
    results.map((r, i) => (r.data ? valuesOf(r.data, columns[i]!, continuous[i]!) : null)),
  );
  const failed = columns.filter((_, i) => results[i]?.error || (results[i]?.data && !codes[i]));

  const categories = useMemo(() => infos.map((o) => (o && !isContinuous(o) ? o.values ?? [] : [])), [infos]);
  const slotLists = useMemo(() => infos.map((o) => o?.color_slots), [infos]);

  const stride = dimensions === 3 ? 3 : 2;
  const total = positions.length / stride;
  const ready = total > 0 && codes.length > 0 && codes.every((c) => c != null && c.length === total);

  // A score panel's colour scale, decided as the single plot decides it (diverging
  // about 0 for a signed score, the clipping in force). Sorted once per column.
  const stats = useMemo(
    () => (ready ? codes.map((c, i) => (continuous[i] ? overlayStats(c!) : null)) : []),
    [ready, codes, continuous],
  );
  const scales = useMemo<(ResolvedColorScale | null)[]>(() => {
    const settings = { sequential: sequentialColormap, diverging: divergingColormap, signedMode: signedColormapMode, clip: clipSetting };
    return stats.map((st, i) => (st ? resolveColorScale(infos[i] && isScoreColumn(infos[i]!) ? "score" : "obs", st, settings) : null));
  }, [stats, infos, sequentialColormap, divergingColormap, signedColormapMode, clipSetting]);

  // --- Every panel draws the same cells (the single plot's sample, if any) ---
  const sample = useMemo(() => sampleCellIndices(total, maxRenderedCells), [total, maxRenderedCells]);
  // ...in the single plot's seeded draw order, a selection last (lib/drawOrder.ts):
  // slot s of every panel is cell indices[s].
  const shuffled = useMemo(() => shuffledOrder(drawnIndices(total, sample)), [total, sample]);
  const indices = useMemo(() => selectedLast(shuffled, selectedIndices), [shuffled, selectedIndices]);
  const drawnPositions = useMemo(() => gatherPositions(positions, indices, stride), [positions, indices, stride]);
  const panelRgba = useMemo(
    () =>
      ready && scales.length === codes.length
        ? codes.map((c, i) => {
            const scale = scales[i];
            return panelColors(indices, {
              colorValues: c,
              colorType: scale ? "continuous" : "categorical",
              range: scale ?? { min: 0, max: 1 },
              colormap: scale?.colormap,
              opacity,
              selected: selectedIndices,
              palette,
              slots: slotLists[i],
            });
          })
        : [],
    [ready, codes, scales, indices, opacity, selectedIndices, palette, slotLists],
  );
  // Group labels per panel: on by default for a column with few groups, or as
  // the user toggled them for this set of columns. A score panel has none.
  const labelKey = compareLabelKey(columns);
  const panelLabels = useMemo(
    () =>
      ready
        ? codes.map((c, i) =>
            !continuous[i] && labelsShown(labelChoice, labelKey, categories[i]!.length)
              ? groupCentroids(positions, stride as 2 | 3, c as Int32Array, categories[i]!, sample)
              : null,
          )
        : [],
    [ready, codes, continuous, labelChoice, labelKey, categories, positions, stride, sample],
  );
  const titles = useMemo(
    () => columns.map((c, i) => (continuous[i] && infos[i] ? scoreTitle(infos[i]!) : prettyObsLabel(c))),
    [columns, continuous, infos],
  );
  const panels = useMemo<GridPanel[]>(
    () =>
      ready && panelRgba.length === columns.length
        ? columns.map((column, i) => ({
            key: column,
            title: titles[i]!,
            subtitle: continuous[i] ? scoreDetail(infos[i]!) : `${categories[i]!.length} groups`,
            indices,
            positions: drawnPositions,
            colors: panelRgba[i]!,
            labels: panelLabels[i] ?? null,
            highlight: highlighted?.column === column ? highlighted.value : null,
          }))
        : [],
    [ready, columns, titles, continuous, infos, categories, indices, drawnPositions, panelRgba, panelLabels, highlighted],
  );

  // Legend swatches as the panels draw them (category code -> palette slot).
  const swatches = useMemo(
    () =>
      categories.map((cats, i) => cats.map((_, k) => mapCategoryToColor(k, palette, slotLists[i]).slice(0, 3) as RGB)),
    [categories, palette, slotLists],
  );
  const exportLegend = useMemo<ScatterLegend[]>(
    () =>
      columns.map((_column, i): ScatterLegend => {
        const scale = scales[i];
        return scale
          ? { type: "continuous", title: titles[i], min: scale.min, max: scale.max, colormap: scale.colormap, centre: scale.centre }
          : { type: "categorical", title: titles[i], categories: categories[i]!, colors: swatches[i]! };
      }),
    [columns, scales, titles, categories, swatches],
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
          continuous[i]
            ? colorByLine({ label: titles[i]!, values: codes[i] ?? null, index: info.index, type: "continuous" })
            : colorByLine({
                label: titles[i]!,
                values: codes[i] ?? null,
                index: info.index,
                type: "categorical",
                categories: categories[i],
                colorOf: (_code, cat) => categoryColor(datasetId, column, categories[i]!, cat, palette, slotLists[i]),
              }),
        )
        .filter((l): l is HoverLine => l != null);
      onHover?.({ index: info.index, x: info.x, y: info.y, lines });
    },
    [columns, continuous, titles, codes, categories, slotLists, datasetId, palette, onHover],
  );

  const sampledNote = sample
    ? `Showing ${sample.length.toLocaleString()} of ${total.toLocaleString()} cells (sampled)`
    : undefined;
  // The clip control sits under the page's first colour bar; it applies to every score panel.
  const firstScale = (start: number, end: number) => scales.slice(start, end).findIndex((s) => s != null) + start;

  return (
    <MultiPanelScatter
      positions={positions}
      dimensions={dimensions}
      panels={panels}
      ready={ready && panels.length > 0}
      loadingText={`Loading ${titles.join(", ")}…`}
      errorText={failed.length > 0 ? `Could not load ${failed.map(prettyObsLabel).join(", ")} to compare.` : null}
      header={
        <>
          Compare colourings <span className="font-semibold text-slate-700">{titles.join(" · ")}</span>
        </>
      }
      sampledNote={sampledNote}
      sampledBadge={
        sample ? `${(sample.length / 1000).toFixed(0)}K of ${(total / 1000).toFixed(0)}K cells (sampled)` : undefined
      }
      onClose={() => setCompareColumns(null)}
      closeTitle="Stop comparing"
      onHover={handleHover}
      headerControls={<SelectionChip />}
      lassoActive={lasso.lassoActive}
      onLasso={lasso.onLasso}
      exportLegend={exportLegend}
      legendPerPanel
      exportFilename={exportFilename(
        dataset?.name,
        `${prettyEmbeddingLabel(embedding || "embedding")}_compare_${columns.join("_")}`,
      )}
      embedding={embedding}
      // The legends of the page's columns (a paged comparison shows the others on their pages).
      legend={({ start, end }) => (
        <div className="flex flex-col gap-3 p-2">
          {columns.slice(start, end).map((column, k) => {
            const i = start + k;
            const scale = scales[i];
            return scale ? (
              <ColorLegend
                key={column}
                type="continuous"
                scale={scale}
                label={titles[i]}
                {...(i === firstScale(start, end)
                  ? { clip: clipSetting, onClipChange: setViewClip, clipOverridden: viewClip !== null }
                  : {})}
              />
            ) : (
              <ColorLegend
                key={column}
                type="categorical"
                categories={categories[i]}
                categoryColors={swatches[i]}
                label={titles[i]}
              />
            );
          })}
        </div>
      )}
    />
  );
}
