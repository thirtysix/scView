import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { DeckGL, type DeckGLRef } from "@deck.gl/react";
import {
  OrbitView,
  OrthographicView,
  type OrbitViewState,
  type OrthographicViewState,
  type PickingInfo,
} from "@deck.gl/core";
import { PointCloudLayer, ScatterplotLayer } from "@deck.gl/layers";
import { AlertTriangle, ImageDown, Loader2, RotateCcw, X } from "lucide-react";
import { fetchEmbeddingBinary } from "@/api/embeddings";
import { decodeArrowBuffer } from "@/lib/arrow";
import { exportFilename } from "@/lib/exportFilename";
import { formatNumber, prettyObsLabel } from "@/lib/formatting";
import type { ScatterLegend } from "@/lib/scatterExport";
import {
  SPLIT_CONTEXT_LAYER,
  buildSplitPanels,
  contextColor,
  dataBounds,
  downloadSplitPng,
  effectiveColorType,
  fitSplitViewState,
  gatherPositions,
  layerInView,
  panelColors,
  panelOfLayer,
  panelTitle,
  sampleCellIndices,
  splitGridLayout,
  splitLayerId,
  splitViewId,
  valueRange,
} from "@/lib/splitView";
import { useCategoryOrder } from "@/hooks/useCategoryColors";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

/**
 * Width kept free on the right of the grid for the plot's legend, which
 * UnifiedViewPanel floats there (right-2 top-10) while a split is on: one
 * legend for every panel.
 */
const LEGEND_COLUMN = "w-[212px]";
const CONTROLLER = { scrollZoom: { speed: 0.002, smooth: true } };

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
}

/**
 * Small multiples: one panel per value of `column`, each drawing that value's
 * cells over the rest in faint grey. One Deck with one view per panel, so it
 * costs one WebGL context however many panels there are (browsers cap live
 * contexts at ~16, and the app already holds several), uploads the coordinates
 * once for the grey context layer, and keeps every panel on one camera simply
 * by giving all views the same view state.
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
}: SplitScatterGridProps) {
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const background = useSettingsStore((s) => s.plotBackground);
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

  // --- Layout ---
  const areaRef = useRef<HTMLDivElement>(null);
  const [area, setArea] = useState({ width: 0, height: 0, left: 0, top: 0 });
  useEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      // Offsets are from inside the grid's border; the tooltip is placed from outside it.
      const border = el.offsetParent?.clientLeft ?? 0;
      if (r) {
        setArea({
          width: r.width,
          height: r.height,
          left: el.offsetLeft + border,
          top: el.offsetTop + border,
        });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const bounds = useMemo(() => dataBounds(positions, stride), [positions, stride]);
  const aspect =
    dimensions === 3 ? 1 : (bounds.maxX - bounds.minX || 1) / (bounds.maxY - bounds.minY || 1);
  const grid = useMemo(
    () => splitGridLayout(panels.length, area.width, area.height, aspect),
    [panels.length, area.width, area.height, aspect],
  );
  const viewW = grid.cells[0]?.view.width ?? 0;
  const viewH = grid.cells[0]?.view.height ?? 0;

  // --- One camera for every panel ---
  const fitted = useMemo(
    () => (viewW > 0 && viewH > 0 ? fitSplitViewState(bounds, dimensions, viewW, viewH) : null),
    [bounds, dimensions, viewW, viewH],
  );
  const [viewState, setViewState] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    if (fitted) setViewState({ ...fitted });
  }, [fitted]);
  const resetView = useCallback(() => {
    if (fitted) setViewState({ ...fitted });
  }, [fitted]);
  const handleViewStateChange = useCallback(
    ({ viewState: vs }: { viewState: Record<string, unknown> }) => setViewState(vs),
    [],
  );

  const views = useMemo(
    () =>
      grid.cells.map(({ view: v }, i) =>
        dimensions === 3
          ? new OrbitView({ id: splitViewId(i), ...v, controller: CONTROLLER, orbitAxis: "Y" })
          : new OrthographicView({ id: splitViewId(i), ...v, flipY: false, controller: CONTROLLER }),
      ),
    [grid, dimensions],
  );
  // Every view gets the same state: that is what links the panels' cameras.
  const viewStates = useMemo(() => {
    if (!viewState) return null;
    const vs = viewState as OrbitViewState | OrthographicViewState;
    return Object.fromEntries(views.map((v) => [v.id, vs]));
  }, [views, viewState]);

  const grey = useMemo(() => contextColor(background, opacity), [background, opacity]);
  const layers = useMemo(() => {
    if (!ready) return [];
    const point = (
      id: string,
      length: number,
      pos: Float32Array,
      color: Uint8Array | [number, number, number, number],
      pickable: boolean,
    ) => {
      const colorAttr =
        color instanceof Uint8Array ? { value: color, size: 4 } : undefined;
      if (dimensions === 3) {
        return new PointCloudLayer({
          id,
          data: {
            length,
            attributes: {
              getPosition: { value: pos, size: 3 },
              ...(colorAttr ? { getColor: colorAttr } : {}),
            },
          },
          ...(colorAttr ? {} : { getColor: color as [number, number, number, number] }),
          pointSize: pointSize * 2,
          pickable,
          autoHighlight: pickable,
          highlightColor: [255, 255, 0, 180],
          updateTriggers: { getColor: [color], pointSize: [pointSize] },
        });
      }
      return new ScatterplotLayer({
        id,
        data: {
          length,
          attributes: {
            getPosition: { value: pos, size: 2 },
            ...(colorAttr ? { getFillColor: colorAttr } : {}),
          },
        },
        ...(colorAttr ? {} : { getFillColor: color as [number, number, number, number] }),
        getRadius: pointSize,
        radiusUnits: "pixels" as const,
        radiusMinPixels: 0.5,
        radiusMaxPixels: 20,
        pickable,
        autoHighlight: pickable,
        highlightColor: [255, 255, 0, 180],
        updateTriggers: { getFillColor: [color], getRadius: [pointSize] },
      });
    };
    const out = [];
    if (showContext) {
      out.push(point(SPLIT_CONTEXT_LAYER, contextPositions.length / stride, contextPositions, grey, false));
    }
    panels.forEach((p, i) => {
      out.push(point(splitLayerId(i), p.indices.length, panelPositions[i]!, panelRgba[i]!, true));
    });
    return out;
  }, [ready, dimensions, showContext, contextPositions, stride, grey, panels, panelPositions, panelRgba, pointSize]);

  const layerFilter = useCallback(
    ({ layer, viewport }: { layer: { id: string }; viewport: { id: string } }) =>
      layerInView(layer.id, viewport.id),
    [],
  );

  // Hover reports the original cell index, with x/y moved from the canvas into
  // the scatter area so UnifiedViewPanel's tooltip lands next to the cursor.
  const handleHover = useCallback(
    (info: PickingInfo) => {
      const p = panelOfLayer(info.layer?.id);
      const panel = p == null ? undefined : panels[p];
      if (info.picked && panel && info.index >= 0 && info.index < panel.indices.length) {
        onHover?.({ index: panel.indices[info.index]!, x: info.x + area.left, y: info.y + area.top });
      } else {
        onHover?.(null);
      }
    },
    [panels, onHover, area.left, area.top],
  );
  // Clear a tooltip left over from the single plot, or from this grid on exit.
  useEffect(() => {
    onHover?.(null);
    return () => onHover?.(null);
  }, [onHover]);

  // --- WebGL: release the context on unmount (toggling split creates one) ---
  const deckRef = useRef<DeckGLRef>(null);
  const [gpuError, setGpuError] = useState(false);
  const [renderKey, setRenderKey] = useState(0);
  useEffect(() => {
    const el = areaRef.current;
    return () => {
      const canvas = el?.querySelector("canvas");
      const gl =
        (canvas?.getContext("webgl2") as WebGL2RenderingContext | null) ??
        (canvas?.getContext("webgl") as WebGLRenderingContext | null);
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);
  const handleGpuError = useCallback((err: Error) => {
    console.error("[SplitScatterGrid] GPU error:", err);
    if (err.message?.toLowerCase().includes("context")) setGpuError(true);
  }, []);

  const drawnCells = sample ? sample.length : total;
  const sampledNote = sample
    ? `Showing ${drawnCells.toLocaleString()} of ${total.toLocaleString()} cells (sampled)`
    : undefined;

  const downloadPng = useCallback(() => {
    const deck = deckRef.current?.deck;
    if (!deck) return;
    downloadSplitPng(deck, {
      grid,
      panels,
      legend: exportLegend,
      colormap,
      filename: exportFilename(
        useDatasetStore.getState().currentDataset?.name,
        `${exportName}_split_${column}`,
      ),
      note: sampledNote,
    });
  }, [grid, panels, exportLegend, colormap, exportName, column, sampledNote]);

  const dark = background === "dark";
  const bgColor = dark ? "#1e293b" : "#ffffff";

  return (
    <div
      className="absolute inset-0 z-10 flex flex-col overflow-hidden rounded-xl border border-slate-200"
      style={{ background: bgColor }}
    >
      {/* Header: what is split, context toggle, camera + export */}
      <div className="flex h-8 flex-shrink-0 items-center gap-2 border-b border-slate-200 bg-white px-3 text-[11px] text-slate-500">
        <span className="truncate">
          Split by <span className="font-semibold text-slate-700">{prettyObsLabel(column)}</span>
          {ready && ` · ${panels.length} panels`}
          {missing > 0 && ` · ${formatNumber(missing)} cells without a value not shown`}
        </span>
        {sample && (
          <span className="flex-shrink-0 text-[10px] text-slate-400">
            {(drawnCells / 1000).toFixed(0)}K of {(total / 1000).toFixed(0)}K cells (sampled)
          </span>
        )}
        <div className="flex-1" />
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
        <button
          onClick={resetView}
          title="Reset view"
          aria-label="Reset view"
          className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={downloadPng}
          disabled={!ready || !viewState || gpuError}
          title="Download PNG of all panels (high resolution, white background, with legend)"
          aria-label="Download PNG"
          className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700 disabled:opacity-40"
        >
          <ImageDown className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => setSplitBy(null)}
          title="Back to one plot"
          aria-label="Stop splitting"
          className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1">
        <div ref={areaRef} className="relative m-2 min-w-0 flex-1">
          {ready && viewStates && views.length > 0 && !gpuError && (
            <DeckGL
              key={renderKey}
              ref={deckRef}
              views={views}
              viewState={viewStates}
              onViewStateChange={handleViewStateChange}
              layers={layers}
              layerFilter={layerFilter}
              onHover={handleHover}
              onError={handleGpuError}
              getCursor={({ isDragging }: { isDragging: boolean }) =>
                isDragging ? "grabbing" : "crosshair"
              }
            />
          )}

          {/* Titles + panel frames (the canvas underneath takes the events) */}
          {ready &&
            grid.cells.map((cell, i) => {
              const p = panels[i];
              if (!p) return null;
              const { label, count } = panelTitle(p);
              return (
                <div key={p.code} className="pointer-events-none">
                  <div
                    className="absolute flex items-baseline gap-2 truncate px-1 text-xs"
                    style={{
                      left: cell.title.x,
                      top: cell.title.y,
                      width: cell.title.width,
                      height: cell.title.height,
                      lineHeight: `${cell.title.height}px`,
                    }}
                    title={`${label}: ${count}`}
                  >
                    <span className={`truncate font-semibold ${dark ? "text-slate-100" : "text-slate-700"}`}>
                      {label}
                    </span>
                    <span className={`flex-shrink-0 ${dark ? "text-slate-400" : "text-slate-500"}`}>{count}</span>
                  </div>
                  <div
                    className={`absolute rounded border ${dark ? "border-slate-600" : "border-slate-200"}`}
                    style={{
                      left: cell.view.x,
                      top: cell.view.y,
                      width: cell.view.width,
                      height: cell.view.height,
                    }}
                  />
                </div>
              );
            })}

          {!ready && !error && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              Loading {prettyObsLabel(column)}…
            </div>
          )}
          {error && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-red-600">
              <AlertTriangle className="h-4 w-4" />
              Could not load {prettyObsLabel(column)} to split by.
            </div>
          )}
          {gpuError && (
            <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-2 rounded-lg bg-slate-900/80 text-white">
              <AlertTriangle className="h-6 w-6 text-amber-400" />
              <p className="text-sm font-medium">GPU context lost</p>
              <button
                onClick={() => {
                  setGpuError(false);
                  setRenderKey((k) => k + 1);
                }}
                className="rounded-md border border-slate-500 bg-slate-700 px-4 py-1.5 text-xs font-medium hover:bg-slate-600"
              >
                Retry
              </button>
            </div>
          )}
        </div>
        {/* Legend column: UnifiedViewPanel's legend floats here */}
        <div className={`${LEGEND_COLUMN} flex-shrink-0`} />
      </div>
    </div>
  );
}
