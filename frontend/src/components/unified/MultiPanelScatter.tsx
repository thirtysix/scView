import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { DeckGL, type DeckGLRef } from "@deck.gl/react";
import {
  OrbitView,
  OrthographicView,
  type Layer,
  type OrbitViewState,
  type OrthographicViewState,
  type PickingInfo,
} from "@deck.gl/core";
import { PointCloudLayer, ScatterplotLayer } from "@deck.gl/layers";
import { AlertTriangle, ImageDown, Loader2, RotateCcw, X } from "lucide-react";
import type { ScatterLegend } from "@/lib/scatterExport";
import { centroidLabelLayers, labelViewOf, placeLabels, type GroupCentroid } from "@/lib/centroidLabels";
import {
  SPLIT_CONTEXT_LAYER,
  contextColor,
  dataBounds,
  downloadSplitPng,
  fitSplitViewState,
  layerInView,
  panelOfLayer,
  splitGridLayout,
  splitLabelLayerId,
  splitLayerId,
  splitViewId,
} from "@/lib/splitView";
import { useSettingsStore } from "@/stores/settingsStore";

/**
 * Width kept free on the right of the grid for the legend: UnifiedViewPanel
 * floats the plot's legend there (right-2 top-10) while a split is on, or the
 * grid draws its own (`legend` prop).
 */
const LEGEND_COLUMN = "w-[212px]";
const CONTROLLER = { scrollZoom: { speed: 0.002, smooth: true } };

/** One panel of the grid: a set of cells, drawn in its own view. */
export interface GridPanel {
  key: string | number;
  /** Shown above the panel, with `subtitle` beside it */
  title: string;
  subtitle: string;
  /** Original indices of the cells the panel draws */
  indices: Uint32Array;
  /** Their coordinates (stride 2 or 3) and RGBA colours */
  positions: Float32Array;
  colors: Uint8Array;
  /** Group labels drawn in this panel (at its cells' centroids), or none */
  labels?: GroupCentroid[] | null;
  /** The highlighted group among them: the other labels fade */
  highlight?: string | null;
}

/** A hovered cell: original index, x/y in the scatter area's pixels, its panel. */
export interface GridHover {
  index: number;
  x: number;
  y: number;
  panel: number;
}

interface MultiPanelScatterProps {
  /** Every cell's coordinates: all panels frame the whole embedding */
  positions: Float32Array;
  dimensions: 2 | 3;
  panels: GridPanel[];
  /** Panels are built (data loaded); until then a loading message shows */
  ready: boolean;
  loadingText: string;
  /** Shown instead of the panels when the data could not be loaded */
  errorText?: string | null;
  /** Coordinates drawn as faint grey behind every panel, or null */
  context?: Float32Array | null;
  /** Left part of the header bar */
  header: ReactNode;
  /** Controls placed before the camera / export / close buttons */
  headerControls?: ReactNode;
  /** "Showing N of M cells (sampled)" under the PNG, when the panels draw a sample */
  sampledNote?: string;
  /** The same, short, for the header */
  sampledBadge?: string;
  onClose: () => void;
  closeTitle: string;
  onHover?: (info: GridHover | null) => void;
  exportLegend?: ScatterLegend | ScatterLegend[] | null;
  exportFilename: string;
  /** Content of the legend column; empty when UnifiedViewPanel floats its own there */
  legend?: ReactNode;
}

/**
 * Small multiples drawn by one Deck with one view per panel, so it costs one
 * WebGL context however many panels there are (browsers cap live contexts at
 * ~16, and the app already holds several), and keeps every panel on one camera
 * simply by giving all views the same view state. Used by the split-by grid
 * (one panel per value of a column) and the compare-colourings grid (the same
 * cells coloured by different columns).
 */
export function MultiPanelScatter({
  positions,
  dimensions,
  panels,
  ready,
  loadingText,
  errorText = null,
  context = null,
  header,
  headerControls,
  sampledNote,
  sampledBadge,
  onClose,
  closeTitle,
  onHover,
  exportLegend = null,
  exportFilename,
  legend,
}: MultiPanelScatterProps) {
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const background = useSettingsStore((s) => s.plotBackground);
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const stride = dimensions === 3 ? 3 : 2;

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
  const pointLayers = useMemo(() => {
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
    const out: Layer[] = [];
    if (context) {
      out.push(point(SPLIT_CONTEXT_LAYER, context.length / stride, context, grey, false));
    }
    panels.forEach((p, i) => {
      out.push(point(splitLayerId(i), p.indices.length, p.positions, p.colors, true));
    });
    return out;
  }, [ready, dimensions, context, stride, grey, panels, pointSize]);
  // Labels are placed for the current camera (every panel has the same view
  // size), inside the panel and off each other; drawn last, over the points.
  const labelLayers = useMemo(() => {
    if (!ready || !viewState || !views[0]) return [];
    const view = labelViewOf(views[0], area.width, area.height, viewState);
    if (!view) return [];
    return panels.flatMap((p, i) =>
      p.labels?.length
        ? centroidLabelLayers(splitLabelLayerId(i), placeLabels(p.labels, view, { highlight: p.highlight }), {
            dark: background === "dark",
            highlight: p.highlight,
          })
        : [],
    );
  }, [ready, viewState, views, area.width, area.height, panels, background]);
  const layers = useMemo(() => [...pointLayers, ...labelLayers], [pointLayers, labelLayers]);

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
        onHover?.({ index: panel.indices[info.index]!, x: info.x + area.left, y: info.y + area.top, panel: p! });
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

  // --- WebGL: release the context on unmount (opening a grid creates one) ---
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
    console.error("[MultiPanelScatter] GPU error:", err);
    if (err.message?.toLowerCase().includes("context")) setGpuError(true);
  }, []);

  const downloadPng = useCallback(() => {
    const deck = deckRef.current?.deck;
    if (!deck) return;
    downloadSplitPng(deck, {
      grid,
      panels: panels.map((p) => ({ label: p.title, count: p.subtitle })),
      legend: exportLegend,
      colormap,
      filename: exportFilename,
      note: sampledNote,
    });
  }, [grid, panels, exportLegend, colormap, exportFilename, sampledNote]);

  const dark = background === "dark";
  const bgColor = dark ? "#1e293b" : "#ffffff";

  return (
    <div
      className="absolute inset-0 z-10 flex flex-col overflow-hidden rounded-xl border border-slate-200"
      style={{ background: bgColor }}
    >
      {/* Header: what is shown, extra controls, camera + export + close */}
      <div className="flex h-8 flex-shrink-0 items-center gap-2 border-b border-slate-200 bg-white px-3 text-[11px] text-slate-500">
        <span className="truncate">{header}</span>
        {sampledBadge && <span className="flex-shrink-0 text-[10px] text-slate-400">{sampledBadge}</span>}
        <div className="flex-1" />
        {headerControls}
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
          onClick={onClose}
          title={closeTitle}
          aria-label={closeTitle}
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
              return (
                <div key={p.key} className="pointer-events-none">
                  <div
                    className="absolute flex items-baseline gap-2 truncate px-1 text-xs"
                    style={{
                      left: cell.title.x,
                      top: cell.title.y,
                      width: cell.title.width,
                      height: cell.title.height,
                      lineHeight: `${cell.title.height}px`,
                    }}
                    title={`${p.title}: ${p.subtitle}`}
                  >
                    <span className={`truncate font-semibold ${dark ? "text-slate-100" : "text-slate-700"}`}>
                      {p.title}
                    </span>
                    <span className={`flex-shrink-0 ${dark ? "text-slate-400" : "text-slate-500"}`}>{p.subtitle}</span>
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

          {!ready && !errorText && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              {loadingText}
            </div>
          )}
          {errorText && (
            <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-red-600">
              <AlertTriangle className="h-4 w-4" />
              {errorText}
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
        {/* Legend column: the grid's own legend, or UnifiedViewPanel's floats here */}
        <div className={`${LEGEND_COLUMN} flex-shrink-0 overflow-y-auto`}>{legend}</div>
      </div>
    </div>
  );
}
