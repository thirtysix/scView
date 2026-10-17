import { useCallback, useMemo, useRef, useState, useEffect } from "react";
import { DeckGL, type DeckGLRef } from "@deck.gl/react";
import { OrthographicView, OrbitView } from "@deck.gl/core";
import { ScatterplotLayer, PointCloudLayer } from "@deck.gl/layers";
import { mapCategoryToColor, mapValueToColor } from "@/lib/colors";
import { useSettingsStore } from "@/stores/settingsStore";
import { RotateCcw, AlertTriangle, ImageDown } from "lucide-react";
import { downloadDeckPng, type ScatterLegend } from "@/lib/scatterExport";
import { exportFilename } from "@/lib/exportFilename";
import { useDatasetStore } from "@/stores/datasetStore";
import { colorByLine } from "@/lib/hoverInfo";
import { centroidLabelLayers, groupCentroids, labelViewOf, placeLabels } from "@/lib/centroidLabels";
import { HoverLineRow } from "@/components/plots/HoverLineRow";

interface EmbeddingScatterProps {
  positions: Float32Array | null;
  colorValues: Float32Array | Int32Array | null;
  colorType: "categorical" | "continuous";
  pointSize?: number;
  opacity?: number;
  onHover?: (info: { index: number; x: number; y: number } | null) => void;
  /** Fired on a click (not a drag) with the original cell index, or null when
   * clicking empty space. deck.gl only emits this for genuine clicks, so it
   * coexists with drag-to-rotate/pan. */
  onClick?: (index: number | null) => void;
  onViewStateChange?: (viewState: Record<string, unknown>) => void;
  selectedIndices?: Set<number> | null;
  background?: "white" | "dark";
  maxRenderedCells?: number;
  dimensions?: 2 | 3;
  /** Compact reference mode: hides the reset button, sampling badge and tooltip. */
  minimal?: boolean;
  /** When dimming non-selected cells, fade them to grey (still visible as
   * context) instead of just lowering their alpha (which can make light colours
   * vanish). Used by the cluster reference map. */
  dimToGray?: boolean;
  /** When provided, the camera follows this external view state (orientation +
   * pan) while keeping its OWN zoom, so two linked plots rotate/pan in
   * conjunction yet each stays fitted to its container. deck.gl only emits
   * onViewStateChange on user interaction, so this won't feedback-loop. */
  externalViewState?: Record<string, unknown> | null;
  /** A camera to show instead of the fitted one (a restored session): it is
   *  kept when the data reloads or the plot resizes, until the user moves the
   *  camera or resets the view. `onRestoreViewState` fires once it is shown. */
  restoreViewState?: Record<string, unknown> | null;
  onRestoreViewState?: () => void;
  /** What the camera is on (dataset + embedding): a change drops a restored camera. */
  cameraKey?: string;
  /** Every camera change, whatever caused it; `fitted` while the camera is the
   *  automatic fit to the data. */
  onCameraChange?: (viewState: Record<string, unknown>, fitted: boolean) => void;
  /** Legend drawn beside the points in the "Download PNG" image */
  exportLegend?: ScatterLegend | null;
  /** Plot part of the PNG's filename (scview_<dataset>_<exportName>.png) */
  exportName?: string;
  /** What the points are coloured by (gene, score, column), for the built-in tooltip */
  colorLabel?: string;
  /** Category names in code order (categorical colouring), for the built-in
   *  tooltip and the group labels */
  categories?: readonly string[] | null;
  /** Draw each category's name at its centroid (categorical colouring only) */
  showLabels?: boolean;
  /** A highlighted category: its label stays, the others fade */
  highlightCategory?: string | null;
}

/**
 * Compute the initial view state that auto-fits the data bounds into the
 * viewport with some padding. Supports 2D and 3D.
 */
function computeInitialViewState2D(
  positions: Float32Array,
  width: number,
  height: number,
) {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (let i = 0; i < positions.length; i += 2) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const dataWidth = maxX - minX || 1;
  const dataHeight = maxY - minY || 1;
  const padding = 0.05;
  const paddedW = dataWidth * (1 + padding * 2);
  const paddedH = dataHeight * (1 + padding * 2);

  const scaleX = (width || 800) / paddedW;
  const scaleY = (height || 600) / paddedH;
  const zoom = Math.log2(Math.min(scaleX, scaleY));

  return {
    target: [minX + dataWidth / 2, minY + dataHeight / 2, 0] as [
      number,
      number,
      number,
    ],
    zoom,
    minZoom: zoom - 4,
    maxZoom: zoom + 10,
  };
}

function computeInitialViewState3D(
  positions: Float32Array,
  width = 600,
  height = 600,
) {
  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;
  let minZ = Infinity, maxZ = -Infinity;

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z < minZ) minZ = z;
    if (z > maxZ) maxZ = z;
  }

  const dataWidth = maxX - minX || 1;
  const dataHeight = maxY - minY || 1;
  const dataDepth = maxZ - minZ || 1;
  const maxDim = Math.max(dataWidth, dataHeight, dataDepth);

  return {
    target: [
      minX + dataWidth / 2,
      minY + dataHeight / 2,
      minZ + dataDepth / 2,
    ] as [number, number, number],
    // Fit to the smaller container dimension (was a fixed 400px) so a small
    // container — e.g. the cluster reference map — pulls back enough to show
    // all clusters instead of cropping in.
    zoom: Math.log2((Math.min(width, height) * 0.85) / maxDim),
    rotationX: 30,
    rotationOrbit: -30,
    minZoom: -5,
    maxZoom: 15,
  };
}

export function EmbeddingScatter({
  positions,
  colorValues,
  colorType,
  pointSize = 2,
  opacity = 0.8,
  onHover,
  onClick,
  onViewStateChange,
  selectedIndices,
  background = "white",
  maxRenderedCells = 100_000,
  dimensions = 2,
  minimal = false,
  dimToGray = false,
  externalViewState = null,
  restoreViewState = null,
  onRestoreViewState,
  cameraKey = "",
  onCameraChange,
  exportLegend = null,
  exportName = "embedding",
  colorLabel = "Value",
  categories = null,
  showLabels = false,
  highlightCategory = null,
}: EmbeddingScatterProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const deckRef = useRef<DeckGLRef>(null);
  const [viewState, setViewState] = useState<Record<string, unknown> | null>(
    null,
  );
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    index: number;
  } | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });

  // WebGL context loss recovery
  const [contextLost, setContextLost] = useState(false);
  const [renderKey, setRenderKey] = useState(0);

  // Track container size with ResizeObserver
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Proactively release the WebGL context when this scatter unmounts. Browsers
  // cap the number of live WebGL contexts (~16); navigating between deck.gl
  // panels (Overview, Trajectory, Unified View…) can accumulate contexts faster
  // than they're GC'd, and exceeding the cap makes the browser silently drop a
  // context — leaving the newest canvas blank with no 'contextlost' event.
  // Forcing loseContext() on unmount frees the GPU resource immediately.
  useEffect(() => {
    const el = containerRef.current;
    return () => {
      const canvas = el?.querySelector("canvas");
      const gl =
        (canvas?.getContext("webgl2") as WebGL2RenderingContext | null) ??
        (canvas?.getContext("webgl") as WebGLRenderingContext | null);
      gl?.getExtension("WEBGL_lose_context")?.loseContext();
    };
  }, []);

  // Attach WebGL context loss/restore listeners to the DeckGL canvas
  useEffect(() => {
    const canvas = containerRef.current?.querySelector("canvas");
    if (!canvas) return;

    const handleLost = (e: Event) => {
      e.preventDefault(); // Required to allow context restoration
      setContextLost(true);
    };
    const handleRestored = () => {
      setContextLost(false);
      setRenderKey((k) => k + 1); // Force DeckGL remount
    };

    canvas.addEventListener("webglcontextlost", handleLost);
    canvas.addEventListener("webglcontextrestored", handleRestored);
    return () => {
      canvas.removeEventListener("webglcontextlost", handleLost);
      canvas.removeEventListener("webglcontextrestored", handleRestored);
    };
  }, [renderKey, viewState]);

  const handleGpuError = useCallback((error: Error) => {
    console.error("[EmbeddingScatter] GPU error:", error);
    if (error.message?.toLowerCase().includes("context")) {
      setContextLost(true);
    }
  }, []);

  const retryRender = useCallback(() => {
    setContextLost(false);
    setRenderKey((k) => k + 1);
  }, []);

  const stride = dimensions === 3 ? 3 : 2;
  const totalCells = positions ? positions.length / stride : 0;

  const categoricalPalette = useSettingsStore((s) => s.categoricalPalette);
  const continuousColormap = useSettingsStore((s) => s.continuousColormap);

  // Precompute the RGBA color array for all cells
  const colorArray = useMemo<Uint8Array>(() => {
    const count = totalCells;
    const rgba = new Uint8Array(count * 4);

    if (!colorValues || colorValues.length === 0) {
      // Default: all cells the same color (steel blue)
      for (let i = 0; i < count; i++) {
        rgba[i * 4] = 70;
        rgba[i * 4 + 1] = 130;
        rgba[i * 4 + 2] = 180;
        rgba[i * 4 + 3] = Math.round(opacity * 255);
      }
      return rgba;
    }

    // Guard against race condition: when colorType updates to "categorical"
    // but colorValues still holds a Float32Array from a previous continuous fetch,
    // treat it as continuous to avoid fractional array index lookups.
    const effectiveColorType =
      colorType === "categorical" && colorValues instanceof Float32Array
        ? "continuous"
        : colorType;

    if (effectiveColorType === "categorical") {
      for (let i = 0; i < count; i++) {
        const catIdx = colorValues[i] ?? 0;
        const dimmed =
          selectedIndices && selectedIndices.size > 0 && !selectedIndices.has(i);
        if (dimmed && dimToGray) {
          // Faded grey — still visible as spatial context.
          rgba[i * 4] = 203;
          rgba[i * 4 + 1] = 213;
          rgba[i * 4 + 2] = 225;
          rgba[i * 4 + 3] = Math.round(opacity * 0.55 * 255);
        } else {
          const [r, g, b] = mapCategoryToColor(catIdx, categoricalPalette);
          rgba[i * 4] = r;
          rgba[i * 4 + 1] = g;
          rgba[i * 4 + 2] = b;
          rgba[i * 4 + 3] = dimmed
            ? Math.round(opacity * 0.15 * 255)
            : Math.round(opacity * 255);
        }
      }
    } else {
      // Continuous — find min/max
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < colorValues.length; i++) {
        const v = colorValues[i]!;
        if (v < min) min = v;
        if (v > max) max = v;
      }
      for (let i = 0; i < count; i++) {
        const v = colorValues[i] ?? 0;
        const [r, g, b] = mapValueToColor(v, min, max, continuousColormap);
        const dimmed =
          selectedIndices && selectedIndices.size > 0 && !selectedIndices.has(i);
        rgba[i * 4] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = dimmed
          ? Math.round(opacity * 0.15 * 255)
          : Math.round(opacity * 255);
      }
    }

    return rgba;
  }, [colorValues, colorType, opacity, selectedIndices, totalCells, dimToGray, categoricalPalette, continuousColormap]);

  // Downsampling: when cell count exceeds threshold, render a random subset
  const { renderPositions, renderColors, indexMap, isSampled } = useMemo(() => {
    if (!positions || totalCells === 0) {
      return { renderPositions: null, renderColors: null, indexMap: null, isSampled: false };
    }

    if (totalCells <= maxRenderedCells) {
      return { renderPositions: positions, renderColors: colorArray, indexMap: null, isSampled: false };
    }

    // Deterministic reservoir sample (seeded for stability)
    const sampleSize = maxRenderedCells;
    const indices = new Uint32Array(sampleSize);

    // Fill first sampleSize slots
    for (let i = 0; i < sampleSize; i++) {
      indices[i] = i;
    }

    // Seeded PRNG (mulberry32)
    let seed = totalCells; // Deterministic seed based on data size
    const rand = () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    // Reservoir sampling (Algorithm R)
    for (let i = sampleSize; i < totalCells; i++) {
      const j = Math.floor(rand() * (i + 1));
      if (j < sampleSize) {
        indices[j] = i;
      }
    }

    // Sort for better GPU cache coherence
    indices.sort();

    // Build compact sampled buffers
    const sampledPos = new Float32Array(sampleSize * stride);
    const sampledCol = new Uint8Array(sampleSize * 4);

    for (let s = 0; s < sampleSize; s++) {
      const orig = indices[s]!;
      for (let d = 0; d < stride; d++) {
        sampledPos[s * stride + d] = positions[orig * stride + d]!;
      }
      sampledCol[s * 4] = colorArray[orig * 4]!;
      sampledCol[s * 4 + 1] = colorArray[orig * 4 + 1]!;
      sampledCol[s * 4 + 2] = colorArray[orig * 4 + 2]!;
      sampledCol[s * 4 + 3] = colorArray[orig * 4 + 3]!;
    }

    return {
      renderPositions: sampledPos,
      renderColors: sampledCol,
      indexMap: indices,
      isSampled: true,
    };
  }, [positions, colorArray, totalCells, maxRenderedCells, stride]);

  const renderCount = renderPositions ? renderPositions.length / stride : 0;

  // Group labels where most of each group's drawn cells are; a group with no
  // drawn cell (sampled away) gets none.
  const centroids = useMemo(() => {
    if (!showLabels || !positions || !categories || !(colorValues instanceof Int32Array)) return null;
    if (colorType !== "categorical") return null;
    return groupCentroids(positions, stride, colorValues, categories, indexMap);
  }, [showLabels, positions, categories, colorValues, colorType, stride, indexMap]);

  // Compute initial view state from data bounds (use full positions for bounds)
  const initialViewState = useMemo(() => {
    if (!positions || positions.length === 0) return null;
    if (dimensions === 3) {
      return computeInitialViewState3D(positions, containerSize.width, containerSize.height);
    }
    return computeInitialViewState2D(
      positions,
      containerSize.width,
      containerSize.height,
    );
  }, [positions, containerSize.width, containerSize.height, dimensions]);

  // A restored camera (restoreViewState) stays until the user moves the camera;
  // `fitted` says whether the camera shown is the automatic fit.
  const restoredRef = useRef<Record<string, unknown> | null>(null);
  const fittedRef = useRef(true);

  // Reset view to initial
  const resetView = useCallback(() => {
    if (initialViewState) {
      restoredRef.current = null;
      fittedRef.current = true;
      setViewState({ ...initialViewState });
    }
  }, [initialViewState]);

  // When initial view state changes (new data), reset (to a restored camera if
  // there is one; the fit's zoom limits still apply)
  useEffect(() => {
    if (initialViewState) {
      fittedRef.current = !restoredRef.current;
      setViewState({ ...initialViewState, ...restoredRef.current });
    }
  }, [initialViewState]);

  // Another dataset or embedding: the restored camera meant the old one.
  useEffect(() => {
    restoredRef.current = null;
  }, [cameraKey]);

  const onRestoreRef = useRef(onRestoreViewState);
  onRestoreRef.current = onRestoreViewState;
  useEffect(() => {
    if (!restoreViewState) return;
    restoredRef.current = restoreViewState;
    if (!initialViewState) return; // applied with the first fit
    fittedRef.current = false;
    setViewState({ ...initialViewState, ...restoreViewState });
    onRestoreRef.current?.();
  }, [restoreViewState, initialViewState]);

  const onCameraRef = useRef(onCameraChange);
  onCameraRef.current = onCameraChange;
  useEffect(() => {
    if (viewState) onCameraRef.current?.(viewState, fittedRef.current);
  }, [viewState]);

  // Follow a linked view's camera (orientation + pan) when controlled, keeping
  // our own zoom so each view stays fitted to its container.
  useEffect(() => {
    if (!externalViewState) return;
    fittedRef.current = false;
    setViewState((prev) => {
      if (!prev) return { ...externalViewState };
      return {
        ...prev,
        ...externalViewState,
        zoom: prev.zoom,
        minZoom: prev.minZoom,
        maxZoom: prev.maxZoom,
      };
    });
  }, [externalViewState]);

  const handleViewStateChange = useCallback(
    ({ viewState: vs }: { viewState: Record<string, unknown> }) => {
      restoredRef.current = null;
      fittedRef.current = false;
      setViewState(vs);
      onViewStateChange?.(vs);
    },
    [onViewStateChange],
  );

  const handleHover = useCallback(
    (info: { index: number; x: number; y: number; picked: boolean }) => {
      if (info.picked && info.index >= 0) {
        // Remap sampled index back to original cell index
        const originalIndex = indexMap ? indexMap[info.index]! : info.index;
        setTooltip({ x: info.x, y: info.y, index: originalIndex });
        onHover?.({ index: originalIndex, x: info.x, y: info.y });
      } else {
        setTooltip(null);
        onHover?.(null);
      }
    },
    [onHover, indexMap],
  );

  const handleClick = useCallback(
    (info: { index: number; picked: boolean }) => {
      if (info.picked && info.index >= 0) {
        const originalIndex = indexMap ? indexMap[info.index]! : info.index;
        onClick?.(originalIndex);
      } else {
        onClick?.(null);
      }
    },
    [onClick, indexMap],
  );


  // Binary attribute data path — feeds typed arrays directly to GPU
  const pointLayers = useMemo(() => {
    if (renderCount === 0 || !renderPositions || !renderColors) return [];

    if (dimensions === 3) {
      return [
        new PointCloudLayer({
          id: "embedding-scatter-3d",
          data: {
            length: renderCount,
            attributes: {
              getPosition: { value: renderPositions, size: 3 },
              getColor: { value: renderColors, size: 4 },
            },
          },
          pointSize: pointSize * 2,
          pickable: true,
          autoHighlight: true,
          highlightColor: [255, 255, 0, 180],
          updateTriggers: {
            getColor: [renderColors],
            pointSize: [pointSize],
          },
        }),
      ];
    }

    return [
      new ScatterplotLayer({
        id: "embedding-scatter",
        data: {
          length: renderCount,
          attributes: {
            getPosition: { value: renderPositions, size: 2 },
            getFillColor: { value: renderColors, size: 4 },
          },
        },
        getRadius: pointSize,
        radiusUnits: "pixels" as const,
        radiusMinPixels: 0.5,
        radiusMaxPixels: 20,
        pickable: true,
        autoHighlight: true,
        highlightColor: [255, 255, 0, 180],
        updateTriggers: {
          getFillColor: [renderColors],
          getRadius: [pointSize],
        },
      }),
    ];
  }, [renderCount, renderPositions, renderColors, pointSize, dimensions]);
  const views = useMemo(
    () =>
      dimensions === 3
        ? new OrbitView({ id: "orbit", controller: true, orbitAxis: "Y" })
        : new OrthographicView({ id: "ortho", flipY: false, controller: true }),
    [dimensions],
  );

  // Labels are placed for the current camera: kept inside the plot, and off
  // each other (the bigger group keeps its spot).
  const labelLayers = useMemo(() => {
    if (!centroids || !viewState) return [];
    const view = labelViewOf(views, containerSize.width, containerSize.height, viewState);
    if (!view) return [];
    return centroidLabelLayers("embedding-labels", placeLabels(centroids, view, { highlight: highlightCategory }), {
      dark: background === "dark",
      highlight: highlightCategory,
    });
  }, [centroids, viewState, views, containerSize.width, containerSize.height, background, highlightCategory]);
  const layers = useMemo(() => [...pointLayers, ...labelLayers], [pointLayers, labelLayers]);

  // One-off high-resolution redraw on click; see downloadDeckPng.
  const downloadPng = useCallback(() => {
    const deck = deckRef.current?.deck;
    if (!deck) return;
    downloadDeckPng(deck, {
      legend: exportLegend,
      colormap: continuousColormap,
      filename: exportFilename(useDatasetStore.getState().currentDataset?.name, exportName),
      note: isSampled
        ? `Showing ${renderCount.toLocaleString()} of ${totalCells.toLocaleString()} cells (sampled)`
        : undefined,
    });
  }, [exportLegend, continuousColormap, exportName, isSampled, renderCount, totalCells]);

  if (!positions || positions.length === 0) {
    return (
      <div
        ref={containerRef}
        className="flex h-full w-full items-center justify-center rounded-lg border border-slate-200 bg-slate-50 text-sm text-slate-400"
      >
        No embedding data to display
      </div>
    );
  }

  const bgColor = background === "dark" ? "#1e293b" : "#ffffff";

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-lg">
      {viewState && !contextLost && (
        <DeckGL
          key={renderKey}
          ref={deckRef}
          views={views}
          viewState={{ ...viewState }}
          onViewStateChange={handleViewStateChange}
          layers={layers}
          onHover={handleHover}
          onClick={handleClick}
          onError={handleGpuError}
          controller={{ scrollZoom: { speed: 0.002, smooth: true } }}
          style={{ background: bgColor }}
          getCursor={({ isDragging }: { isDragging: boolean }) =>
            isDragging ? "grabbing" : "crosshair"
          }
        />
      )}

      {/* WebGL context lost overlay */}
      {contextLost && (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 rounded-lg bg-slate-900/80 text-white">
          <AlertTriangle className="h-8 w-8 text-amber-400" />
          <p className="text-sm font-medium">GPU context lost</p>
          <p className="max-w-xs text-center text-xs text-slate-300">
            The WebGL rendering context was lost, possibly due to GPU memory pressure.
          </p>
          <button
            onClick={retryRender}
            className="mt-1 rounded-md border border-slate-500 bg-slate-700 px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-slate-600"
          >
            Retry
          </button>
        </div>
      )}

      {/* Reset view button */}
      {!minimal && (
        <button
          onClick={resetView}
          title="Reset view"
          className="absolute right-3 top-3 z-10 rounded-md border border-slate-300 bg-white/90 p-1.5 text-slate-600 shadow-sm transition-colors hover:bg-slate-100 hover:text-slate-800"
        >
          <RotateCcw className="h-4 w-4" />
        </button>
      )}
      {!minimal && viewState && !contextLost && (
        <button
          onClick={downloadPng}
          title="Download PNG (high resolution, white background, with legend)"
          aria-label="Download PNG"
          className="absolute right-3 top-12 z-10 rounded-md border border-slate-300 bg-white/90 p-1.5 text-slate-600 shadow-sm transition-colors hover:bg-slate-100 hover:text-slate-800"
        >
          <ImageDown className="h-4 w-4" />
        </button>
      )}

      {/* Sampling indicator */}
      {!minimal && isSampled && (
        <div className="absolute left-3 top-3 z-10 rounded-md border border-slate-300 bg-white/90 px-2 py-1 text-[10px] text-slate-500">
          Showing {(renderCount / 1000).toFixed(0)}K of {(totalCells / 1000).toFixed(0)}K cells (sampled)
        </div>
      )}

      {/* Tooltip — suppressed when parent provides onHover (custom tooltip) */}
      {!minimal && tooltip && !onHover && (() => {
        const line = colorByLine({
          label: colorLabel,
          values: colorValues,
          index: tooltip.index,
          type: colorType,
          categories,
          colorOf: (code) => mapCategoryToColor(code, categoricalPalette).slice(0, 3) as [number, number, number],
        });
        return (
          <div
            className="pointer-events-none absolute z-20 max-w-xs rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs shadow-lg"
            style={{ left: tooltip.x + 12, top: tooltip.y - 12 }}
          >
            {line && <HoverLineRow line={line} />}
            <div className={line ? "text-[11px] text-slate-400" : "font-medium text-slate-700"}>
              Cell {tooltip.index.toLocaleString()}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
