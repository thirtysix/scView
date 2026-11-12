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
import { AlertTriangle, Loader2, RotateCcw, X } from "lucide-react";
import type { ScatterLegend } from "@/lib/scatterExport";
import {
  buildScatterSvg,
  canvasRasterizer,
  nextFrame,
  pointRadius,
  projectPoints,
  svgLabels,
  type PointMode,
  type SvgPanel,
} from "@/lib/scatterSvg";
import { saveFigure } from "@/lib/pdfExport";
import { ScatterExportMenu, type AllPanelsExport, type FigureFormat } from "@/components/plots/ScatterExportMenu";
import { EmbeddingCaveat } from "@/components/plots/EmbeddingCaveat";
import { GridPager } from "@/components/common/GridPager";
import {
  centroidLabelLayers,
  labelViewOf,
  placeLabels,
  type GroupCentroid,
  type LabelView,
  type PlacedLabel,
} from "@/lib/centroidLabels";
import {
  SPLIT_CONTEXT_LAYER,
  contextColor,
  dataBounds,
  downloadSplitPng,
  downloadSplitSheetPng,
  fitSplitViewState,
  layerInView,
  panelOfLayer,
  rescaleViewState,
  splitGridLayout,
  splitLabelLayerId,
  splitLayerId,
  splitSheetLayout,
  splitViewId,
  type SplitRect,
} from "@/lib/splitView";
import { clampPage, layoutCount, pageCount, pagePanelsText, pageRange } from "@/lib/gridPaging";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { LIGHT_SLATE, useScatterBackground } from "@/lib/theme";
import { GridLasso, type GridLassoResult } from "@/components/unified/GridLasso";

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
  /** `exportLegend` holds one legend per panel: a page's export carries only its panels' legends */
  legendPerPanel?: boolean;
  exportFilename: string;
  /** Content of the legend column (or, given the page's panels start..end - 1, the page's);
   *  empty when UnifiedViewPanel floats its own there */
  legend?: ReactNode | ((page: { start: number; end: number }) => ReactNode);
  /** The embedding's obsm key: a UMAP or t-SNE gets the cluster-distance caveat in the header */
  embedding?: string | null;
  /** Lasso on: a drag in a panel draws an outline there instead of moving the camera */
  lassoActive?: boolean;
  /** A lasso closed in one panel: the outline in that view's pixels and the camera it was drawn under */
  onLasso?: (lasso: GridLassoEvent) => void;
}

/** A lasso closed in one panel (see `onLasso`); `panel` indexes `panels`, whatever the page. */
export interface GridLassoEvent extends GridLassoResult {
  viewState: Record<string, unknown>;
}

/**
 * Small multiples drawn by one Deck with one view per panel, so it costs one
 * WebGL context however many panels there are (browsers cap live contexts at
 * ~16, and the app already holds several), and keeps every panel on one camera
 * simply by giving all views the same view state. Used by the split-by grid
 * (one panel per value of a column), the compare-colourings grid (the same
 * cells coloured by different columns) and the gene feature grid.
 *
 * More panels than a page holds are paged (GridPager, lib/gridPaging.ts): only
 * the current page's panels get views and layers; every page is laid out for a
 * full page, so the camera and panel size stay the same from page to page.
 * Hover and lasso report indices into `panels`, not the page. The download
 * menu saves the page, or every panel on one sheet laid out off screen.
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
  legendPerPanel = false,
  exportFilename,
  legend,
  embedding = null,
  lassoActive = false,
  onLasso,
}: MultiPanelScatterProps) {
  const pointSize = useSettingsStore((s) => s.pointSize);
  const opacity = useSettingsStore((s) => s.opacity);
  const background = useScatterBackground();
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const stride = dimensions === 3 ? 3 : 2;

  // --- Paging: the page's panels are panels[start..end) ---
  const perPage = useSplitViewStore((s) => s.panelsPerPage);
  const storedPage = useSplitViewStore((s) => s.page);
  const setPage = useSplitViewStore((s) => s.setPage);
  const setPanelsPerPage = useSplitViewStore((s) => s.setPanelsPerPage);
  const nPanels = panels.length;
  const page = clampPage(storedPage, nPanels, perPage);
  const { start, end } = pageRange(nPanels, perPage, page);
  const paged = pageCount(nPanels, perPage) > 1;
  const pagePanels = useMemo(() => panels.slice(start, end), [panels, start, end]);
  const slots = layoutCount(nPanels, perPage);

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
  // Laid out for a full page; the page's panels take the first cells (a short
  // last page leaves the rest empty, with no view, title or lasso).
  const fullGrid = useMemo(
    () => splitGridLayout(slots, area.width, area.height, aspect),
    [slots, area.width, area.height, aspect],
  );
  const grid = useMemo(
    () => ({ ...fullGrid, cells: fullGrid.cells.slice(0, pagePanels.length) }),
    [fullGrid, pagePanels.length],
  );
  const viewW = fullGrid.cells[0]?.view.width ?? 0;
  const viewH = fullGrid.cells[0]?.view.height ?? 0;

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
    // The context is drawn once per view: its cost grows with the panels a page shows.
    if (context) {
      out.push(point(SPLIT_CONTEXT_LAYER, context.length / stride, context, grey, false));
    }
    pagePanels.forEach((p, i) => {
      out.push(point(splitLayerId(i), p.indices.length, p.positions, p.colors, true));
    });
    return out;
  }, [ready, dimensions, context, stride, grey, pagePanels, pointSize]);
  // Labels are placed for the current camera (every panel has the same view
  // size), inside the panel and off each other; drawn last, over the points.
  const placedLabels = useMemo(() => {
    if (!ready || !viewState || !views[0]) return null;
    const view = labelViewOf(views[0], area.width, area.height, viewState);
    if (!view) return null;
    return pagePanels.map((p) => (p.labels?.length ? placeLabels(p.labels, view, { highlight: p.highlight }) : []));
  }, [ready, viewState, views, area.width, area.height, pagePanels]);
  const labelLayers = useMemo(
    () =>
      placedLabels
        ? pagePanels.flatMap((p, i) =>
            placedLabels[i]?.length
              ? centroidLabelLayers(splitLabelLayerId(i), placedLabels[i]!, {
                  dark: background === "dark",
                  highlight: p.highlight,
                })
              : [],
          )
        : [],
    [placedLabels, pagePanels, background],
  );
  const layers = useMemo(() => [...pointLayers, ...labelLayers], [pointLayers, labelLayers]);

  const layerFilter = useCallback(
    ({ layer, viewport }: { layer: { id: string }; viewport: { id: string } }) =>
      layerInView(layer.id, viewport.id),
    [],
  );

  // Hover reports the original cell index, with x/y moved from the canvas into
  // the scatter area so UnifiedViewPanel's tooltip lands next to the cursor,
  // and the panel's index in `panels` (the page's slot plus the page's start).
  const handleHover = useCallback(
    (info: PickingInfo) => {
      const p = panelOfLayer(info.layer?.id);
      const panel = p == null ? undefined : pagePanels[p];
      if (info.picked && panel && info.index >= 0 && info.index < panel.indices.length) {
        onHover?.({ index: panel.indices[info.index]!, x: info.x + area.left, y: info.y + area.top, panel: start + p! });
      } else {
        onHover?.(null);
      }
    },
    [pagePanels, start, onHover, area.left, area.top],
  );
  // Clear a tooltip left over from the single plot, or from this grid on exit.
  useEffect(() => {
    onHover?.(null);
    return () => onHover?.(null);
  }, [onHover]);

  // --- WebGL: release the context on unmount (opening a grid creates one) ---
  const deckRef = useRef<DeckGLRef>(null);
  const [gpuError, setGpuError] = useState(false);
  // While the download menu is open the grid sits above the legend floated over it.
  const [menuOpen, setMenuOpen] = useState(false);
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

  // --- Downloads: the page as on screen, or every panel on one sheet ---
  // One legend per panel (compare colourings): a page carries its own panels' legends.
  const pageLegend = useMemo(
    () => (legendPerPanel && Array.isArray(exportLegend) ? exportLegend.slice(start, end) : exportLegend),
    [legendPerPanel, exportLegend, start, end],
  );
  const pageFilename = paged ? `${exportFilename}_page${page + 1}` : exportFilename;
  const sheetFilename = `${exportFilename}_all`;

  const downloadPng = useCallback(() => {
    const deck = deckRef.current?.deck;
    if (!deck) return;
    downloadSplitPng(deck, {
      grid,
      panels: pagePanels.map((p) => ({ label: p.title, count: p.subtitle })),
      legend: pageLegend,
      colormap,
      filename: pageFilename,
      note: sampledNote,
    });
  }, [grid, pagePanels, pageLegend, colormap, pageFilename, sampledNote]);

  // Panels as figure panels (lib/scatterSvg.ts): each panel's points projected
  // with the shared camera, its title band and frame, the grey context on white.
  const svgPanelsOf = useCallback(
    (
      cells: readonly { title: SplitRect; view: SplitRect }[],
      list: readonly GridPanel[],
      view: LabelView,
      placed: readonly (readonly PlacedLabel[] | undefined)[] | null,
    ): SvgPanel[] => {
      const radius = pointRadius(pointSize, stride);
      const greyOnWhite = contextColor("white", opacity);
      // Every view has the same size and camera: the context projects once.
      const contextXy = context ? projectPoints(view.project, context, stride, new Uint8Array(0)).xy : null;
      return cells.flatMap((cell, i) => {
        const p = list[i];
        if (!p) return [];
        const fg = projectPoints(view.project, p.positions, stride, p.colors);
        const labels = placed?.[i];
        return [
          {
            view: cell.view,
            title: { box: cell.title, label: p.title, subtitle: p.subtitle },
            frame: true,
            sets: [
              ...(contextXy ? [{ xy: contextXy, colors: greyOnWhite, radius }] : []),
              { ...fg, radius },
            ],
            labels: labels?.length ? svgLabels(labels, view.project, p.highlight) : [],
          },
        ];
      });
    },
    [pointSize, stride, opacity, context],
  );

  type Figure = { width: number; height: number; panels: SvgPanel[] };
  // The page, laid out as on screen.
  const pageFigure = useCallback((): Figure | null => {
    if (!viewState || !views[0]) return null;
    const view = labelViewOf(views[0], area.width, area.height, viewState);
    if (!view) return null;
    return { width: area.width, height: area.height, panels: svgPanelsOf(grid.cells, pagePanels, view, placedLabels) };
  }, [viewState, views, area.width, area.height, svgPanelsOf, grid, pagePanels, placedLabels]);
  // Every panel on one sheet, off screen: laid out for all of them at about the
  // on-screen panel size, the camera rescaled to frame what each panel frames
  // now, labels placed again for that view. The live grid stays on its page.
  const sheetFigure = useCallback((): Figure | null => {
    if (!viewState || !(viewW > 0 && viewH > 0)) return null;
    const sheet = splitSheetLayout(nPanels, slots, area.width, area.height, aspect);
    const rect = sheet.grid.cells[0]?.view;
    if (!rect) return null;
    const sheetView =
      dimensions === 3
        ? new OrbitView({ id: "sheet", ...rect, orbitAxis: "Y" })
        : new OrthographicView({ id: "sheet", ...rect, flipY: false });
    const vs = rescaleViewState(viewState, bounds, dimensions, { width: viewW, height: viewH }, rect);
    const view = labelViewOf(sheetView, sheet.width, sheet.height, vs);
    if (!view) return null;
    const placed = panels.map((p) => (p.labels?.length ? placeLabels(p.labels, view, { highlight: p.highlight }) : []));
    return { width: sheet.width, height: sheet.height, panels: svgPanelsOf(sheet.grid.cells, panels, view, placed) };
  }, [viewState, viewW, viewH, nPanels, slots, area.width, area.height, aspect, dimensions, bounds, panels, svgPanelsOf]);

  const pointsOf = useCallback(
    (list: readonly GridPanel[]) =>
      (context ? (context.length / stride) * list.length : 0) + list.reduce((n, p) => n + p.indices.length, 0),
    [context, stride],
  );
  const figurePoints = useMemo(() => pointsOf(pagePanels), [pointsOf, pagePanels]);
  const sheetPoints = useMemo(() => pointsOf(panels), [pointsOf, panels]);

  const saveSvgFigure = useCallback(
    async (fig: Figure, legend: typeof exportLegend, format: FigureFormat, mode: PointMode, filename: string) => {
      const figure = buildScatterSvg({
        width: fig.width,
        height: fig.height,
        panels: fig.panels,
        legend,
        colormap,
        notes: sampledNote ? [sampledNote] : [],
        mode,
        rasterize: canvasRasterizer,
      });
      await saveFigure(format, figure, filename);
    },
    [colormap, sampledNote],
  );
  const downloadFigure = useCallback(
    async (format: FigureFormat, mode: PointMode) => {
      await nextFrame();
      const fig = pageFigure();
      if (fig) await saveSvgFigure(fig, pageLegend, format, mode, pageFilename);
    },
    [pageFigure, saveSvgFigure, pageLegend, pageFilename],
  );
  const downloadSheetFigure = useCallback(
    async (format: FigureFormat, mode: PointMode) => {
      await nextFrame();
      const fig = sheetFigure();
      if (fig) await saveSvgFigure(fig, exportLegend, format, mode, sheetFilename);
    },
    [sheetFigure, saveSvgFigure, exportLegend, sheetFilename],
  );
  const downloadSheetPng = useCallback(async () => {
    await nextFrame();
    const fig = sheetFigure();
    if (!fig) return;
    downloadSplitSheetPng({ ...fig, legend: exportLegend, colormap, filename: sheetFilename, note: sampledNote });
  }, [sheetFigure, exportLegend, colormap, sheetFilename, sampledNote]);
  const allPanels = useMemo<AllPanelsExport | null>(
    () =>
      paged
        ? { count: nPanels, nPoints: sheetPoints, onPng: downloadSheetPng, onFigure: downloadSheetFigure }
        : null,
    [paged, nPanels, sheetPoints, downloadSheetPng, downloadSheetFigure],
  );

  const handleLasso = useCallback(
    (lasso: GridLassoResult) => {
      if (viewState) onLasso?.({ ...lasso, panel: start + lasso.panel, viewState });
    },
    [viewState, onLasso, start],
  );

  const dark = background === "dark";
  const bgColor = dark ? "#1e293b" : "#ffffff";
  // Titles and frames sit on the plot background, which has its own fixed colour: slate classes
  // would follow the app theme instead (dark theme remaps slate-100 to a dark surface).
  const chrome = dark
    ? { title: LIGHT_SLATE[100], subtitle: LIGHT_SLATE[400], frame: LIGHT_SLATE[600] }
    : { title: LIGHT_SLATE[700], subtitle: LIGHT_SLATE[500], frame: LIGHT_SLATE[200] };

  return (
    <div
      className={`absolute inset-0 ${menuOpen ? "z-30" : "z-10"} flex flex-col overflow-hidden rounded-xl border border-slate-200`}
      style={{ background: bgColor }}
    >
      {/* Header: what is shown, extra controls, camera + export + close */}
      <div data-testid="grid-header" className="flex min-h-8 flex-shrink-0 flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-slate-200 bg-white px-3 py-0.5 text-[11px] text-slate-500">
        {/* What is shown keeps its room; the caveat gives way first (it truncates, its tip keeps the
            text), and a narrow header wraps to a second row instead of overlapping the pager */}
        <span className="max-w-[50%] flex-shrink-0 truncate">{header}</span>
        {sampledBadge && <span className="flex-shrink-0 text-[10px] text-slate-400">{sampledBadge}</span>}
        <div className="flex-1" />
        <EmbeddingCaveat embedding={embedding} className="min-w-0 max-w-full shrink overflow-hidden" />
        {headerControls}
        <GridPager
          n={nPanels}
          perPage={perPage}
          page={page}
          onPerPageChange={setPanelsPerPage}
          onPageChange={setPage}
        />
        <button
          onClick={resetView}
          title="Reset view"
          aria-label="Reset view"
          className="rounded p-1 text-slate-500 hover:bg-slate-100 hover:text-slate-700"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </button>
        <ScatterExportMenu
          nPoints={figurePoints}
          onPng={downloadPng}
          onFigure={downloadFigure}
          label="Download grid"
          disabled={!ready || !viewState || gpuError}
          onOpenChange={setMenuOpen}
          pageNote={paged ? `This page, ${pagePanelsText(nPanels, perPage, page).toLowerCase()}.` : undefined}
          allPanels={allPanels}
        />
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

          {ready && lassoActive && onLasso && viewState && !gpuError && (
            <GridLasso cells={grid.cells} onComplete={handleLasso} />
          )}

          {/* Titles + panel frames (the canvas underneath takes the events) */}
          {ready &&
            grid.cells.map((cell, i) => {
              const p = pagePanels[i];
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
                    <span className="truncate font-semibold" style={{ color: chrome.title }}>
                      {p.title}
                    </span>
                    <span className="flex-shrink-0" style={{ color: chrome.subtitle }}>{p.subtitle}</span>
                  </div>
                  <div
                    className="absolute rounded border"
                    style={{
                      borderColor: chrome.frame,
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
        <div className={`${LEGEND_COLUMN} flex-shrink-0 overflow-y-auto`}>
          {typeof legend === "function" ? legend({ start, end }) : legend}
        </div>
      </div>
    </div>
  );
}
