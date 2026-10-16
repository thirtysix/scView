/**
 * Split-by small multiples for the Unified View scatter: one panel per value of
 * a categorical obs column (stim / condition / sample), all drawn by one deck.gl
 * Deck with one view per panel (see SplitScatterGrid).
 *
 * The panels share one camera, one colour mapping and one continuous colour
 * scale, so a gene's expression reads the same in every panel. Everything here
 * is plain arrays in, typed arrays out, so it can be tested without WebGL.
 */
import type { ObsColumnInfo } from "@/api/types";
import {
  mapCategoryToColor,
  mapValueToColor,
  type CategoricalPaletteName,
  type ColormapName,
} from "@/lib/colors";
import {
  LEGEND_FONT,
  LEGEND_TITLE_FONT,
  captureScale,
  composeScatterPng,
  downloadCanvasPng,
  type CapturableDeck,
  type ScatterLegend,
} from "@/lib/scatterExport";

export const SPLIT_MIN_VALUES = 2;
export const SPLIT_MAX_VALUES = 6;
/** Gap between panels, and the title band above each one, in CSS pixels. */
export const SPLIT_GAP = 8;
export const SPLIT_TITLE_H = 22;

// ---------------------------------------------------------------------------
// Which columns can split the plot
// ---------------------------------------------------------------------------

export interface SplitColumnOption {
  name: string;
  nValues: number;
  /** Why the column can't be chosen, or null when it can. */
  disabledReason: string | null;
}

const isCategorical = (c: ObsColumnInfo) =>
  c.dtype === "category" || c.dtype === "object" || c.dtype === "bool";

/**
 * Categorical obs columns offered by the "Split by" control. Columns with 2–6
 * values can be chosen; those with more come back disabled with the reason.
 * Single-value columns, and those over the backend's 100-value cap (no values
 * listed), are left out.
 */
export function splitColumnOptions(columns: readonly ObsColumnInfo[]): SplitColumnOption[] {
  const out: SplitColumnOption[] = [];
  for (const c of columns) {
    if (!isCategorical(c) || !c.values) continue;
    const n = c.values.length;
    if (n < SPLIT_MIN_VALUES) continue;
    out.push({
      name: c.name,
      nValues: n,
      disabledReason:
        n > SPLIT_MAX_VALUES ? `${n} values; a split shows at most ${SPLIT_MAX_VALUES}` : null,
    });
  }
  return out.sort(
    (a, b) =>
      Number(a.disabledReason != null) - Number(b.disabledReason != null) ||
      a.name.localeCompare(b.name),
  );
}

// ---------------------------------------------------------------------------
// Cells per panel
// ---------------------------------------------------------------------------

/**
 * The cells EmbeddingScatter draws when a dataset is over `max` cells: the same
 * seeded reservoir sample, sorted, so the panels show the cells the single plot
 * showed. null when every cell is drawn.
 */
export function sampleCellIndices(total: number, max: number): Uint32Array | null {
  if (total <= max) return null;
  const indices = new Uint32Array(max);
  for (let i = 0; i < max; i++) indices[i] = i;
  let seed = total; // mulberry32, seeded like EmbeddingScatter's
  const rand = () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = max; i < total; i++) {
    const j = Math.floor(rand() * (i + 1));
    if (j < max) indices[j] = i;
  }
  indices.sort();
  return indices;
}

export interface SplitPanel {
  /** The value's code in the split column (its index in the column's values). */
  code: number;
  label: string;
  /** Cells with this value, drawn or not. */
  count: number;
  /** Original indices of the drawn cells with this value, ascending. */
  indices: Uint32Array;
}

/**
 * One panel per value of the split column that has cells, in the column's
 * order. `codes` holds each cell's value code (-1 = missing); `sample` limits
 * the drawn cells as above. `missing` counts cells with no value: they are in
 * no panel (only in the grey context).
 */
export function buildSplitPanels(
  codes: ArrayLike<number>,
  labels: readonly string[],
  sample: Uint32Array | null,
): { panels: SplitPanel[]; missing: number } {
  const k = labels.length;
  const counts = new Uint32Array(k);
  let missing = 0;
  for (let i = 0; i < codes.length; i++) {
    const c = codes[i]!;
    if (c >= 0 && c < k) counts[c] = counts[c]! + 1;
    else missing++;
  }
  const drawn = sample ? new Uint32Array(k) : counts;
  if (sample) {
    for (let s = 0; s < sample.length; s++) {
      const c = codes[sample[s]!]!;
      if (c >= 0 && c < k) drawn[c] = drawn[c]! + 1;
    }
  }
  const lists = Array.from({ length: k }, (_, c) => new Uint32Array(drawn[c]!));
  const fill = new Uint32Array(k);
  const n = sample ? sample.length : codes.length;
  for (let s = 0; s < n; s++) {
    const i = sample ? sample[s]! : s;
    const c = codes[i]!;
    if (c < 0 || c >= k) continue;
    lists[c]![fill[c]!] = i;
    fill[c] = fill[c]! + 1;
  }
  const panels: SplitPanel[] = [];
  for (let c = 0; c < k; c++) {
    if (counts[c]! > 0) panels.push({ code: c, label: labels[c]!, count: counts[c]!, indices: lists[c]! });
  }
  return { panels, missing };
}

/** The coordinates of `indices` out of interleaved `positions` (stride 2 or 3). */
export function gatherPositions(positions: Float32Array, indices: Uint32Array, stride: number): Float32Array {
  const out = new Float32Array(indices.length * stride);
  for (let s = 0; s < indices.length; s++) {
    const o = indices[s]! * stride;
    for (let d = 0; d < stride; d++) out[s * stride + d] = positions[o + d]!;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Colour: one mapping and one scale for every panel
// ---------------------------------------------------------------------------

/**
 * Categorical codes arrive as Int32Array; a Float32Array is a continuous
 * overlay even while colorType still says categorical (EmbeddingScatter's guard
 * for the same race).
 */
export function effectiveColorType(
  values: Float32Array | Int32Array | null,
  colorType: "categorical" | "continuous",
): "categorical" | "continuous" {
  return colorType === "categorical" && values instanceof Float32Array ? "continuous" : colorType;
}

/**
 * Min and max of `values` over every cell, skipping missing values: the one
 * colour scale all panels share, and the range the single plot and its legend
 * use, so the same expression has the same colour in every panel.
 */
export function valueRange(values: ArrayLike<number> | null): { min: number; max: number } {
  let min = Infinity;
  let max = -Infinity;
  if (values) {
    for (let i = 0; i < values.length; i++) {
      const v = values[i]!;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  return min <= max ? { min, max } : { min: 0, max: 0 };
}

export interface PanelColorOptions {
  colorValues: Float32Array | Int32Array | null;
  colorType: "categorical" | "continuous";
  /** The shared scale from valueRange (continuous colouring only). */
  range: { min: number; max: number };
  opacity: number;
  /** Cells outside a non-empty selection are faded, as in the single plot. */
  selected?: Set<number> | null;
  palette?: CategoricalPaletteName;
  colormap?: ColormapName;
}

/**
 * RGBA for the cells `indices`, coloured exactly as EmbeddingScatter colours
 * them in the single plot (category code -> palette slot, value -> colormap on
 * the shared range), so a panel is that plot's cells for one value.
 */
export function panelColors(indices: Uint32Array, opts: PanelColorOptions): Uint8Array {
  const { colorValues, range, opacity, selected, palette, colormap } = opts;
  const rgba = new Uint8Array(indices.length * 4);
  const full = Math.round(opacity * 255);
  const faded = Math.round(opacity * 0.15 * 255);
  const anySelected = !!selected && selected.size > 0;
  const type = effectiveColorType(colorValues, opts.colorType);
  for (let s = 0; s < indices.length; s++) {
    const i = indices[s]!;
    let r = 70, g = 130, b = 180; // steel blue: nothing to colour by
    if (colorValues && colorValues.length > 0) {
      [r, g, b] =
        type === "categorical"
          ? mapCategoryToColor(colorValues[i] ?? 0, palette)
          : mapValueToColor(colorValues[i] ?? 0, range.min, range.max, colormap);
    }
    rgba[s * 4] = r;
    rgba[s * 4 + 1] = g;
    rgba[s * 4 + 2] = b;
    rgba[s * 4 + 3] = anySelected && !selected!.has(i) ? faded : full;
  }
  return rgba;
}

/** Faint grey for the other cells drawn behind each panel's own. */
export function contextColor(
  background: "white" | "dark",
  opacity: number,
): [number, number, number, number] {
  return background === "dark"
    ? [100, 116, 139, Math.round(opacity * 0.35 * 255)]
    : [203, 213, 225, Math.round(opacity * 0.4 * 255)];
}

// ---------------------------------------------------------------------------
// Layout and camera
// ---------------------------------------------------------------------------

export interface SplitRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SplitGrid {
  cols: number;
  rows: number;
  /** Per panel: the title band and, below it, the plot's view. */
  cells: { title: SplitRect; view: SplitRect }[];
}

/**
 * Arrange `n` equal panels in a `width` x `height` area (CSS pixels), choosing
 * the column count that draws data of the given aspect ratio (width / height)
 * largest. Panels fill rows left to right; every view is the same size, so one
 * camera fits them all.
 */
export function splitGridLayout(
  n: number,
  width: number,
  height: number,
  aspect = 1,
  gap = SPLIT_GAP,
  titleHeight = SPLIT_TITLE_H,
): SplitGrid {
  if (n <= 0 || width <= 0 || height <= 0) return { cols: 0, rows: 0, cells: [] };
  const a = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  let cols = 1;
  let best = -Infinity;
  for (let c = 1; c <= n; c++) {
    const r = Math.ceil(n / c);
    const w = (width - gap * (c - 1)) / c;
    const h = (height - gap * (r - 1)) / r - titleHeight;
    const scale = Math.min(w / a, h);
    if (scale > best) {
      best = scale;
      cols = c;
    }
  }
  const rows = Math.ceil(n / cols);
  const cellW = Math.max(1, Math.floor((width - gap * (cols - 1)) / cols));
  const cellH = Math.max(titleHeight + 1, Math.floor((height - gap * (rows - 1)) / rows));
  const cells = Array.from({ length: n }, (_, i) => {
    const x = (i % cols) * (cellW + gap);
    const y = Math.floor(i / cols) * (cellH + gap);
    return {
      title: { x, y, width: cellW, height: titleHeight },
      view: { x, y: y + titleHeight, width: cellW, height: cellH - titleHeight },
    };
  });
  return { cols, rows, cells };
}

export interface DataBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

/** Bounds of every cell (the panels all frame the whole embedding). */
export function dataBounds(positions: Float32Array, stride: number): DataBounds {
  const b = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: 0, maxZ: 0 };
  if (stride === 3) {
    b.minZ = Infinity;
    b.maxZ = -Infinity;
  }
  for (let i = 0; i < positions.length; i += stride) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    if (x < b.minX) b.minX = x;
    if (x > b.maxX) b.maxX = x;
    if (y < b.minY) b.minY = y;
    if (y > b.maxY) b.maxY = y;
    if (stride === 3) {
      const z = positions[i + 2]!;
      if (z < b.minZ) b.minZ = z;
      if (z > b.maxZ) b.maxZ = z;
    }
  }
  return b;
}

/**
 * The shared camera that fits the whole embedding into one panel's view, as
 * EmbeddingScatter fits it into its container (5% padding in 2D; the same
 * starting orbit in 3D).
 */
export function fitSplitViewState(
  b: DataBounds,
  dimensions: 2 | 3,
  width: number,
  height: number,
): Record<string, unknown> {
  const dataW = b.maxX - b.minX || 1;
  const dataH = b.maxY - b.minY || 1;
  if (dimensions === 3) {
    const dataD = b.maxZ - b.minZ || 1;
    return {
      target: [b.minX + dataW / 2, b.minY + dataH / 2, b.minZ + dataD / 2],
      zoom: Math.log2((Math.min(width, height) * 0.85) / Math.max(dataW, dataH, dataD)),
      rotationX: 30,
      rotationOrbit: -30,
      minZoom: -5,
      maxZoom: 15,
    };
  }
  const zoom = Math.log2(Math.min((width || 800) / (dataW * 1.1), (height || 600) / (dataH * 1.1)));
  return {
    target: [b.minX + dataW / 2, b.minY + dataH / 2, 0],
    zoom,
    minZoom: zoom - 4,
    maxZoom: zoom + 10,
  };
}

// ---------------------------------------------------------------------------
// deck.gl ids: view `split-<i>` draws the context layer and layer `split-fg-<i>`
// ---------------------------------------------------------------------------

export const SPLIT_CONTEXT_LAYER = "split-context";
export const splitViewId = (i: number) => `split-${i}`;
export const splitLayerId = (i: number) => `split-fg-${i}`;

/** The panel a foreground layer belongs to, or null for any other layer. */
export function panelOfLayer(layerId: string | undefined): number | null {
  const m = layerId?.match(/^split-fg-(\d+)$/);
  return m ? Number(m[1]) : null;
}

/** deck.gl layerFilter: the grey context in every view, each panel's cells in its own. */
export function layerInView(layerId: string, viewId: string): boolean {
  if (layerId === SPLIT_CONTEXT_LAYER) return true;
  const p = panelOfLayer(layerId);
  return p != null && viewId === splitViewId(p);
}

// ---------------------------------------------------------------------------
// PNG export
// ---------------------------------------------------------------------------

/** Value and cell count shown above a panel. */
export function panelTitle(p: Pick<SplitPanel, "label" | "count">): { label: string; count: string } {
  return { label: p.label, count: `${p.count.toLocaleString()} cells` };
}

/**
 * Save the split view as one PNG: every panel re-rendered at a higher pixel
 * ratio (one Deck draws them all), the titles drawn into their bands, then the
 * shared legend beside the grid, as the single plot's "Download PNG" does.
 */
export function downloadSplitPng(
  deck: CapturableDeck,
  opts: {
    grid: SplitGrid;
    panels: Pick<SplitPanel, "label" | "count">[];
    legend: ScatterLegend | null;
    colormap: ColormapName;
    filename: string;
    note?: string;
  },
): void {
  const canvas = deck.getCanvas();
  if (!canvas) return;
  const scale = captureScale(canvas.clientWidth, canvas.clientHeight);
  let image: HTMLCanvasElement;
  try {
    deck.setProps({ useDevicePixels: scale });
    deck.redraw("png export");
    const annotated = document.createElement("canvas");
    annotated.width = canvas.width;
    annotated.height = canvas.height;
    const ctx = annotated.getContext("2d");
    if (!ctx) throw new Error("2D canvas not available");
    ctx.drawImage(canvas, 0, 0);
    ctx.scale(scale, scale);
    ctx.textBaseline = "middle";
    opts.grid.cells.forEach((cell, i) => {
      const p = opts.panels[i];
      if (!p) return;
      const { label, count } = panelTitle(p);
      const y = cell.title.y + cell.title.height / 2;
      ctx.font = LEGEND_TITLE_FONT;
      ctx.fillStyle = "#334155";
      ctx.fillText(label, cell.title.x + 4, y);
      const w = ctx.measureText(label).width;
      ctx.font = LEGEND_FONT;
      ctx.fillStyle = "#64748b";
      ctx.fillText(count, cell.title.x + 4 + w + 8, y);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.strokeRect(cell.view.x + 0.5, cell.view.y + 0.5, cell.view.width - 1, cell.view.height - 1);
    });
    image = composeScatterPng(annotated, opts.legend, scale, opts.colormap, opts.note);
  } finally {
    deck.setProps({ useDevicePixels: true });
    deck.redraw("png export done");
  }
  downloadCanvasPng(image, opts.filename);
}
