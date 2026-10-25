/**
 * PNG export for the deck.gl embedding scatter: the WebGL canvas, re-rendered
 * at a higher pixel ratio, composited onto white with the colour legend drawn
 * beside it.
 *
 * The capture is a one-off synchronous redraw (see EmbeddingScatter), so normal
 * rendering keeps preserveDrawingBuffer off. Layout is split from drawing so it
 * can be tested without a 2D canvas (jsdom has none).
 */
import { colormapColor, type AnyColormapName } from "@/lib/colors";

export type ScatterLegend =
  | {
      type: "categorical";
      title?: string;
      categories: string[];
      colors: [number, number, number][];
    }
  // colormap: the one this legend was drawn with (a diverging map for signed
  // scores), over the sequential default; centre: its labelled midpoint (0).
  // caption: what the range means (colorScaleCaption), printed under SVG/PDF figures.
  | {
      type: "continuous";
      title?: string;
      min: number;
      max: number;
      colormap?: AnyColormapName;
      centre?: number | null;
      caption?: string;
    };

/** Render the capture at this multiple of the on-screen size... */
export const SCATTER_PNG_SCALE = 3;
/** ...unless that would exceed this many pixels on a side (GPU texture limits). */
export const MAX_CAPTURE_DIM = 8192;

/** Pixel ratio for a capture of a `width` x `height` CSS-pixel canvas. */
export function captureScale(
  width: number,
  height: number,
  wanted = SCATTER_PNG_SCALE,
  maxDim = MAX_CAPTURE_DIM,
): number {
  const longest = Math.max(width, height, 1);
  return Math.max(1, Math.min(wanted, Math.floor((maxDim / longest) * 100) / 100));
}

// Legend metrics, in CSS pixels (multiplied by the capture scale when drawn).
const PAD = 16;
const TITLE_H = 20;
const ROW_H = 18;
const SWATCH = 10;
const GAP = 6;
const COL_GAP = 14;
const BAR_W = 14;
const BAR_MAX_H = 240;
const MAX_LABEL = 32;
const MAX_ENTRIES = 120;
/** The legend metrics, for the SVG export (lib/scatterSvg.ts) to draw the same legend. */
export const LEGEND_METRICS = { PAD, TITLE_H, ROW_H, SWATCH, GAP, BAR_W } as const;
export const LEGEND_FONT = "12px Inter, system-ui, sans-serif";
export const LEGEND_TITLE_FONT = "600 13px Inter, system-ui, sans-serif";

export type Measure = (text: string, font: string) => number;

export interface LegendLayout {
  width: number;
  height: number;
  title: string;
  /** Categorical: one entry per drawn row, labels truncated. A null colour is
   *  the "+N more" line for categories left out. */
  entries: { label: string; color: [number, number, number] | null; x: number; y: number }[];
  /** Continuous: the colour bar's box, its end labels, a midpoint label (a
   *  diverging scale's 0) and the map it is drawn with (null: the default) */
  bar: {
    x: number;
    y: number;
    h: number;
    maxLabel: string;
    minLabel: string;
    midLabel: string | null;
    colormap: AnyColormapName | null;
  } | null;
}

function truncate(label: string): string {
  return label.length > MAX_LABEL ? `${label.slice(0, MAX_LABEL - 1)}…` : label;
}

function fmtValue(v: number): string {
  if (!Number.isFinite(v)) return "";
  return Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : v.toFixed(2);
}

/**
 * Place the legend for a plot `plotHeight` CSS pixels tall: categories fill a
 * column top to bottom, then wrap into the next one.
 */
export function legendLayout(legend: ScatterLegend, plotHeight: number, measure: Measure): LegendLayout {
  const title = legend.title ?? "";
  const top = PAD + (title ? TITLE_H : 0);
  const titleW = title ? measure(title, LEGEND_TITLE_FONT) : 0;

  if (legend.type === "continuous") {
    const h = Math.max(60, Math.min(BAR_MAX_H, plotHeight - top - PAD));
    const maxLabel = fmtValue(legend.max);
    const minLabel = fmtValue(legend.min);
    const midLabel = legend.centre != null ? fmtValue(legend.centre) : null;
    const labelW = Math.max(measure(maxLabel, LEGEND_FONT), measure(minLabel, LEGEND_FONT), midLabel ? measure(midLabel, LEGEND_FONT) : 0);
    return {
      width: PAD * 2 + Math.max(titleW, BAR_W + GAP + labelW),
      height: top + h + PAD,
      title,
      entries: [],
      bar: { x: PAD, y: top, h, maxLabel, minLabel, midLabel, colormap: legend.colormap ?? null },
    };
  }

  const items: { label: string; color: [number, number, number] | null }[] = legend.categories
    .slice(0, MAX_ENTRIES)
    .map((c, i) => ({ label: truncate(c), color: legend.colors[i] ?? [127, 127, 127] }));
  const hidden = legend.categories.length - items.length;
  if (hidden > 0) items.push({ label: `+${hidden} more`, color: null });
  const perCol = Math.max(1, Math.floor((plotHeight - top - PAD) / ROW_H));
  const nCols = Math.max(1, Math.ceil(items.length / perCol));

  const entries: LegendLayout["entries"] = [];
  let x = PAD;
  for (let c = 0; c < nCols; c++) {
    const col = items.slice(c * perCol, (c + 1) * perCol);
    col.forEach((item, r) => entries.push({ ...item, x, y: top + r * ROW_H }));
    const textW = Math.max(0, ...col.map((item) => measure(item.label, LEGEND_FONT)));
    x += SWATCH + GAP + textW + COL_GAP;
  }
  return {
    width: Math.max(x - COL_GAP + PAD, PAD * 2 + titleW),
    height: top + Math.min(items.length, perCol) * ROW_H + PAD,
    title,
    entries,
    bar: null,
  };
}

/** One legend, or several side by side (one per panel's column), or none. */
export type ScatterLegends = ScatterLegend | ScatterLegend[] | null;

/**
 * Place each legend for a plot `plotHeight` CSS pixels tall, left to right:
 * their x offsets from the plot's right edge, and the size of the whole strip.
 */
export function legendsLayout(
  legends: ScatterLegends,
  plotHeight: number,
  measure: Measure,
): { layouts: (LegendLayout & { x: number })[]; width: number; height: number } {
  const list = legends == null ? [] : Array.isArray(legends) ? legends : [legends];
  let x = 0;
  const layouts = list.map((l) => {
    const layout = { ...legendLayout(l, plotHeight, measure), x };
    x += layout.width;
    return layout;
  });
  return { layouts, width: x, height: Math.max(0, ...layouts.map((l) => l.height)) };
}

/**
 * Draw `source` (the scatter, already rendered at `scale`) on white, with the
 * legend (or legends) to its right and an optional note (e.g. "sampled") under
 * the plot.
 */
export function composeScatterPng(
  source: HTMLCanvasElement,
  legend: ScatterLegends,
  scale: number,
  colormap: AnyColormapName,
  note?: string,
): HTMLCanvasElement {
  const out = document.createElement("canvas");
  const probe = out.getContext("2d");
  if (!probe) throw new Error("2D canvas not available");
  const measure: Measure = (text, font) => {
    probe.font = font;
    return probe.measureText(text).width;
  };
  const plotH = source.height / scale;
  const strip = legendsLayout(legend, plotH, measure);
  out.width = Math.ceil(source.width + strip.width * scale);
  out.height = Math.ceil(Math.max(source.height, strip.height * scale));

  const ctx = out.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(source, 0, 0);

  ctx.scale(scale, scale);
  ctx.textBaseline = "middle";
  if (note) {
    ctx.font = LEGEND_FONT;
    ctx.fillStyle = "#64748b";
    ctx.fillText(note, PAD, plotH - PAD);
  }
  for (const layout of strip.layouts) {
    const dx = source.width / scale + layout.x;
    ctx.translate(dx, 0);
    drawLegend(ctx, layout, colormap);
    ctx.translate(-dx, 0);
  }
  return out;
}

function drawLegend(ctx: CanvasRenderingContext2D, layout: LegendLayout, colormap: AnyColormapName): void {
  if (layout.title) {
    ctx.font = LEGEND_TITLE_FONT;
    ctx.fillStyle = "#334155";
    ctx.fillText(layout.title, PAD, PAD + TITLE_H / 2 - 2);
  }
  ctx.font = LEGEND_FONT;
  for (const e of layout.entries) {
    if (e.color) {
      ctx.fillStyle = `rgb(${e.color[0]},${e.color[1]},${e.color[2]})`;
      ctx.beginPath();
      ctx.arc(e.x + SWATCH / 2, e.y + ROW_H / 2, SWATCH / 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = e.color ? "#334155" : "#64748b";
    ctx.fillText(e.label, e.x + SWATCH + GAP, e.y + ROW_H / 2);
  }
  if (layout.bar) {
    const { x, y, h, maxLabel, minLabel, midLabel } = layout.bar;
    const map = layout.bar.colormap ?? colormap;
    const grad = ctx.createLinearGradient(0, y + h, 0, y);
    for (let i = 0; i <= 32; i++) {
      const [r, g, b] = colormapColor(map, i / 32);
      grad.addColorStop(i / 32, `rgb(${r},${g},${b})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, BAR_W, h);
    ctx.fillStyle = "#334155";
    ctx.fillText(maxLabel, x + BAR_W + GAP, y + 6);
    if (midLabel !== null) ctx.fillText(midLabel, x + BAR_W + GAP, y + h / 2);
    ctx.fillText(minLabel, x + BAR_W + GAP, y + h - 6);
  }
}

/** The part of a deck.gl Deck the capture needs. */
export interface CapturableDeck {
  getCanvas(): HTMLCanvasElement | null;
  setProps(props: { useDevicePixels: boolean | number }): void;
  redraw(reason?: string): void;
}

/**
 * Re-render `deck` once at a higher pixel ratio, copy it onto a white canvas with
 * the legend, then put the live view back as it was.
 *
 * Without preserveDrawingBuffer the WebGL canvas is cleared once the browser
 * composites it, so the copy must happen in the same task as the redraw; it
 * does, because redraw() renders synchronously. Normal frames are untouched.
 */
export function downloadDeckPng(
  deck: CapturableDeck,
  opts: {
    legend: ScatterLegends;
    colormap: AnyColormapName;
    filename: string;
    note?: string;
    wantedScale?: number;
  },
): void {
  const canvas = deck.getCanvas();
  if (!canvas) return;
  const scale = captureScale(canvas.clientWidth, canvas.clientHeight, opts.wantedScale);
  let image: HTMLCanvasElement;
  try {
    deck.setProps({ useDevicePixels: scale });
    deck.redraw("png export");
    image = composeScatterPng(canvas, opts.legend, scale, opts.colormap, opts.note);
  } finally {
    deck.setProps({ useDevicePixels: true });
    deck.redraw("png export done"); // resizing cleared the canvas
  }
  downloadCanvasPng(image, opts.filename);
}

/** Save a canvas as a PNG download. */
export function downloadCanvasPng(canvas: HTMLCanvasElement, filename: string): void {
  canvas.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.endsWith(".png") ? filename : `${filename}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }, "image/png");
}
