/**
 * SVG (and, through lib/pdfExport.ts, PDF) export of the deck.gl embedding
 * scatter and of the split / compare grids: the points as vector circles in
 * the order, colours, size and opacity drawn on screen, the group labels, axis
 * names, a title and the same legend the PNG gets (lib/scatterExport.ts).
 *
 * Above HYBRID_THRESHOLD points a vector file gets large and slow to open, so
 * the points can instead be one embedded high-resolution PNG with every piece
 * of text (title, labels, legend, axes) still vector: what journals accept for
 * dense scatter plots, and the default there.
 *
 * Exports are always on white, whatever the screen's background. Building the
 * SVG is pure string work on screen-space points (projectPoints does the camera),
 * so it is tested without a browser; only rasterising needs a canvas.
 */
import { colormapColor, type AnyColormapName } from "@/lib/colors";
import {
  LEGEND_METRICS,
  LEGEND_TITLE_FONT,
  legendsLayout,
  type Measure,
  type ScatterLegends,
} from "@/lib/scatterExport";
import { prettyEmbeddingLabel } from "@/lib/formatting";
import { LABEL_SIZE } from "@/lib/centroidLabels";

/** Above this many drawn points the export defaults to points as an image. */
export const HYBRID_THRESHOLD = 50_000;
/** The embedded points image is drawn at this multiple of the on-screen size. */
export const RASTER_SCALE = 3;

/** Points as vector circles, or as one embedded image (text stays vector). */
export type PointMode = "vector" | "raster";

export function defaultPointMode(nPoints: number): PointMode {
  return nPoints > HYBRID_THRESHOLD ? "raster" : "vector";
}

export interface SvgRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Points in a view's pixels (from its top-left, y down), in draw order. */
export interface SvgPointSet {
  /** x, y per point */
  xy: Float32Array;
  /** RGBA per point, or one colour for all */
  colors: Uint8Array | readonly [number, number, number, number];
  /** Circle radius, px */
  radius: number;
}

/** A group label at its placed position (view pixels); `faded` when another group is highlighted. */
export interface SvgLabel {
  text: string;
  x: number;
  y: number;
  faded: boolean;
}

export interface SvgPanel {
  /** The plot box, in the plot area's pixels */
  view: SvgRect;
  /** Point layers drawn in order (the grey context first) */
  sets: SvgPointSet[];
  labels?: SvgLabel[];
  /** Title band above the view (grids) */
  title?: { box: SvgRect; label: string; subtitle?: string };
  /** Thin frame around the view (grids) */
  frame?: boolean;
}

/** Draws a panel's point sets into an image of its view, returning a data URL. */
export type Rasterizer = (sets: SvgPointSet[], width: number, height: number, scale: number) => string;

export interface ScatterSvgOptions {
  /** Size of the plot area holding every panel, px */
  width: number;
  height: number;
  panels: SvgPanel[];
  legend: ScatterLegends;
  colormap: AnyColormapName;
  title?: string;
  /** Axis names, x then y (single plots in 2D) */
  axes?: readonly [string, string] | null;
  /** Lines printed under the figure (e.g. "sampled"); legend captions come first */
  notes?: string[];
  mode: PointMode;
  rasterize?: Rasterizer;
  measure?: Measure;
}

export interface ScatterSvg {
  svg: string;
  width: number;
  height: number;
  /** Circles written as vector shapes (0 in raster mode) */
  circles: number;
}

const FONT_FAMILY = "Inter, Helvetica, Arial, sans-serif";
const TEXT = "#334155";
const MUTED = "#64748b";
const FRAME = "#e2e8f0";
const MARGIN = 12;
const TITLE_BAND = 26;
const AXIS_BAND = 18;
const LEGEND_GAP = 8;
const NOTE_LINE = 15;
const NOTE_SIZE = 11;

/** Total points an export would draw (every set of every panel). */
export function countPoints(panels: readonly SvgPanel[]): number {
  let n = 0;
  for (const p of panels) for (const s of p.sets) n += s.xy.length / 2;
  return n;
}

/**
 * "UMAP1"/"UMAP2", "tSNE1"/"tSNE2", "PC3"/"PC4" (a chosen PCA pair). A pair
 * the embedding does not have (`nDims` known) falls back to PC1/PC2, as the
 * plot itself does (useEmbedding's embeddingDims).
 */
export function embeddingAxisLabels(
  embedding: string | null | undefined,
  dims?: readonly [number, number],
  nDims?: number,
): [string, string] {
  if (embedding === "spatial") return ["x (pixels)", "y (pixels)"];
  const key = (embedding ?? "").replace(/^X_/, "").toLowerCase();
  if (key.startsWith("pca")) {
    const ok = dims && dims[0] !== dims[1] && (nDims === undefined || Math.max(...dims) < nDims);
    const [a, b] = ok ? dims : [0, 1];
    return [`PC${a + 1}`, `PC${b + 1}`];
  }
  const base = key.startsWith("tsne") ? "tSNE" : prettyEmbeddingLabel(embedding || "embedding");
  return [`${base}1`, `${base}2`];
}

export function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

const num = (v: number) => String(Math.round(v * 10) / 10);
const hex = (r: number, g: number, b: number) =>
  `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
const alphaText = (a: number) => String(Math.round((a / 255) * 1000) / 1000);

let measureCtx: OffscreenCanvasRenderingContext2D | null | undefined;

/** Text width with a canvas where there is one, else an estimate from the font size. */
export const measureText: Measure = (text, font) => {
  if (measureCtx === undefined) {
    try {
      measureCtx = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1).getContext("2d") : null;
    } catch {
      measureCtx = null;
    }
  }
  if (measureCtx) {
    measureCtx.font = font;
    return measureCtx.measureText(text).width;
  }
  const size = Number(/(\d+(?:\.\d+)?)px/.exec(font)?.[1] ?? 12);
  return text.length * size * 0.56;
};

/** Split `text` into lines no wider than `width`. */
export function wrapText(text: string, width: number, font: string, measure: Measure): string[] {
  const lines: string[] = [];
  let line = "";
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const next = line ? `${line} ${word}` : word;
    if (line && measure(next, font) > width) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

/** The commonest alpha of an RGBA array (it goes on the group, not each circle). */
function commonAlpha(colors: Uint8Array): number {
  const counts = new Uint32Array(256);
  for (let i = 3; i < colors.length; i += 4) counts[colors[i]!]!++;
  let best = 255;
  for (let a = 0; a < 256; a++) if (counts[a]! > counts[best]!) best = a;
  return best;
}

/** Circles of one point set, skipping points outside the view (and fully transparent ones). */
function circlesSvg(set: SvgPointSet, view: SvgRect, out: string[]): number {
  const n = set.xy.length / 2;
  const r = set.radius;
  const rText = num(r);
  const inside = (x: number, y: number) => x >= -r && x <= view.width + r && y >= -r && y <= view.height + r;
  let written = 0;
  if (!(set.colors instanceof Uint8Array)) {
    const [cr, cg, cb, ca] = set.colors;
    if (ca === 0) return 0;
    out.push(`<g fill="${hex(cr, cg, cb)}" fill-opacity="${alphaText(ca)}">`);
    for (let i = 0; i < n; i++) {
      const x = set.xy[i * 2]!;
      const y = set.xy[i * 2 + 1]!;
      if (!inside(x, y)) continue;
      out.push(`<circle cx="${num(x)}" cy="${num(y)}" r="${rText}"/>`);
      written++;
    }
    out.push("</g>");
    return written;
  }
  const colors = set.colors;
  const alpha = commonAlpha(colors);
  out.push(`<g fill-opacity="${alphaText(alpha)}">`);
  for (let i = 0; i < n; i++) {
    const a = colors[i * 4 + 3]!;
    if (a === 0) continue;
    const x = set.xy[i * 2]!;
    const y = set.xy[i * 2 + 1]!;
    if (!inside(x, y)) continue;
    const fill = hex(colors[i * 4]!, colors[i * 4 + 1]!, colors[i * 4 + 2]!);
    const op = a === alpha ? "" : ` fill-opacity="${alphaText(a)}"`;
    out.push(`<circle cx="${num(x)}" cy="${num(y)}" r="${rText}" fill="${fill}"${op}/>`);
    written++;
  }
  out.push("</g>");
  return written;
}

function textSvg(
  x: number,
  y: number,
  text: string,
  opts: { size?: number; weight?: number; fill?: string; anchor?: "start" | "middle" | "end"; rotate?: boolean } = {},
): string {
  const { size = 12, weight = 400, fill = TEXT, anchor = "start", rotate = false } = opts;
  const transform = rotate ? ` transform="rotate(-90 ${num(x)} ${num(y)})"` : "";
  return (
    `<text x="${num(x)}" y="${num(y)}" dy="0.35em" font-size="${size}"${weight >= 600 ? ' font-weight="bold"' : ""} fill="${fill}"` +
    ` text-anchor="${anchor}"${transform}>${escapeXml(text)}</text>`
  );
}

function labelsSvg(labels: readonly SvgLabel[], out: string[]): void {
  const draw = (l: SvgLabel) => {
    const at = `x="${num(l.x)}" y="${num(l.y)}" dy="0.35em" text-anchor="middle" font-size="${LABEL_SIZE}" font-weight="bold"`;
    const text = escapeXml(l.text.length > 32 ? `${l.text.slice(0, 31)}…` : l.text);
    // The halo first (a white stroke), then the text over it: PDF converters
    // do not all honour paint-order.
    out.push(`<text ${at} fill="#ffffff" stroke="#ffffff" stroke-width="4" stroke-linejoin="round">${text}</text>`);
    out.push(`<text ${at} fill="#0f172a">${text}</text>`);
  };
  const faded = labels.filter((l) => l.faded);
  if (faded.length) {
    out.push(`<g opacity="0.3" class="labels-faded">`);
    faded.forEach(draw);
    out.push("</g>");
  }
  labels.filter((l) => !l.faded).forEach(draw);
}

/** Build the SVG of one plot or a grid of panels, with the legend to the right. */
export function buildScatterSvg(o: ScatterSvgOptions): ScatterSvg {
  const measure = o.measure ?? measureText;
  const { PAD, TITLE_H, ROW_H, SWATCH, GAP, BAR_W } = LEGEND_METRICS;
  const axes = o.axes ?? null;
  const titleH = o.title ? TITLE_BAND : 0;
  const left = MARGIN + (axes ? AXIS_BAND : 0);
  const top = MARGIN + titleH;
  const plotBottom = o.height + (axes ? AXIS_BAND : 0);

  const strip = legendsLayout(o.legend, o.height, measure);
  const legendX = left + o.width + (strip.width > 0 ? LEGEND_GAP : 0);
  const legendList = o.legend == null ? [] : Array.isArray(o.legend) ? o.legend : [o.legend];
  const captions = legendList.flatMap((l) => (l.type === "continuous" && l.caption ? [l.caption] : []));

  const width = Math.ceil(legendX + strip.width + MARGIN);
  const noteFont = `${NOTE_SIZE}px Inter, system-ui, sans-serif`;
  const noteLines = [...captions, ...(o.notes ?? [])].flatMap((n) =>
    wrapText(n, width - 2 * MARGIN, noteFont, measure),
  );
  const bodyH = Math.max(plotBottom, strip.height);
  const height = Math.ceil(top + bodyH + (noteLines.length ? 6 + noteLines.length * NOTE_LINE : 0) + MARGIN);

  const out: string[] = [];
  out.push(
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1"` +
      ` width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="${FONT_FAMILY}">`,
  );
  out.push(`<rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"/>`);

  // Gradients for continuous legends, and a clip box per panel
  out.push("<defs>");
  strip.layouts.forEach((layout, k) => {
    if (!layout.bar) return;
    const map = layout.bar.colormap ?? o.colormap;
    out.push(`<linearGradient id="legend-bar-${k}" x1="0" y1="1" x2="0" y2="0">`);
    for (let i = 0; i <= 32; i++) {
      const [r, g, b] = colormapColor(map, i / 32);
      out.push(`<stop offset="${num((i / 32) * 100)}%" stop-color="${hex(r, g, b)}"/>`);
    }
    out.push("</linearGradient>");
  });
  o.panels.forEach((p, i) => {
    out.push(
      `<clipPath id="panel-clip-${i}"><rect x="0" y="0" width="${num(p.view.width)}" height="${num(p.view.height)}"/></clipPath>`,
    );
  });
  out.push("</defs>");

  if (o.title) out.push(textSvg(MARGIN, MARGIN + TITLE_BAND / 2 - 2, o.title, { size: 14, weight: 600 }));

  let circles = 0;
  out.push(`<g transform="translate(${left} ${top})">`);
  o.panels.forEach((p, i) => {
    if (p.title) {
      const { box, label, subtitle } = p.title;
      const y = box.y + box.height / 2;
      out.push(textSvg(box.x + 4, y, label, { weight: 600 }));
      if (subtitle) {
        const w = measure(label, LEGEND_TITLE_FONT);
        out.push(textSvg(box.x + 4 + w + 8, y, subtitle, { fill: MUTED }));
      }
    }
    const { x, y, width: w, height: h } = p.view;
    out.push(`<g class="panel" transform="translate(${num(x)} ${num(y)})">`);
    out.push(`<g class="points" clip-path="url(#panel-clip-${i})">`);
    if (o.mode === "raster" && p.sets.length) {
      if (!o.rasterize) throw new Error("No rasteriser for points as an image.");
      const href = o.rasterize(p.sets, w, h, RASTER_SCALE);
      out.push(
        `<image x="0" y="0" width="${num(w)}" height="${num(h)}" preserveAspectRatio="none" xlink:href="${href}"/>`,
      );
    } else {
      for (const set of p.sets) circles += circlesSvg(set, p.view, out);
    }
    out.push("</g>");
    if (p.labels?.length) labelsSvg(p.labels, out);
    if (p.frame) {
      out.push(
        `<rect x="0.5" y="0.5" width="${num(w - 1)}" height="${num(h - 1)}" fill="none" stroke="${FRAME}" stroke-width="1"/>`,
      );
    }
    out.push("</g>");
  });
  if (axes) {
    out.push(textSvg(o.width / 2, o.height + AXIS_BAND / 2, axes[0], { anchor: "middle", fill: MUTED }));
    out.push(textSvg(-AXIS_BAND / 2, o.height / 2, axes[1], { anchor: "middle", fill: MUTED, rotate: true }));
  }
  out.push("</g>");

  // Legend(s), as in the PNG
  strip.layouts.forEach((layout, k) => {
    out.push(`<g class="legend" transform="translate(${num(legendX + layout.x)} ${top})">`);
    if (layout.title) out.push(textSvg(PAD, PAD + TITLE_H / 2 - 2, layout.title, { size: 13, weight: 600 }));
    for (const e of layout.entries) {
      if (e.color) {
        const [r, g, b] = e.color;
        out.push(
          `<circle cx="${num(e.x + SWATCH / 2)}" cy="${num(e.y + ROW_H / 2)}" r="${num(SWATCH / 2)}" fill="${hex(r, g, b)}"/>`,
        );
      }
      out.push(textSvg(e.x + SWATCH + GAP, e.y + ROW_H / 2, e.label, { fill: e.color ? TEXT : MUTED }));
    }
    if (layout.bar) {
      const { x, y, h, maxLabel, minLabel, midLabel } = layout.bar;
      out.push(`<rect x="${x}" y="${y}" width="${BAR_W}" height="${num(h)}" fill="url(#legend-bar-${k})"/>`);
      out.push(textSvg(x + BAR_W + GAP, y + 6, maxLabel));
      if (midLabel !== null) out.push(textSvg(x + BAR_W + GAP, y + h / 2, midLabel));
      out.push(textSvg(x + BAR_W + GAP, y + h - 6, minLabel));
    }
    out.push("</g>");
  });

  noteLines.forEach((line, i) => {
    out.push(textSvg(MARGIN, top + bodyH + 6 + NOTE_LINE * i + NOTE_LINE / 2, line, { size: NOTE_SIZE, fill: MUTED }));
  });
  out.push("</svg>");
  return { svg: out.join("\n"), width, height, circles };
}

/**
 * World positions -> view pixels with `project` (a deck.gl viewport's), in draw
 * order. In 3D the points are sorted far to near, as the GPU's depth test shows
 * them (projected z grows with distance).
 */
export function projectPoints(
  project: (p: number[]) => number[],
  positions: Float32Array,
  stride: 2 | 3,
  colors: Uint8Array,
): { xy: Float32Array; colors: Uint8Array } {
  const n = positions.length / stride;
  const xy = new Float32Array(n * 2);
  const depth = stride === 3 ? new Float32Array(n) : null;
  const p = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    p[0] = positions[i * stride]!;
    p[1] = positions[i * stride + 1]!;
    p[2] = stride === 3 ? positions[i * stride + 2]! : 0;
    const s = project(stride === 3 ? p : [p[0], p[1]]);
    xy[i * 2] = s[0]!;
    xy[i * 2 + 1] = s[1]!;
    if (depth) depth[i] = s[2] ?? 0;
  }
  if (!depth) return { xy, colors };
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => depth[b]! - depth[a]!);
  const sxy = new Float32Array(n * 2);
  const sc = new Uint8Array(n * 4);
  order.forEach((src, dst) => {
    sxy[dst * 2] = xy[src * 2]!;
    sxy[dst * 2 + 1] = xy[src * 2 + 1]!;
    sc.set(colors.subarray(src * 4, src * 4 + 4), dst * 4);
  });
  return { xy: sxy, colors: sc };
}

/** A point's radius on screen, px: ScatterplotLayer's clamped radius (2D) or PointCloudLayer's (3D). */
export function pointRadius(pointSize: number, dimensions: 2 | 3): number {
  return dimensions === 3 ? pointSize * 2 : Math.min(20, Math.max(0.5, pointSize));
}

/** Placed group labels (lib/centroidLabels.ts) at their spot in view pixels. */
export function svgLabels(
  placed: readonly { label: string; position: number[]; offset: [number, number] }[],
  project: (p: number[]) => number[],
  highlight: string | null | undefined,
): SvgLabel[] {
  const hasHighlight = highlight != null && placed.some((l) => l.label === highlight);
  return placed.map((l) => {
    const at = project(l.position);
    return {
      text: l.label,
      x: at[0]! + l.offset[0],
      y: at[1]! + l.offset[1],
      faded: hasHighlight && l.label !== highlight,
    };
  });
}

/** Let the menu show its spinner before a long synchronous build. */
export const nextFrame = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/** The browser rasteriser: the points drawn as circles on a transparent canvas. */
export const canvasRasterizer: Rasterizer = (sets, width, height, scale) => {
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("2D canvas not available");
  ctx.scale(scale, scale);
  for (const set of sets) {
    const n = set.xy.length / 2;
    const r = set.radius;
    const one = set.colors instanceof Uint8Array ? null : set.colors;
    if (one) ctx.fillStyle = `rgba(${one[0]},${one[1]},${one[2]},${one[3] / 255})`;
    for (let i = 0; i < n; i++) {
      if (!one) {
        const c = set.colors as Uint8Array;
        if (c[i * 4 + 3] === 0) continue;
        ctx.fillStyle = `rgba(${c[i * 4]},${c[i * 4 + 1]},${c[i * 4 + 2]},${c[i * 4 + 3]! / 255})`;
      }
      ctx.beginPath();
      ctx.arc(set.xy[i * 2]!, set.xy[i * 2 + 1]!, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  return canvas.toDataURL("image/png");
};
