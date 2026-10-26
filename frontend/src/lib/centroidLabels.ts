/**
 * Group labels drawn on the embedding: each group's name where most of its
 * cells are (the middle of its densest region, so a scattered group is not
 * labelled inside another group, as its median would be), as a deck.gl
 * TextLayer with a halo that reads on light and dark points alike.
 *
 * Placement is redone for the current camera (placeLabels): a label is pushed
 * back inside the view when its text would cross an edge, and where labels
 * would overlap the bigger group keeps its label (the smaller one moves a line
 * up or down, or waits until zooming in makes room). Boxes drawn over the plot
 * (its legend) are kept clear the same way.
 */
import { TextLayer } from "@deck.gl/layers";
import type { View } from "@deck.gl/core";

/** Labels are on by default for a column with at most this many groups. */
export const LABELS_AUTO_MAX_GROUPS = 40;
const MAX_LABEL = 32;
/** Text size (px) and font, shared by the TextLayer and the width estimate. */
export const LABEL_SIZE = 12;
const LABEL_FONT = "Inter, system-ui, sans-serif";
const LABEL_WEIGHT = 600;
/** Halo around the glyphs (outlineWidth 12 of a 64 px SDF drawn at 12 px), px */
const HALO = 2;
/** Space kept free at the view's edge, and between two labels, px */
const EDGE = 2;
const GAP = 2;

/** Histogram bins along the embedding's longest axis, for the density peak. */
const PEAK_BINS = { 2: 64, 3: 24 } as const;
/** Box-blur radius in bins: the peak is the densest window of (2r+1)^d bins. */
const PEAK_RADIUS = { 2: 2, 3: 1 } as const;

/** The user's explicit choice for one colouring (its key); else the default applies. */
export interface LabelChoice {
  key: string;
  show: boolean;
}

/** The label-choice key of a compare-colourings set (one choice for all its panels). */
export const compareLabelKey = (columns: readonly string[]) => `compare:${columns.join("\u0000")}`;

/** Whether a colouring with `nGroups` groups shows its labels. */
export function labelsShown(choice: LabelChoice | null, key: string, nGroups: number): boolean {
  if (choice && choice.key === key) return choice.show;
  return nGroups > 0 && nGroups <= LABELS_AUTO_MAX_GROUPS;
}

export interface GroupCentroid {
  /** The group's code (index in `labels`) */
  code: number;
  label: string;
  /** Label anchor, the middle of the group's densest region: [x, y] or [x, y, z] */
  position: number[];
  /** Drawn cells in the group */
  count: number;
}

/**
 * The label anchor of every group with at least one drawn cell, in the
 * column's order. `codes` holds each cell's group (-1 = none); `indices`
 * limits the cells to those drawn (a sample, or one split panel's), null = all
 * cells. Cells with non-finite coordinates are skipped.
 *
 * The anchor is the group's cell nearest the mean of its cells in its densest
 * window (so it always sits on the group, even a scattered one): the drawn
 * cells are binned on a square grid over their bounds (64 bins along the
 * longest axis in 2D, 24 in 3D), each group's counts are box-blurred, and the
 * peak wins (a tie goes to the bin nearest the group's mean). Three passes over
 * the cells plus a blur per group, so it stays fast at 200k cells.
 */
export function groupCentroids(
  positions: Float32Array,
  stride: 2 | 3,
  codes: ArrayLike<number>,
  labels: readonly string[],
  indices: ArrayLike<number> | null = null,
): GroupCentroid[] {
  const k = labels.length;
  const n = indices ? indices.length : Math.min(codes.length, positions.length / stride);
  const cellAt = (s: number) => (indices ? indices[s]! : s);

  // Pass 1: each drawn cell's group (-1 = skipped), group sizes and means, bounds.
  const group = new Int32Array(n).fill(-1);
  const counts = new Uint32Array(k);
  const sums = new Float64Array(k * stride);
  const lo = new Float64Array(stride).fill(Infinity);
  const hi = new Float64Array(stride).fill(-Infinity);
  for (let s = 0; s < n; s++) {
    const i = cellAt(s);
    const c = codes[i]!;
    if (!(c >= 0 && c < k) || !Number.isInteger(c)) continue;
    let finite = true;
    for (let d = 0; d < stride; d++) if (!Number.isFinite(positions[i * stride + d]!)) finite = false;
    if (!finite) continue;
    group[s] = c;
    counts[c] = counts[c]! + 1;
    for (let d = 0; d < stride; d++) {
      const v = positions[i * stride + d]!;
      sums[c * stride + d] = sums[c * stride + d]! + v;
      if (v < lo[d]!) lo[d] = v;
      if (v > hi[d]!) hi[d] = v;
    }
  }

  // Square bins: the longest axis gets PEAK_BINS bins, the others as many as fit.
  const nb = PEAK_BINS[stride];
  const r = PEAK_RADIUS[stride];
  let span = 0;
  for (let d = 0; d < stride; d++) span = Math.max(span, hi[d]! - lo[d]!);
  const binW = span > 0 ? span / nb : 1;
  const dims = Array.from({ length: stride }, (_, d) =>
    Math.max(1, Math.min(nb, Math.ceil((hi[d]! - lo[d]!) / binW))),
  );
  const steps = dims.map((_, d) => dims.slice(0, d).reduce((a, b) => a * b, 1));
  const nBins = dims.reduce((a, b) => a * b, 1);

  // Pass 2: each cell's bin, and every group's histogram.
  const coord = new Uint8Array(n * stride);
  const hist = new Uint32Array(k * nBins);
  for (let s = 0; s < n; s++) {
    const c = group[s]!;
    if (c < 0) continue;
    const i = cellAt(s);
    let flat = 0;
    for (let d = 0; d < stride; d++) {
      const b = Math.min(dims[d]! - 1, Math.floor((positions[i * stride + d]! - lo[d]!) / binW));
      coord[s * stride + d] = b;
      flat += b * steps[d]!;
    }
    hist[c * nBins + flat] = hist[c * nBins + flat]! + 1;
  }

  // Per group: blur, then the densest bin (a tie goes to the one nearest the mean).
  const peak: number[][] = Array.from({ length: k }, () => []);
  const smooth = new Float32Array(nBins);
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    smooth.set(hist.subarray(c * nBins, (c + 1) * nBins));
    boxBlur(smooth, dims, steps, r);
    const mean = Array.from({ length: stride }, (_, d) => (sums[c * stride + d]! / counts[c]! - lo[d]!) / binW);
    let best = -1;
    let bestDist = Infinity;
    let bestBin = 0;
    for (let b = 0; b < nBins; b++) {
      const v = smooth[b]!;
      if (v < best) continue;
      let dist = 0;
      for (let d = 0; d < stride; d++) dist += (Math.floor(b / steps[d]!) % dims[d]! + 0.5 - mean[d]!) ** 2;
      if (v > best || dist < bestDist) {
        best = v;
        bestDist = dist;
        bestBin = b;
      }
    }
    peak[c] = dims.map((len, d) => Math.floor(bestBin / steps[d]!) % len);
  }

  // Pass 3: the anchor is the mean of the group's cells in its densest window.
  const winSums = new Float64Array(k * stride);
  const winCounts = new Uint32Array(k);
  for (let s = 0; s < n; s++) {
    const c = group[s]!;
    if (c < 0) continue;
    const p = peak[c]!;
    let inside = true;
    for (let d = 0; d < stride; d++) if (Math.abs(coord[s * stride + d]! - p[d]!) > r) inside = false;
    if (!inside) continue;
    const i = cellAt(s);
    winCounts[c] = winCounts[c]! + 1;
    for (let d = 0; d < stride; d++) winSums[c * stride + d] = winSums[c * stride + d]! + positions[i * stride + d]!;
  }

  // Pass 4: snap to the group's cell nearest that mean. A scattered group's window can
  // hold a few cells far apart, whose mean is empty space (a label floating off the group).
  const means = new Float64Array(k * stride);
  for (let c = 0; c < k; c++)
    for (let d = 0; d < stride; d++) means[c * stride + d] = winSums[c * stride + d]! / winCounts[c]!;
  const nearest = new Int32Array(k).fill(-1);
  const nearestDist = new Float64Array(k).fill(Infinity);
  for (let s = 0; s < n; s++) {
    const c = group[s]!;
    if (c < 0) continue;
    const i = cellAt(s);
    let dist = 0;
    for (let d = 0; d < stride; d++) dist += (positions[i * stride + d]! - means[c * stride + d]!) ** 2;
    if (dist < nearestDist[c]!) {
      nearestDist[c] = dist;
      nearest[c] = i;
    }
  }

  const out: GroupCentroid[] = [];
  for (let c = 0; c < k; c++) {
    if (counts[c] === 0) continue;
    const i = nearest[c]!;
    const position = Array.from({ length: stride }, (_, d) => positions[i * stride + d]!);
    out.push({ code: c, label: labels[c]!, position, count: counts[c]! });
  }
  return out;
}

/** In-place separable box blur of radius `r` (window clipped at the grid's edge). */
function boxBlur(a: Float32Array, dims: readonly number[], steps: readonly number[], r: number) {
  const line = new Float32Array(Math.max(...dims));
  for (let d = 0; d < dims.length; d++) {
    const len = dims[d]!;
    const step = steps[d]!;
    if (len === 1 || r === 0) continue;
    const block = step * len;
    for (let outer = 0; outer < a.length; outer += block) {
      for (let inner = 0; inner < step; inner++) {
        const base = outer + inner;
        for (let j = 0; j < len; j++) line[j] = a[base + j * step]!;
        let sum = 0;
        for (let j = 0; j <= Math.min(r, len - 1); j++) sum += line[j]!;
        for (let j = 0; j < len; j++) {
          a[base + j * step] = sum;
          if (j + r + 1 < len) sum += line[j + r + 1]!;
          if (j - r >= 0) sum -= line[j - r]!;
        }
      }
    }
  }
}

const truncate = (s: string) => (s.length > MAX_LABEL ? `${s.slice(0, MAX_LABEL - 1)}…` : s);

// ---------------------------------------------------------------------------
// Placement for the current camera: inside the view, and not on top of another
// ---------------------------------------------------------------------------

/** A view's size and its world -> pixel projection (from its top-left, y down). */
export interface LabelView {
  width: number;
  height: number;
  project: (position: number[]) => number[];
}

/** The LabelView of a deck.gl view (2D or 3D) of that size, at that camera. */
export function labelViewOf(
  view: View,
  width: number,
  height: number,
  viewState: Record<string, unknown>,
): LabelView | null {
  if (!(width > 0 && height > 0)) return null;
  const vp = view.makeViewport({ width, height, viewState });
  if (!vp) return null;
  return { width: vp.width, height: vp.height, project: (p) => vp.project(p) };
}

let measureCtx: OffscreenCanvasRenderingContext2D | null | undefined;

/**
 * A label's text width in px. Measured with a canvas where there is one (an
 * OffscreenCanvas, so jsdom and old browsers take the estimate instead).
 */
export function labelWidth(text: string): number {
  if (measureCtx === undefined) {
    try {
      measureCtx = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1).getContext("2d") : null;
    } catch {
      measureCtx = null;
    }
  }
  if (measureCtx) {
    measureCtx.font = `${LABEL_WEIGHT} ${LABEL_SIZE}px ${LABEL_FONT}`;
    return measureCtx.measureText(text).width;
  }
  return text.length * LABEL_SIZE * 0.6;
}

/** A label placed for one camera: its screen shift (px, y down) from the anchor. */
export interface PlacedLabel extends GroupCentroid {
  offset: [number, number];
}

/** A screen box (px from the view's top-left, y down). */
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** Centre of a `size`-long box near `centre`, kept `EDGE` px inside [0, extent]. */
function keepInside(centre: number, size: number, extent: number): number {
  const min = EDGE + size / 2;
  const max = extent - EDGE - size / 2;
  if (min > max) return min; // longer than the view: start at its edge
  return Math.min(max, Math.max(min, centre));
}

const overlaps = (a: Box, b: Box) =>
  a.x0 < b.x1 + GAP && b.x0 < a.x1 + GAP && a.y0 < b.y1 + GAP && b.y0 < a.y1 + GAP;

/**
 * The labels to draw at this camera, in the input's order. A label whose
 * anchor is out of view is left out; one whose text would cross an edge is
 * shifted back inside. Labels are placed biggest group first (a highlighted
 * group before all), each at its anchor, else a line above, else a line below;
 * a label that fits nowhere is left out until zooming in makes room. `avoid`
 * boxes (the legend over the plot) count as already taken.
 */
export function placeLabels(
  centroids: readonly GroupCentroid[],
  view: LabelView,
  opts: { highlight?: string | null; measure?: (text: string) => number; avoid?: readonly Box[] } = {},
): PlacedLabel[] {
  const { highlight = null, measure = labelWidth, avoid = [] } = opts;
  const order = centroids
    .map((_, i) => i)
    .sort((a, b) => {
      const ca = centroids[a]!;
      const cb = centroids[b]!;
      return (
        Number(cb.label === highlight) - Number(ca.label === highlight) ||
        cb.count - ca.count ||
        ca.code - cb.code
      );
    });
  const placed: Box[] = [...avoid];
  const offsets = new Map<number, [number, number]>();
  const h = LABEL_SIZE + 2 * HALO;
  for (const i of order) {
    const c = centroids[i]!;
    const at = view.project(c.position);
    const sx = at[0]!;
    const sy = at[1]!;
    if (!(sx >= 0 && sx <= view.width && sy >= 0 && sy <= view.height)) continue;
    const w = measure(truncate(c.label)) + 2 * HALO;
    for (const dy of [0, -(h + GAP), h + GAP]) {
      const cx = keepInside(sx, w, view.width);
      const cy = keepInside(sy + dy, h, view.height);
      const box = { x0: cx - w / 2, y0: cy - h / 2, x1: cx + w / 2, y1: cy + h / 2 };
      if (placed.some((p) => overlaps(p, box))) continue;
      placed.push(box);
      offsets.set(i, [cx - sx, cy - sy]);
      break;
    }
  }
  return centroids.flatMap((c, i) => {
    const offset = offsets.get(i);
    return offset ? [{ ...c, offset }] : [];
  });
}

type DrawnLabel = GroupCentroid & { offset?: [number, number] };

/**
 * TextLayer(s) for the labels (placed ones carry their screen shift). With a
 * highlighted group, the others are drawn faded in a second layer (a layer's
 * opacity fades the text and its halo together, which a per-label colour
 * cannot do with an SDF outline).
 */
export function centroidLabelLayers(
  id: string,
  centroids: readonly DrawnLabel[],
  opts: { dark: boolean; highlight?: string | null },
): TextLayer<DrawnLabel>[] {
  const { dark, highlight = null } = opts;
  const hasHighlight = highlight != null && centroids.some((c) => c.label === highlight);
  const make = (layerId: string, data: readonly DrawnLabel[], opacity: number) =>
    new TextLayer<DrawnLabel>({
      id: layerId,
      data,
      getPosition: (d) => d.position as [number, number] | [number, number, number],
      getPixelOffset: (d) => d.offset ?? [0, 0],
      getText: (d) => truncate(d.label),
      getSize: LABEL_SIZE,
      sizeUnits: "pixels",
      getColor: dark ? [248, 250, 252, 255] : [15, 23, 42, 255],
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
      fontFamily: LABEL_FONT,
      fontWeight: LABEL_WEIGHT,
      characterSet: "auto",
      // A wide SDF radius and buffer leave room for a ~2 px halo at 12 px.
      fontSettings: { sdf: true, fontSize: 64, buffer: 12, radius: 16, smoothing: 0.15 },
      outlineWidth: 12,
      outlineColor: dark ? [15, 23, 42, 235] : [255, 255, 255, 235],
      opacity,
      pickable: false,
      // Over the points, also in 3D where nearer points would hide a label.
      parameters: { depthCompare: "always", depthWriteEnabled: false },
    });
  if (!hasHighlight) return [make(id, centroids, 1)];
  return [
    make(id, centroids.filter((c) => c.label !== highlight), 0.3),
    make(`${id}-hl`, centroids.filter((c) => c.label === highlight), 1),
  ];
}
