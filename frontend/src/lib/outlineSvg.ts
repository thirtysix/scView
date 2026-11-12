/**
 * Segmented cell outlines (lib/cellOutlines.ts) in the SVG / PDF figure of the tissue plot: the
 * mesh the plot draws, projected with the same camera as the points, each outline filled in its
 * cell's colour and edged as on screen (on white, so the light-background edge).
 *
 * Only outlines that reach into the exported view are written, as path data in tenths of a pixel,
 * relative moves, simplified to within SIMPLIFY_PX, and one <path> per colour (a few thousand
 * rings each at most), so the file stays small. Over MAX_FIGURE_OUTLINES outlines or
 * MAX_FIGURE_OUTLINE_VERTICES vertices the outlines are left out and the caller says so.
 */
import { EDGE_RGBA, drawsEdges, type OutlineMesh } from "@/lib/cellOutlines";

/** Above this many outlines in view a figure leaves them out (the plot itself stops at 30,000). */
export const MAX_FIGURE_OUTLINES = 50_000;
/** ...or above this many vertices once simplified (about 9 bytes each: ~5 MB of path data). */
export const MAX_FIGURE_OUTLINE_VERTICES = 600_000;
/** A vertex nearer than this (px) to the line its neighbours make is dropped. */
export const SIMPLIFY_PX = 0.3;
/** Rings per <path>: keeps each path's data a size every SVG and PDF reader takes. */
const RINGS_PER_PATH = 2_000;

type Rgba = readonly [number, number, number, number];

/** Outlines ready to write: rings in view pixels, grouped by colour, and the edge. */
export interface SvgOutlines {
  /** Path data per colour, in the order the colours first appear in the mesh */
  groups: { rgba: Rgba; d: string[] }[];
  /** The edge round each outline; null when cells are too small on screen for one (as on the plot) */
  edge: { rgba: Rgba; width: number } | null;
}

export interface FigureOutlines {
  /** Null when there are too many (`omitted`) */
  outlines: SvgOutlines | null;
  /** Outlines that reach into the view */
  count: number;
  /** Their vertices once simplified (counted until a limit is passed) */
  vertices: number;
  omitted: boolean;
}

export interface FigureOutlineLimits {
  maxOutlines: number;
  maxVertices: number;
}

const DEFAULT_LIMITS: FigureOutlineLimits = {
  maxOutlines: MAX_FIGURE_OUTLINES,
  maxVertices: MAX_FIGURE_OUTLINE_VERTICES,
};

/** Tenths of a pixel as text ("12.3", "-0.4", "7"). */
const tenths = (n: number) => String(n / 10);
const hex = (r: number, g: number, b: number) => `#${((1 << 24) | (r << 16) | (g << 8) | b).toString(16).slice(1)}`;
const alphaText = (a: number) => String(Math.round((a / 255) * 1000) / 1000);

/**
 * Which vertices of a closed ring (pixels, x, y pairs, `n` of them) stay within `tol` of the
 * original: Douglas-Peucker from vertex 0 and the vertex farthest from it.
 */
export function simplifyRing(px: Float64Array, n: number, tol: number): Uint8Array {
  const keep = new Uint8Array(n);
  if (n <= 3) return keep.fill(1);
  let far = 0;
  let farD = -1;
  for (let i = 1; i < n; i++) {
    const d = (px[i * 2]! - px[0]!) ** 2 + (px[i * 2 + 1]! - px[1]!) ** 2;
    if (d > farD) {
      farD = d;
      far = i;
    }
  }
  keep[0] = 1;
  keep[far] = 1;
  const tol2 = tol * tol;
  // Spans [a, b] (b may be n: vertex 0 again), split at their farthest vertex while it is off.
  const stack: number[] = [0, far, far, n];
  while (stack.length) {
    const b = stack.pop()!;
    const a = stack.pop()!;
    if (b - a < 2) continue;
    const ax = px[a * 2]!;
    const ay = px[a * 2 + 1]!;
    const bi = b % n;
    const dx = px[bi * 2]! - ax;
    const dy = px[bi * 2 + 1]! - ay;
    const len2 = dx * dx + dy * dy;
    let worst = -1;
    let worstD = tol2;
    for (let i = a + 1; i < b; i++) {
      const ex = px[i * 2]! - ax;
      const ey = px[i * 2 + 1]! - ay;
      const d = len2 > 0 ? (ex * dy - ey * dx) ** 2 / len2 : ex * ex + ey * ey;
      if (d > worstD) {
        worstD = d;
        worst = i;
      }
    }
    if (worst < 0) continue;
    keep[worst] = 1;
    stack.push(a, worst, worst, b);
  }
  return keep;
}

/**
 * The outlines of `mesh` in a figure of `view` pixels: `project` is the plot's camera (world ->
 * view pixels, affine on the 2D tissue plot), `cellColors` the RGBA per cell the plot colours them
 * with, `scale` the screen pixels a world unit (2 ** zoom), which decides the edge as on screen.
 */
export function figureOutlines(
  mesh: OutlineMesh,
  cellColors: Uint8Array,
  project: (p: number[]) => number[],
  view: { width: number; height: number },
  scale: number,
  limits: FigureOutlineLimits = DEFAULT_LIMITS,
): FigureOutlines {
  // The 2D camera is affine: three projections give it, instead of one call a vertex.
  const o = project([0, 0]);
  const ux = project([1, 0]);
  const uy = project([0, 1]);
  const ox = o[0]!;
  const oy = o[1]!;
  const xx = ux[0]! - ox;
  const xy = ux[1]! - oy;
  const yx = uy[0]! - ox;
  const yy = uy[1]! - oy;

  const edge = drawsEdges(mesh, scale) ? { rgba: EDGE_RGBA.light, width: 1 } : null;
  const pad = edge ? edge.width : 0;
  const { cells, startIndices, positions } = mesh;
  const byColor = new Map<number, { rgba: Rgba; d: string[]; rings: string[] }>();
  let px = new Float64Array(64);
  let count = 0;
  let vertices = 0;
  let omitted = false;
  for (let k = 0; k < cells.length; k++) {
    const from = startIndices[k]!;
    const n = startIndices[k + 1]! - from;
    if (n < 3) continue;
    if (px.length < n * 2) px = new Float64Array(n * 4);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const wx = positions[(from + i) * 2]!;
      const wy = positions[(from + i) * 2 + 1]!;
      const sx = ox + wx * xx + wy * yx;
      const sy = oy + wx * xy + wy * yy;
      px[i * 2] = sx;
      px[i * 2 + 1] = sy;
      if (sx < minX) minX = sx;
      if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy;
      if (sy > maxY) maxY = sy;
    }
    // Only outlines that reach into the view (the panel's clip trims the rest).
    if (maxX < -pad || minX > view.width + pad || maxY < -pad || minY > view.height + pad) continue;
    count++;
    if (omitted) continue;
    const keep = simplifyRing(px, n, SIMPLIFY_PX);
    // In tenths of a pixel, relative to the previous kept vertex; repeats dropped.
    let d = "";
    let written = 0;
    let firstX = 0;
    let firstY = 0;
    let lastX = 0;
    let lastY = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      const tx = Math.round(px[i * 2]! * 10);
      const ty = Math.round(px[i * 2 + 1]! * 10);
      if (written === 0) {
        d = `M${tenths(tx)} ${tenths(ty)}`;
        firstX = tx;
        firstY = ty;
      } else {
        if (tx === lastX && ty === lastY) continue;
        const dy = ty - lastY;
        d += `l${tenths(tx - lastX)}${dy < 0 ? "" : " "}${tenths(dy)}`;
      }
      lastX = tx;
      lastY = ty;
      written++;
    }
    if (written > 1 && lastX === firstX && lastY === firstY) written--;
    // Smaller than a fraction of a pixel: nothing to see.
    if (written < 3) continue;
    vertices += written;
    if (count > limits.maxOutlines || vertices > limits.maxVertices) {
      omitted = true;
      byColor.clear();
      continue;
    }
    const at = cells[k]! * 4;
    const r = cellColors[at] ?? 0;
    const g = cellColors[at + 1] ?? 0;
    const b = cellColors[at + 2] ?? 0;
    const a = cellColors[at + 3] ?? 0;
    const key = ((r << 24) | (g << 16) | (b << 8) | a) >>> 0;
    let group = byColor.get(key);
    if (!group) {
      group = { rgba: [r, g, b, a], d: [], rings: [] };
      byColor.set(key, group);
    }
    group.rings.push(`${d}z`);
    if (group.rings.length >= RINGS_PER_PATH) {
      group.d.push(group.rings.join(""));
      group.rings = [];
    }
  }
  if (omitted) return { outlines: null, count, vertices, omitted };
  const groups = [...byColor.values()].map((g) => ({
    rgba: g.rgba,
    d: g.rings.length ? [...g.d, g.rings.join("")] : g.d,
  }));
  return { outlines: { groups, edge }, count, vertices, omitted: false };
}

/**
 * The outlines as SVG, appended to `out`: a group per colour, filled in the cell's colour and
 * opacity (none when it is see-through) with the edge stroked over it. Clipped by `clipId`.
 */
export function outlinesSvg(o: SvgOutlines, clipId: string, out: string[]): void {
  if (!o.groups.length) return;
  const stroke = o.edge
    ? ` stroke="${hex(o.edge.rgba[0], o.edge.rgba[1], o.edge.rgba[2])}" stroke-opacity="${alphaText(o.edge.rgba[3])}"` +
      ` stroke-width="${o.edge.width}" stroke-linejoin="round"`
    : "";
  out.push(`<g class="outlines" clip-path="url(#${clipId})"${stroke}>`);
  for (const g of o.groups) {
    const [r, gr, b, a] = g.rgba;
    if (a === 0 && !o.edge) continue;
    const fill = a === 0 ? `fill="none"` : `fill="${hex(r, gr, b)}"${a === 255 ? "" : ` fill-opacity="${alphaText(a)}"`}`;
    for (const d of g.d) out.push(`<path ${fill} d="${d}"/>`);
  }
  out.push("</g>");
}

/** The figure's note when the outlines were left out. */
export function omittedOutlinesNote(f: FigureOutlines, kind: "cell" | "nucleus", limits: FigureOutlineLimits = DEFAULT_LIMITS): string {
  const word = kind === "cell" ? "Cell outlines" : "Nucleus outlines";
  const over =
    f.count > limits.maxOutlines
      ? `${f.count.toLocaleString()} in view, over the ${limits.maxOutlines.toLocaleString()} a figure holds`
      : `over ${limits.maxVertices.toLocaleString()} outline points in view`;
  return `${word} left out (${over}): zoom in for them. The cells are drawn as points.`;
}
