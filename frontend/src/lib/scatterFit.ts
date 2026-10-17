/**
 * The automatic 2D camera fit of the embedding scatter, kept clear of a box
 * drawn over the plot (its legend): when the plain fit would hide cells under
 * the box, the cells are fitted into the largest strip beside it instead.
 */

/** A screen box (px from the plot's top-left, y down). */
export interface ScreenBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Fit2D {
  target: [number, number, number];
  zoom: number;
}

/** Margin around the data, as a fraction of its extent on each side. */
const PADDING = 0.05;
/** Cells the plain fit may leave under the box (fraction of all) before it moves. */
export const HIDDEN_MAX = 0.002;
/** A fit beside the box may shrink the plot to this fraction of the plain fit's scale, no less. */
const MIN_SCALE = 0.5;
/** Space kept between the box and the strip the cells are fitted into, px. */
const MARGIN = 4;

interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function bounds(positions: Float32Array): Bounds {
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
  return { minX, maxX, minY, maxY };
}

/**
 * The camera (OrthographicView, flipY false: screen y grows downwards while
 * data y grows upwards) that fits the data's padded bounds into `rect`.
 */
function fitInto(b: Bounds, rect: ScreenBox, width: number, height: number): Fit2D & { scale: number } {
  const w = b.maxX - b.minX || 1;
  const h = b.maxY - b.minY || 1;
  const scale = Math.min((rect.x1 - rect.x0) / (w * (1 + 2 * PADDING)), (rect.y1 - rect.y0) / (h * (1 + 2 * PADDING)));
  const cx = b.minX + w / 2;
  const cy = b.minY + h / 2;
  return {
    target: [cx - ((rect.x0 + rect.x1) / 2 - width / 2) / scale, cy + ((rect.y0 + rect.y1) / 2 - height / 2) / scale, 0],
    zoom: Math.log2(scale),
    scale,
  };
}

/** How many cells a camera puts under `box`. */
function countUnder(positions: Float32Array, fit: Fit2D, box: ScreenBox, width: number, height: number): number {
  const s = 2 ** fit.zoom;
  const [tx, ty] = fit.target;
  let n = 0;
  for (let i = 0; i < positions.length; i += 2) {
    const sx = width / 2 + (positions[i]! - tx) * s;
    const sy = height / 2 - (positions[i + 1]! - ty) * s;
    if (sx >= box.x0 && sx <= box.x1 && sy >= box.y0 && sy <= box.y1) n++;
  }
  return n;
}

/**
 * The fit of `positions` (x, y pairs) in a `width` x `height` plot: the plain
 * fit (the whole plot, 5% padding) unless it leaves more than HIDDEN_MAX of
 * the cells under `avoid`; then the fit into the largest strip of the plot
 * left, right, above or below the box, if that keeps at least half the scale.
 */
export function fitAvoiding(
  positions: Float32Array,
  width: number,
  height: number,
  avoid: ScreenBox | null = null,
): Fit2D {
  const b = bounds(positions);
  const plain = fitInto(b, { x0: 0, y0: 0, x1: width, y1: height }, width, height);
  const result = (f: Fit2D): Fit2D => ({ target: f.target, zoom: f.zoom });
  const n = positions.length / 2;
  if (!avoid || n === 0 || !(width > 0 && height > 0)) return result(plain);
  if (countUnder(positions, plain, avoid, width, height) <= HIDDEN_MAX * n) return result(plain);

  const strips: ScreenBox[] = [
    { x0: avoid.x1 + MARGIN, y0: 0, x1: width, y1: height }, // right of it
    { x0: 0, y0: 0, x1: avoid.x0 - MARGIN, y1: height }, // left
    { x0: 0, y0: 0, x1: width, y1: avoid.y0 - MARGIN }, // above
    { x0: 0, y0: avoid.y1 + MARGIN, x1: width, y1: height }, // below
  ].filter((r) => r.x1 - r.x0 > 0 && r.y1 - r.y0 > 0);
  let best: (Fit2D & { scale: number }) | null = null;
  for (const r of strips) {
    const f = fitInto(b, r, width, height);
    if (!best || f.scale > best.scale) best = f;
  }
  return best && best.scale >= MIN_SCALE * plain.scale ? result(best) : result(plain);
}
