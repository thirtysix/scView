/**
 * C039 follow-up: a lasso drawn in one panel of a split / compare / gene grid.
 * Every panel is a view of the same size on the same camera
 * (MultiPanelScatter), so an outline in one panel's own pixels maps to data
 * space exactly as an outline on the single plot does (LassoSelector).
 */
import { cellsInPolygon3D, pointInPolygon, screenToData } from "@/components/plots/LassoSelector";

/**
 * The cells under `polygon` (pixels from the panel view's top-left) of a view
 * `width` x `height` under `viewState`, among every cell of `positions` (not
 * only the drawn sample). `keep(i)` limits it, e.g. to the panel's own cells.
 */
export function cellsInPanelLasso(
  positions: Float32Array,
  dimensions: 2 | 3,
  polygon: [number, number][],
  viewState: Record<string, unknown>,
  width: number,
  height: number,
  keep?: (i: number) => boolean,
): Set<number> {
  if (polygon.length < 3 || width <= 0 || height <= 0) return new Set();
  if (dimensions === 3) {
    const all = cellsInPolygon3D(positions, polygon, viewState, width, height);
    if (!keep) return all;
    const out = new Set<number>();
    for (const i of all) if (keep(i)) out.add(i);
    return out;
  }
  const data = polygon.map(([sx, sy]) => screenToData(sx, sy, viewState, width, height));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [x, y] of data) {
    x0 = Math.min(x0, x); x1 = Math.max(x1, x);
    y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  }
  const out = new Set<number>();
  const n = Math.floor(positions.length / 2);
  for (let i = 0; i < n; i++) {
    const cx = positions[i * 2]!;
    const cy = positions[i * 2 + 1]!;
    if (cx < x0 || cx > x1 || cy < y0 || cy > y1) continue;
    if (keep && !keep(i)) continue;
    if (pointInPolygon(cx, cy, data)) out.add(i);
  }
  return out;
}

/** "120 selected of 4,210", or the plain count when nothing is selected. */
export function panelSelectionText(selected: number | null, total: number): string {
  return selected == null
    ? `${total.toLocaleString()} cells`
    : `${selected.toLocaleString()} selected of ${total.toLocaleString()}`;
}

/** Which panel of a grid a point (pixels in the grid area) falls in, or -1. */
export function panelAt(
  cells: readonly { view: { x: number; y: number; width: number; height: number } }[],
  x: number,
  y: number,
): number {
  return cells.findIndex(
    ({ view: v }) => x >= v.x && x <= v.x + v.width && y >= v.y && y <= v.y + v.height,
  );
}
