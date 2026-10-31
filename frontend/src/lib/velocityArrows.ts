/** Velocity arrows on the main scatter: grid vectors to line segments (shaft + two head strokes). */
import type { VelocityGrid } from "@/api/velocity";

export interface ArrowSegment {
  from: [number, number];
  to: [number, number];
}

export const ARROW_DEFAULTS = { on: true, scale: 1, density: 1 } as const;
export const ARROW_SCALE_RANGE = [0.25, 4] as const;
export const ARROW_DENSITY_RANGE = [0.4, 1.6] as const;
/** The 95th-percentile arrow at scale 1 is this many grid steps long: long enough to read, short of the next arrow's tail. */
const STEPS_AT_P95 = 1.6;
const HEAD_SHARE = 0.32;
const HEAD_ANGLE = (28 * Math.PI) / 180;
/** Arrows under this share of the 95th percentile are dropped: a dot says nothing about direction. */
const MIN_SHARE = 0.04;

/** Embedding units per unit of the grid's vectors. */
export function arrowFactor(grid: Pick<VelocityGrid, "step" | "length_p95">, scale: number): number {
  return grid.length_p95 > 0 ? (grid.step * STEPS_AT_P95 * scale) / grid.length_p95 : 0;
}

export function arrowSegments(grid: VelocityGrid, scale: number): ArrowSegment[] {
  const factor = arrowFactor(grid, scale);
  const out: ArrowSegment[] = [];
  if (!factor) return out;
  const floor = grid.length_p95 * MIN_SHARE;
  for (let i = 0; i < grid.n; i++) {
    const u = grid.u[i]!;
    const v = grid.v[i]!;
    const length = Math.hypot(u, v);
    if (!(length > floor)) continue;
    const x0 = grid.x[i]!;
    const y0 = grid.y[i]!;
    const x1 = x0 + u * factor;
    const y1 = y0 + v * factor;
    out.push({ from: [x0, y0], to: [x1, y1] });
    const head = length * factor * HEAD_SHARE;
    const angle = Math.atan2(v, u);
    for (const side of [-1, 1]) {
      const a = angle + Math.PI + side * HEAD_ANGLE;
      out.push({ from: [x1, y1], to: [x1 + Math.cos(a) * head, y1 + Math.sin(a) * head] });
    }
  }
  return out;
}

/** The PCA plot showing another pair than PC1 and PC2. */
export function isOtherPcPair(embedding: string, dims: readonly [number, number]): boolean {
  return embedding === "X_pca" && (dims[0] !== 0 || dims[1] !== 1);
}

/** Why no arrows are drawn on this view, or null when they can be. */
export function arrowsBlocked(dimensions: number, embedding: string, pcaPair: boolean): string | null {
  if (dimensions === 3) return "Arrows are not drawn on a 3D embedding: switch to a 2D one.";
  if (pcaPair) return "Arrows are drawn on PC1 and PC2 only: pick that pair.";
  if (embedding === "spatial") return "Arrows are not drawn on the tissue plot.";
  return null;
}

/** Near black on a light plot, near white on a dark one; the colours under them vary, so semi-opaque stays out. */
export function arrowColor(background: "white" | "dark"): [number, number, number, number] {
  return background === "dark" ? [240, 244, 248, 235] : [20, 24, 32, 235];
}
