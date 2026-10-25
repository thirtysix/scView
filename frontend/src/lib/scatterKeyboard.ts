/**
 * Keyboard use of the embedding scatter (components/plots/ScatterKeyboard):
 * pan and zoom the camera, and a rectangle selection drawn with a crosshair,
 * the keyboard's stand-in for the lasso. The rectangle goes through the same
 * cell test as the lasso (components/plots/LassoSelector), so a 3D embedding
 * selects every cell under the rectangle, at any depth.
 */
import { cellsInPolygon3D, pointInPolygon, screenToData } from "@/components/plots/LassoSelector";
import { prettyEmbeddingLabel } from "@/lib/formatting";

/** Share of the plot's width or height one arrow press pans (a 2D plot). */
export const PAN_FRACTION = 0.1;
/** Zoom change of one +/- press (deck.gl zoom is log2: 0.5 = x1.41). */
export const ZOOM_STEP = 0.5;
/** Degrees one arrow press turns a 3D plot. */
export const ROTATE_STEP = 10;
/** Pixels the crosshair moves per arrow press (Shift: CROSSHAIR_FAST). */
export const CROSSHAIR_STEP = 10;
export const CROSSHAIR_FAST = 50;

type ViewState = Record<string, unknown>;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/**
 * The camera after a pan/zoom key, or null when the key does not move it.
 * 2D: arrows pan by PAN_FRACTION of the plot, + / - zoom about the centre.
 * 3D (orbit camera): arrows turn the plot, + / - zoom. "0" (fit) is the
 * caller's: it knows the fitted camera.
 */
export function cameraAfterKey(
  viewState: ViewState,
  key: string,
  size: { width: number; height: number },
  dimensions: 2 | 3,
): ViewState | null {
  const zoom = (viewState.zoom as number) ?? 0;
  const minZoom = (viewState.minZoom as number | undefined) ?? -Infinity;
  const maxZoom = (viewState.maxZoom as number | undefined) ?? Infinity;
  if (key === "+" || key === "=") return { ...viewState, zoom: clamp(zoom + ZOOM_STEP, minZoom, maxZoom) };
  if (key === "-" || key === "_") return { ...viewState, zoom: clamp(zoom - ZOOM_STEP, minZoom, maxZoom) };
  const dir: Record<string, [number, number]> = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, 1],
    ArrowDown: [0, -1],
  };
  const d = dir[key];
  if (!d) return null;
  if (dimensions === 3) {
    const orbit = (viewState.rotationOrbit as number) ?? 0;
    const x = (viewState.rotationX as number) ?? 0;
    return {
      ...viewState,
      rotationOrbit: orbit + d[0] * ROTATE_STEP,
      rotationX: clamp(x + d[1] * ROTATE_STEP, -90, 90),
    };
  }
  // The view moves the way the arrow points: the camera target goes that way
  // (data y grows upwards on the scatter, flipY false).
  const scale = Math.pow(2, zoom);
  const target = (viewState.target as number[]) ?? [0, 0, 0];
  const dx = (d[0] * size.width * PAN_FRACTION) / scale;
  const dy = (d[1] * size.height * PAN_FRACTION) / scale;
  return { ...viewState, target: [target[0]! + dx, target[1]! + dy, ...target.slice(2)] };
}

/** The crosshair after an arrow key (pixels in the plot, kept inside it), or null. */
export function crosshairAfterKey(
  at: [number, number],
  key: string,
  fast: boolean,
  size: { width: number; height: number },
): [number, number] | null {
  const step = fast ? CROSSHAIR_FAST : CROSSHAIR_STEP;
  const dir: Record<string, [number, number]> = {
    ArrowLeft: [-1, 0],
    ArrowRight: [1, 0],
    ArrowUp: [0, -1],
    ArrowDown: [0, 1],
  };
  const d = dir[key];
  if (!d) return null;
  return [clamp(at[0] + d[0] * step, 0, size.width), clamp(at[1] + d[1] * step, 0, size.height)];
}

/** The rectangle with corners a and b, as a closed screen polygon. */
export function rectPolygon(a: [number, number], b: [number, number]): [number, number][] {
  const [x0, x1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
  const [y0, y1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

/**
 * The cells under a screen rectangle (corners in pixels from the plot's
 * top-left), with the camera shown: the lasso's own test.
 */
export function cellsInScreenRect(
  positions: Float32Array,
  dimensions: 2 | 3,
  a: [number, number],
  b: [number, number],
  viewState: ViewState,
  size: { width: number; height: number },
): Set<number> {
  const polygon = rectPolygon(a, b);
  if (dimensions === 3) return cellsInPolygon3D(positions, polygon, viewState, size.width, size.height);
  const data = polygon.map(([x, y]) => screenToData(x, y, viewState, size.width, size.height));
  const selected = new Set<number>();
  const n = positions.length / 2;
  for (let i = 0; i < n; i++) {
    if (pointInPolygon(positions[i * 2]!, positions[i * 2 + 1]!, data)) selected.add(i);
  }
  return selected;
}

/** The scatter's accessible name: what is plotted, coloured by what, how many cells. */
export function scatterAriaLabel(opts: {
  embedding: string | null;
  colorLabel: string | null;
  total: number;
  shown: number;
  dimensions: 2 | 3;
}): string {
  const what = opts.embedding ? prettyEmbeddingLabel(opts.embedding) : "Embedding";
  const parts = [`${what}${opts.dimensions === 3 ? " (3D)" : ""} scatter plot`];
  if (opts.colorLabel) parts.push(`coloured by ${opts.colorLabel}`);
  const total = opts.total.toLocaleString("en-US");
  parts.push(
    opts.shown < opts.total
      ? `${opts.shown.toLocaleString("en-US")} of ${total} cells shown (sampled)`
      : `${total} cells`,
  );
  return parts.join(", ");
}

/** The keys, read out with the plot (aria-describedby). */
export function scatterKeyHelp(dimensions: 2 | 3, canSelect: boolean): string {
  const move = dimensions === 3 ? "Arrow keys turn the plot" : "Arrow keys pan";
  const select = canSelect
    ? " S starts a rectangle selection: move the crosshair with the arrows (Shift for bigger steps), Enter sets each corner, Escape cancels."
    : "";
  return `${move}, plus and minus zoom, 0 fits the plot.${select}`;
}
