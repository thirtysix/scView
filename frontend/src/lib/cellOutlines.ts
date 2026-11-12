/**
 * Segmented outlines on the tissue plot (backend core/cell_outlines.py): only what the view holds
 * is turned into a mesh, once few enough cells are in view, so the whole section stays points and
 * triangulation stays small. Works from the store alone: an outline is in the units of
 * obsm["spatial"], whatever they are (microns, full-resolution pixels), and the mesh in the tissue
 * world (x, -y) of lib/spatial.ts.
 */

/** One outline per cell in obs order: an open ring; a cell with none has an empty one. */
export interface CellOutlines {
  nCells: number;
  offsets: Int32Array;
  xy: Float32Array;
}

/** Cells in view at which outlines replace points (a 1000 px plot then gives a cell about 5 px). */
export const MAX_OUTLINED_CELLS = 30_000;
/**
 * A segmentation with many vertices a cell stays points earlier: Visium HD's have 29 a cell and are
 * concave, and 739,422 of them took 135 to 285 ms to triangulate; this keeps a rebuild near 100 ms.
 */
export const MAX_OUTLINE_VERTICES = 400_000;
/** Below this many pixels across a cell its edge would only darken its colour. */
export const MIN_EDGE_PIXELS = 4;
/** The edge line drawn round each outline (1 px), on a light and on a dark background. */
export const EDGE_RGBA = {
  light: [30, 41, 59, 130],
  dark: [255, 255, 255, 110],
} as const satisfies Record<string, readonly [number, number, number, number]>;

/** Whether the outlines get an edge at `scale` screen pixels a world unit (cells wide enough). */
export function drawsEdges(mesh: { meanSize: number }, scale: number): boolean {
  return mesh.meanSize * scale >= MIN_EDGE_PIXELS;
}

/** `GET /datasets/{id}/spatial/outlines`, little-endian (no copy of the vertices). */
export function parseOutlines(buffer: ArrayBuffer): CellOutlines {
  const head = new Int32Array(buffer, 0, 2);
  const nCells = head[0]!;
  const nVertices = head[1]!;
  const expected = 8 + 4 * (nCells + 1) + 8 * nVertices;
  if (buffer.byteLength !== expected) {
    throw new Error(`The outlines answer is ${buffer.byteLength} bytes, not the ${expected} it announces.`);
  }
  return {
    nCells,
    offsets: new Int32Array(buffer, 8, nCells + 1),
    xy: new Float32Array(buffer, 8 + 4 * (nCells + 1), 2 * nVertices),
  };
}

/** What a camera shows, in world units. */
export interface WorldRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** Screen pixels per world unit */
  scale: number;
}

/** The world rectangle of a 2D orthographic camera on a plot of `width` x `height` pixels. */
export function viewRect(viewState: Record<string, unknown>, width: number, height: number): WorldRect | null {
  const target = viewState.target;
  const zoom = viewState.zoom;
  if (!Array.isArray(target) || typeof zoom !== "number" || !(width > 0) || !(height > 0)) return null;
  const scale = 2 ** zoom;
  const cx = Number(target[0]);
  const cy = Number(target[1]);
  const hw = width / 2 / scale;
  const hh = height / 2 / scale;
  return { x0: cx - hw, x1: cx + hw, y0: cy - hh, y1: cy + hh, scale };
}

/** The mesh of some cells, ready for deck.gl's binary attributes. */
export interface OutlineMesh {
  /** The cell of each polygon (polygon index -> cell index) */
  cells: Int32Array;
  /** 1 for a cell with an outline here; its point is not drawn */
  covered: Uint8Array;
  /** Polygon i is vertices startIndices[i]..startIndices[i + 1] */
  startIndices: Uint32Array;
  /** World x, y per vertex */
  positions: Float32Array;
  /** Triangles, into `positions` */
  indices: Uint32Array;
  /** Edges: x0, y0, x1, y1 each */
  edges: Float32Array;
  nVertices: number;
  /** World units across a cell, on average */
  meanSize: number;
}

export interface OutlinePlan {
  /** Cells with an outline in the rectangle */
  inView: number;
  /** Null: too many in view, they stay points */
  mesh: OutlineMesh | null;
  /** About how many cells a view may hold for this segmentation's outlines */
  limit: number;
}

let prevAt: Int32Array = new Int32Array(64);
let nextAt: Int32Array = new Int32Array(64);

/** Twice the signed area of triangle a, b, c of the vertices from `start`, by `sign`'s winding. */
function turn(xy: Float32Array, start: number, sign: number, a: number, b: number, c: number): number {
  const ax = xy[(start + a) * 2]!;
  const ay = xy[(start + a) * 2 + 1]!;
  return sign * ((xy[(start + b) * 2]! - ax) * (xy[(start + c) * 2 + 1]! - ay) - (xy[(start + b) * 2 + 1]! - ay) * (xy[(start + c) * 2]! - ax));
}

/**
 * Triangles of a simple ring by ear clipping, written to `out` from `at` (n - 2 triangles, always).
 * `xy` holds the mesh's vertices, the ring is vertices start..start + n. A ring with no reflex
 * vertex (most segmented cells) is a fan; otherwise only reflex vertices can spoil an ear, so the
 * cost grows with them. A ring that crosses itself leaves no ear at some point: the rest is a fan.
 */
export function triangulateRing(xy: Float32Array, start: number, n: number, out: Uint32Array, at: number): number {
  if (n < 3) return at;
  let area = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    area += xy[(start + j) * 2]! * xy[(start + i) * 2 + 1]! - xy[(start + i) * 2]! * xy[(start + j) * 2 + 1]!;
  }
  const sign = area >= 0 ? 1 : -1;
  if (prevAt.length < n) {
    prevAt = new Int32Array(n * 2);
    nextAt = new Int32Array(n * 2);
  }
  let reflex = 0;
  for (let i = 0; i < n; i++) {
    prevAt[i] = (i + n - 1) % n;
    nextAt[i] = (i + 1) % n;
    if (turn(xy, start, sign, prevAt[i]!, i, nextAt[i]!) < 0) reflex++;
  }
  if (reflex === 0) {
    for (let i = 1; i < n - 1; i++) {
      out[at++] = start;
      out[at++] = start + i;
      out[at++] = start + i + 1;
    }
    return at;
  }
  let left = n;
  let i = 0;
  let misses = 0;
  while (left > 3 && misses <= left) {
    const p = prevAt[i]!;
    const q = nextAt[i]!;
    let ear = turn(xy, start, sign, p, i, q) > 0;
    if (ear) {
      for (let v = nextAt[q]!; v !== p; v = nextAt[v]!) {
        if (
          turn(xy, start, sign, prevAt[v]!, v, nextAt[v]!) <= 0 &&
          turn(xy, start, sign, p, i, v) >= 0 &&
          turn(xy, start, sign, i, q, v) >= 0 &&
          turn(xy, start, sign, q, p, v) >= 0
        ) {
          ear = false;
          break;
        }
      }
    }
    if (ear) {
      out[at++] = start + p;
      out[at++] = start + i;
      out[at++] = start + q;
      nextAt[p] = q;
      prevAt[q] = p;
      left--;
      misses = 0;
    } else {
      misses++;
    }
    i = q;
  }
  // The last triangle, or what no ear could take.
  let b = nextAt[i]!;
  for (let k = 0; k < left - 2; k++) {
    const c = nextAt[b]!;
    out[at++] = start + i;
    out[at++] = start + b;
    out[at++] = start + c;
    b = c;
  }
  return at;
}

/** The mesh of `cells` (in that order), in the tissue world. */
export function buildMesh(outlines: CellOutlines, cells: Int32Array): OutlineMesh {
  const { offsets, xy, nCells } = outlines;
  let nVertices = 0;
  let nIndices = 0;
  for (let k = 0; k < cells.length; k++) {
    const n = offsets[cells[k]! + 1]! - offsets[cells[k]!]!;
    nVertices += n;
    nIndices += (n - 2) * 3;
  }
  const positions = new Float32Array(nVertices * 2);
  const startIndices = new Uint32Array(cells.length + 1);
  const indices = new Uint32Array(nIndices);
  const edges = new Float32Array(nVertices * 4);
  const covered = new Uint8Array(nCells);
  let v = 0;
  let at = 0;
  let size = 0;
  for (let k = 0; k < cells.length; k++) {
    const cell = cells[k]!;
    covered[cell] = 1;
    const from = offsets[cell]!;
    const n = offsets[cell + 1]! - from;
    startIndices[k] = v;
    let minX = Infinity;
    let maxX = -Infinity;
    for (let i = 0; i < n; i++) {
      const px = xy[(from + i) * 2]!;
      const py = -xy[(from + i) * 2 + 1]!;
      positions[(v + i) * 2] = px;
      positions[(v + i) * 2 + 1] = py;
      if (px < minX) minX = px;
      if (px > maxX) maxX = px;
      const j = (i + 1) % n;
      edges[(v + i) * 4] = px;
      edges[(v + i) * 4 + 1] = py;
      edges[(v + i) * 4 + 2] = xy[(from + j) * 2]!;
      edges[(v + i) * 4 + 3] = -xy[(from + j) * 2 + 1]!;
    }
    size += maxX - minX;
    at = triangulateRing(positions, v, n, indices, at);
    v += n;
  }
  startIndices[cells.length] = v;
  return {
    cells,
    covered,
    startIndices,
    positions,
    indices,
    edges,
    nVertices,
    meanSize: cells.length ? size / cells.length : 0,
  };
}

/**
 * Which cells a view draws as outlines, from their positions (world, interleaved x, y): those in
 * the rectangle, plus a margin when it fits, so a small pan still lands on outlines. A cell with
 * fewer than three vertices has no shape and stays a point.
 */
export function planOutlines(
  outlines: CellOutlines,
  positions: Float32Array,
  rect: WorldRect,
  maxCells = MAX_OUTLINED_CELLS,
  maxVertices = MAX_OUTLINE_VERTICES,
): OutlinePlan {
  const { offsets, nCells } = outlines;
  const n = Math.min(nCells, positions.length / 2);
  const margin = 0.25;
  const mx = (rect.x1 - rect.x0) * margin;
  const my = (rect.y1 - rect.y0) * margin;
  const wide: number[] = [];
  let inView = 0;
  let vertices = 0;
  for (let c = 0; c < n; c++) {
    const x = positions[c * 2]!;
    const y = positions[c * 2 + 1]!;
    if (!(x >= rect.x0 - mx && x <= rect.x1 + mx && y >= rect.y0 - my && y <= rect.y1 + my)) continue;
    const m = offsets[c + 1]! - offsets[c]!;
    if (m < 3) continue;
    if (x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1) {
      inView++;
      vertices += m;
    }
    wide.push(c);
  }
  const limit = inView ? Math.min(maxCells, Math.floor((maxVertices * inView) / vertices)) : maxCells;
  if (inView > maxCells || vertices > maxVertices) return { inView, mesh: null, limit };
  // The margin only when it stays in bounds: else the rectangle alone.
  const cells =
    wide.length <= maxCells
      ? Int32Array.from(wide)
      : Int32Array.from(
          wide.filter((c) => {
            const x = positions[c * 2]!;
            const y = positions[c * 2 + 1]!;
            return x >= rect.x0 && x <= rect.x1 && y >= rect.y0 && y <= rect.y1;
          }),
        );
  return { inView, mesh: buildMesh(outlines, cells), limit };
}

/** Each vertex in its cell's colour, from the RGBA per cell the points use. */
export function meshColors(mesh: OutlineMesh, cellColors: Uint8Array): Uint8Array {
  const out = new Uint8Array(mesh.nVertices * 4);
  const { cells, startIndices } = mesh;
  for (let k = 0; k < cells.length; k++) {
    const at = cells[k]! * 4;
    const r = cellColors[at]!;
    const g = cellColors[at + 1]!;
    const b = cellColors[at + 2]!;
    const a = cellColors[at + 3]!;
    for (let v = startIndices[k]!; v < startIndices[k + 1]!; v++) {
      out[v * 4] = r;
      out[v * 4 + 1] = g;
      out[v * 4 + 2] = b;
      out[v * 4 + 3] = a;
    }
  }
  return out;
}

/** The points' colours with those under an outline see-through (they stay pickable underneath). */
export function hideCovered(renderColors: Uint8Array, indexMap: ArrayLike<number>, covered: Uint8Array): Uint8Array {
  const out = renderColors.slice();
  for (let s = 0; s < indexMap.length; s++) {
    if (covered[indexMap[s]!]) out[s * 4 + 3] = 0;
  }
  return out;
}

/** The short note on the plot: what the view is showing. */
export function outlineNote(plan: OutlinePlan, kind: "cell" | "nucleus"): string {
  const word = kind === "cell" ? "cell outlines" : "nucleus outlines";
  if (plan.mesh) return `Drawing the ${word} in view (${plan.inView.toLocaleString()}).`;
  return `${plan.inView.toLocaleString()} in view: zoom in to under ${plan.limit.toLocaleString()} for the ${word}.`;
}
