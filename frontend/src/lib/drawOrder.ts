/**
 * The order the embedding scatters draw their cells in (N01).
 *
 * deck.gl paints points in buffer order, the dataset's own: cells often come
 * cluster by cluster or sample by sample, so the groups stored last cover the
 * ones stored first wherever they overlap, and a large cluster can hide a small
 * one under it. The scatters draw their cells in a seeded random order instead,
 * the same for every colouring of the same cells (a recolour does not reshuffle
 * the picture), with a selection drawn last, over the dimmed rest.
 *
 * Drawing slot `s` shows cell `order[s]`: a picked slot maps back to its cell
 * through the order. Colours, selections, labels and the lasso work on cell
 * indices and never see it.
 */

/** Seed of the draw order: a fixed value, so the same cells always come out the same. */
export const DRAW_SEED = 0x5eed;

/** mulberry32: a small seeded PRNG giving floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The cells `cells` (all 0..n-1 when a number) in a seeded random order
 *  (Fisher-Yates): the same input gives the same order. */
export function shuffledOrder(cells: Uint32Array | number, seed = DRAW_SEED): Uint32Array {
  let out: Uint32Array;
  if (typeof cells === "number") {
    out = new Uint32Array(cells);
    for (let i = 0; i < cells; i++) out[i] = i;
  } else {
    out = Uint32Array.from(cells);
  }
  const rand = mulberry32(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = out[i]!;
    out[i] = out[j]!;
    out[j] = t;
  }
  return out;
}

/** `order` with the cells in `selected` moved to the end (drawn on top), each
 *  part keeping its order. Without a selection, `order` itself. */
export function selectedLast(order: Uint32Array, selected: ReadonlySet<number> | null | undefined): Uint32Array {
  if (!selected || selected.size === 0) return order;
  // A selection can name cells that are not drawn (sampled away): count first.
  const picked = new Uint8Array(order.length);
  let inside = 0;
  for (let s = 0; s < order.length; s++) {
    if (selected.has(order[s]!)) {
      picked[s] = 1;
      inside++;
    }
  }
  if (inside === 0) return order;
  const out = new Uint32Array(order.length);
  let front = 0;
  let back = order.length - inside;
  for (let s = 0; s < order.length; s++) {
    if (picked[s]) out[back++] = order[s]!;
    else out[front++] = order[s]!;
  }
  return out;
}

/** The `stride` values of each cell in `order`, out of per-cell interleaved `values`. */
export function gatherByOrder<T extends Float32Array | Uint8Array>(values: T, order: Uint32Array, stride: number): T {
  const out = new (values.constructor as new (n: number) => T)(order.length * stride);
  for (let s = 0; s < order.length; s++) {
    const o = order[s]! * stride;
    for (let d = 0; d < stride; d++) out[s * stride + d] = values[o + d]!;
  }
  return out;
}
