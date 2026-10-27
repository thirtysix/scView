import type { LayoutRelation, LayoutSummary, LayoutThumb } from "@/api/layouts";

// Helpers for the layout stability panel (components/unified/LayoutStability.tsx).

/** One sentence for a relation: how often the two groups touch, and against the user's plot. */
export function relationSentence(r: LayoutRelation): string {
  const pair = `${r.a} and ${r.b}`;
  const of = `${r.n_touch} of ${r.n_layouts} layout${r.n_layouts === 1 ? "" : "s"}`;
  if (r.flag === "rare_in_plot") return `The contact between ${pair} in your plot appears in ${of}.`;
  if (r.flag === "hidden_in_plot") return `${pair} touch in ${of} but are apart in your plot.`;
  if (r.status === "every") return `${pair} touch in every layout.`;
  if (r.status === "never") return `${pair} touch in no layout.`;
  return `${pair} touch in ${of}.`;
}

function pct(n: number, of: number): string {
  const v = of ? (100 * n) / of : 0;
  return `${v < 10 && v > 0 ? v.toFixed(1) : Math.round(v)}%`;
}

/** The plot's contact cells as shares of each group: "8 of 3's 100 cells (8%) and 9 of 7's 80 (11%)". */
export function contactTitle(r: LayoutRelation): string | null {
  if (!r.current_contact) return null;
  const [ca, cb] = r.current_contact;
  return (
    `Contact cells in your plot: ${ca} of ${r.a}'s ${r.n_a.toLocaleString()} cells (${pct(ca, r.n_a)}) and ` +
    `${cb} of ${r.b}'s ${r.n_b.toLocaleString()} (${pct(cb, r.n_b)})`
  );
}

/** Short verdict for the table's last column. */
export function relationVerdict(r: LayoutRelation): string {
  if (r.flag === "rare_in_plot") return "in your plot, rare";
  if (r.flag === "hidden_in_plot") return "common, not in your plot";
  if (r.status === "every") return "every layout";
  if (r.status === "never") return "never";
  return "some layouts";
}

export function relationTone(r: LayoutRelation): string {
  if (r.flag === "rare_in_plot") return "text-amber-700";
  if (r.flag === "hidden_in_plot") return "text-blue-700";
  if (r.status === "every") return "text-emerald-700";
  return "text-slate-600";
}

/** The run in one sentence: pairs that touch in every, some and no layout. */
export function countsSentence(s: LayoutSummary): string {
  const c = s.counts;
  const n = s.n_done;
  return (
    `Of ${s.n_pairs.toLocaleString()} pairs of groups, ${c.every ?? 0} touch in ${n === 1 ? "the" : "every"} layout, ` +
    `${c.some ?? 0} in some and ${c.never ?? 0} in none.`
  );
}

/** What the user's plot shows that the other layouts do not (or the reverse). */
export function plotSentence(s: LayoutSummary): string | null {
  if (!s.current) return null;
  const rare = s.counts.rare_in_plot ?? 0;
  const hidden = s.counts.hidden_in_plot ?? 0;
  if (rare === 0 && hidden === 0)
    return "Every contact in your plot appears in at least half of the layouts, and none it lacks appears in more than half.";
  const parts: string[] = [];
  if (rare) parts.push(`${rare} contact${rare === 1 ? "" : "s"} in your plot appear${rare === 1 ? "s" : ""} in fewer than half of the layouts`);
  if (hidden) parts.push(`${hidden} pair${hidden === 1 ? "" : "s"} apart in your plot touch${hidden === 1 ? "es" : ""} in more than half`);
  return `${parts.join("; ")}.`;
}

/** Relations to list: all but the "never" pairs unless asked, in the server's order. */
export function visibleRelations(rows: readonly LayoutRelation[], showNever: boolean): LayoutRelation[] {
  return showNever ? [...rows] : rows.filter((r) => r.status !== "never" || r.current === true);
}

/** Pixel mapping of a thumbnail: the data's box into width x height, aspect kept, y up. */
export function fitBox(
  xs: readonly number[],
  ys: readonly number[],
  width: number,
  height: number,
  pad = 6,
): (x: number, y: number) => [number, number] {
  let x0 = Infinity,
    x1 = -Infinity,
    y0 = Infinity,
    y1 = -Infinity;
  for (let i = 0; i < xs.length; i++) {
    const x = xs[i]!,
      y = ys[i]!;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  if (!Number.isFinite(x0)) return () => [width / 2, height / 2];
  const span = Math.max(x1 - x0, y1 - y0) || 1;
  const scale = Math.min(width - 2 * pad, height - 2 * pad) / span;
  const cx = (x0 + x1) / 2,
    cy = (y0 + y1) / 2;
  return (x, y) => [width / 2 + (x - cx) * scale, height / 2 - (y - cy) * scale];
}

type RGB = readonly [number, number, number];

export const MUTED: RGB = [203, 213, 225]; // slate-300: other cells while a pair is outlined
export const HALO: RGB = [30, 41, 59]; // slate-800: the outline around the pair's cells
export const NO_GROUP: RGB = [148, 163, 184]; // slate-400: a cell without a group

function rgb(c: RGB, alpha = 1): string {
  return alpha === 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

/**
 * Draw one thumbnail: each cell in its group's colour; with `outline`, the cells
 * of those group codes on top with a dark halo and every other cell muted, so the
 * two groups of a relation stand out in every layout. Returns how many cells were
 * drawn and how many outlined.
 */
export function drawThumbnail(
  ctx: CanvasRenderingContext2D,
  opts: {
    thumb: Pick<LayoutThumb, "cells" | "x" | "y">;
    /** Group code per obs index (the colour-by codes), or null while loading */
    codes: ArrayLike<number> | null;
    colorOf: (code: number) => RGB;
    width: number;
    height: number;
    outline?: ReadonlySet<number> | null;
    pointSize?: number;
    /** Canvas fill: the scatter background (#1e293b on a dark one, as EmbeddingScatter) */
    background?: string;
  },
): { drawn: number; outlined: number } {
  const { thumb, codes, colorOf, width, height, outline } = opts;
  const size = opts.pointSize ?? 2;
  const half = size / 2;
  const at = fitBox(thumb.x, thumb.y, width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = opts.background ?? "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const codeOf = (i: number) => (codes ? Number(codes[thumb.cells[i]!] ?? -1) : -1);
  const n = thumb.cells.length;
  const on = (code: number) => !!outline && outline.has(code);
  let outlined = 0;
  // Pass 1: every cell not outlined.
  for (let i = 0; i < n; i++) {
    const code = codeOf(i);
    if (on(code)) continue;
    ctx.fillStyle = outline ? rgb(MUTED, 0.6) : rgb(code >= 0 ? colorOf(code) : NO_GROUP, 0.85);
    const [px, py] = at(thumb.x[i]!, thumb.y[i]!);
    ctx.fillRect(px - half, py - half, size, size);
  }
  if (outline && outline.size) {
    // Pass 2: the halo, then pass 3: the outlined cells in their colours.
    ctx.fillStyle = rgb(HALO);
    for (let i = 0; i < n; i++) {
      if (!on(codeOf(i))) continue;
      const [px, py] = at(thumb.x[i]!, thumb.y[i]!);
      ctx.fillRect(px - half - 1, py - half - 1, size + 2, size + 2);
    }
    for (let i = 0; i < n; i++) {
      const code = codeOf(i);
      if (!on(code)) continue;
      ctx.fillStyle = rgb(colorOf(code));
      const [px, py] = at(thumb.x[i]!, thumb.y[i]!);
      ctx.fillRect(px - half, py - half, size, size);
      outlined++;
    }
  }
  return { drawn: n, outlined };
}

/** "seed 1 · 10 neighbours" (or "Your plot") for a thumbnail's label. */
export function shortLayoutLabel(seed: number | null, k: number | null): string {
  return seed == null ? "Your plot" : `seed ${seed} · ${k} neighbours`;
}
