/**
 * The Communication tab's data shaping: the result's column table as rows, the
 * filters and sort, the dot plot (sender -> receiver pairs x ligand-receptor
 * pairs), the circle plot of interactions between groups and the sender /
 * receiver roles. Pure functions, so they are tested without a browser.
 */
import type { CommunicationResult, RowNum, RowText } from "@/api/communication";

export type CommRow = Record<RowText, string> & Record<RowNum, number | null>;

const TEXT: RowText[] = ["source", "target", "ligand_complex", "receptor_complex", "ligand", "receptor"];
const NUM: RowNum[] = [
  "magnitude_rank", "specificity_rank", "lr_means", "cellphone_pvals", "expr_prod",
  "scaled_weight", "lr_logfc", "spec_weight", "lrscore", "ligand_props", "receptor_props",
];

export function toRows(res: CommunicationResult): CommRow[] {
  const n = res.columns.source.length;
  const rows: CommRow[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = {} as CommRow;
    for (const c of TEXT) r[c] = res.columns[c][i] ?? "";
    for (const c of NUM) r[c] = res.columns[c][i] ?? null;
    rows[i] = r;
  }
  return rows;
}

/** "ligand -> receptor" with the complexes as LIANA writes them (A_B). */
export const lrLabel = (r: Pick<CommRow, "ligand_complex" | "receptor_complex">) =>
  `${r.ligand_complex} → ${r.receptor_complex}`;
export const pairLabel = (r: Pick<CommRow, "source" | "target">) => `${r.source} → ${r.target}`;

export interface RowFilter {
  source: string; // "" = any
  target: string;
  query: string; // matched in the ligand or receptor (complex) name, ignoring case
}

export function filterRows(rows: CommRow[], f: RowFilter): CommRow[] {
  const q = f.query.trim().toUpperCase();
  return rows.filter(
    (r) =>
      (!f.source || r.source === f.source) &&
      (!f.target || r.target === f.target) &&
      (!q || r.ligand_complex.toUpperCase().includes(q) || r.receptor_complex.toUpperCase().includes(q)),
  );
}

export type SortKey = RowText | RowNum;

/** Sorted copy; nulls last either way. */
export function sortRows(rows: CommRow[], key: SortKey, ascending: boolean): CommRow[] {
  const dir = ascending ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = a[key];
    const y = b[key];
    if (x === null || x === undefined) return y === null || y === undefined ? 0 : 1;
    if (y === null || y === undefined) return -1;
    if (typeof x === "string" || typeof y === "string") return dir * String(x).localeCompare(String(y));
    return dir * (x - y);
  });
}

/** -log10 of a rank or p-value, finite (a 0 counts as 1e-300). */
export const negLog10 = (v: number | null) => (v === null ? 0 : -Math.log10(Math.max(v, 1e-300)));

export interface DotPlot {
  x: string[]; // sender -> receiver pairs
  y: string[]; // ligand -> receptor pairs, strongest at the top
  points: { x: string; y: string; size: number; color: number; row: CommRow }[];
  maxSize: number;
}

/**
 * The `topN` ligand-receptor pairs with the best magnitude rank among `rows`,
 * against the sender -> receiver pairs they occur in (at most `maxPairs`, those
 * with the best rank first). Size = -log10 specificity rank, colour = -log10
 * magnitude rank, as LIANA's dot plot.
 */
export function dotPlotData(rows: CommRow[], topN = 15, maxPairs = 20): DotPlot {
  const best = new Map<string, number>();
  for (const r of rows) {
    const k = lrLabel(r);
    const v = r.magnitude_rank ?? 1;
    if (!best.has(k) || v < best.get(k)!) best.set(k, v);
  }
  const lr = [...best.entries()].sort((a, b) => a[1] - b[1]).slice(0, topN).map(([k]) => k);
  const keepLr = new Set(lr);
  const pairBest = new Map<string, number>();
  for (const r of rows) {
    if (!keepLr.has(lrLabel(r))) continue;
    const k = pairLabel(r);
    const v = r.magnitude_rank ?? 1;
    if (!pairBest.has(k) || v < pairBest.get(k)!) pairBest.set(k, v);
  }
  const pairs = [...pairBest.entries()].sort((a, b) => a[1] - b[1]).slice(0, maxPairs).map(([k]) => k);
  const keepPairs = new Set(pairs);
  const points = rows
    .filter((r) => keepLr.has(lrLabel(r)) && keepPairs.has(pairLabel(r)))
    .map((r) => ({ x: pairLabel(r), y: lrLabel(r), size: negLog10(r.specificity_rank),
                   color: negLog10(r.magnitude_rank), row: r }));
  const maxSize = points.reduce((m, p) => Math.max(m, p.size), 0);
  // Plotly draws category y from the bottom: reverse so the strongest is on top.
  return { x: [...pairs].sort(), y: [...lr].reverse(), points, maxSize };
}

export type Weight = "count" | "strength";

export function pairMatrix(res: CommunicationResult, weight: Weight): number[][] {
  return weight === "count" ? res.pair_counts : res.pair_strength.map((row) => row.map((v) => v ?? 0));
}

export interface CircleEdge {
  source: number;
  target: number;
  value: number;
  width: number; // px
  x: number[]; // the curve, sampled
  y: number[];
}

export interface CircleLayout {
  nodes: { x: number; y: number; group: string; autocrine: number; out: number; in: number }[];
  edges: CircleEdge[];
}

/**
 * Groups on a circle and one curved edge per sender -> receiver pair with any
 * weight, bent towards the centre (a quadratic Bezier through a point pulled
 * in), so A -> B and B -> A are told apart by which side they bend to. Width is
 * proportional to the weight, up to `maxWidth`; edges under `minShare` of the
 * largest are left out. Autocrine weights are reported per node, not drawn.
 */
export function circleLayout(groups: string[], m: number[][], maxWidth = 8, minShare = 0.02): CircleLayout {
  const n = groups.length;
  const nodes = groups.map((group, i) => {
    const a = Math.PI / 2 - (2 * Math.PI * i) / n;
    const row = m[i] ?? [];
    return {
      x: Math.cos(a),
      y: Math.sin(a),
      group,
      autocrine: row[i] ?? 0,
      out: row.reduce((s, v, j) => s + (j === i ? 0 : v), 0),
      in: m.reduce((s, r, j) => s + (j === i ? 0 : r[i] ?? 0), 0),
    };
  });
  let max = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) max = Math.max(max, m[i]?.[j] ?? 0);
  const edges: CircleEdge[] = [];
  if (max <= 0) return { nodes, edges };
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = m[i]?.[j] ?? 0;
      if (i === j || v <= 0 || v < minShare * max) continue;
      const a = nodes[i]!;
      const b = nodes[j]!;
      // Control point: the chord's midpoint pulled towards the centre and to the
      // left of the direction of travel.
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const cx = mx * 0.35 - dy * 0.18;
      const cy = my * 0.35 + dx * 0.18;
      const xs: number[] = [];
      const ys: number[] = [];
      for (let k = 0; k <= 16; k++) {
        const t = k / 16;
        const u = 1 - t;
        xs.push(u * u * a.x * 0.92 + 2 * u * t * cx + t * t * b.x * 0.92);
        ys.push(u * u * a.y * 0.92 + 2 * u * t * cy + t * t * b.y * 0.92);
      }
      edges.push({ source: i, target: j, value: v, width: Math.max(0.5, (v / max) * maxWidth), x: xs, y: ys });
    }
  }
  // Thin edges first, so the strong ones are drawn on top.
  edges.sort((p, q) => p.value - q.value);
  return { nodes, edges };
}

/** Per group: outgoing (as sender) and incoming (as receiver) weight, autocrine on both sides. */
export function roleBars(res: CommunicationResult, weight: Weight): { group: string; out: number; in: number }[] {
  const out = weight === "count" ? res.roles.outgoing : res.roles.outgoing_strength.map((v) => v ?? 0);
  const inc = weight === "count" ? res.roles.incoming : res.roles.incoming_strength.map((v) => v ?? 0);
  return res.groups.map((group, i) => ({ group, out: out[i] ?? 0, in: inc[i] ?? 0 }));
}
