/**
 * Hierarchical clustering of gene profiles across groups, for the marker
 * heatmap's "Cluster genes" (C034): correlation distance (1 - Pearson r) and
 * average linkage (UPGMA), done here because the profiles are already in the
 * browser (at most a few hundred genes x groups).
 *
 * Deterministic: the closest pair merges first, ties going to the pair of
 * clusters listed first (by their first gene in the input order). The tree
 * comes out as scipy-style linkage rows (`LinkageRow` in lib/dendrogram.ts), with
 * the child holding the earlier input gene first, so equal distances keep the
 * input order and `leafOrder` reads the tree left to right.
 */
import type { LinkageRow } from "@/lib/dendrogram";

/** 1 - Pearson correlation of two profiles, in 0..2. A flat profile has no
 *  correlation with anything: distance 1 (as r = 0). */
export function correlationDistance(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 1;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i]!;
    mb += b[i]!;
  }
  ma /= n;
  mb /= n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i]! - ma;
    const db = b[i]! - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  if (!(saa > 0) || !(sbb > 0)) return 1;
  const r = Math.max(-1, Math.min(1, sab / Math.sqrt(saa * sbb)));
  return 1 - r;
}

/** Average-linkage tree of n items from their pairwise distances
 *  (`dist[i][j]`, symmetric). Rows as scipy's: [node a, node b, height, size];
 *  node ids below n are items, n + k the node made by row k. */
export function averageLinkage(dist: readonly (readonly number[])[]): LinkageRow[] {
  const n = dist.length;
  // Active clusters in a fixed order: by their first item.
  let active: { id: number; first: number; size: number }[] = Array.from({ length: n }, (_, i) => ({
    id: i,
    first: i,
    size: 1,
  }));
  const d = new Map<number, Map<number, number>>();
  const get = (a: number, b: number) => d.get(a)!.get(b)!;
  const set = (a: number, b: number, v: number) => {
    d.get(a)!.set(b, v);
    d.get(b)!.set(a, v);
  };
  for (let i = 0; i < n; i++) d.set(i, new Map());
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) set(i, j, dist[i]![j]!);

  const rows: LinkageRow[] = [];
  while (active.length > 1) {
    let best = Infinity;
    let bi = 0;
    let bj = 1;
    for (let i = 0; i < active.length; i++) {
      for (let j = i + 1; j < active.length; j++) {
        const v = get(active[i]!.id, active[j]!.id);
        if (v < best) {
          best = v;
          bi = i;
          bj = j;
        }
      }
    }
    const a = active[bi]!;
    const b = active[bj]!; // a lists earlier: its first item is the earlier one
    const id = n + rows.length;
    const size = a.size + b.size;
    rows.push([a.id, b.id, best, size]);
    d.set(id, new Map());
    for (const c of active) {
      if (c === a || c === b) continue;
      // Lance-Williams update for average linkage: the size-weighted mean.
      set(id, c.id, (a.size * get(a.id, c.id) + b.size * get(b.id, c.id)) / size);
    }
    const merged = { id, first: a.first, size };
    active = active.filter((c) => c !== a && c !== b);
    const at = active.findIndex((c) => c.first > merged.first);
    active.splice(at < 0 ? active.length : at, 0, merged);
  }
  return rows;
}

/** Leaves of a linkage tree left to right (scipy's leaves_list). */
export function leafOrder(rows: readonly LinkageRow[], n: number): number[] {
  if (n === 0) return [];
  if (rows.length === 0) return n === 1 ? [0] : [];
  const out: number[] = [];
  const stack = [n + rows.length - 1];
  while (stack.length) {
    const id = stack.pop()!;
    if (id < n) {
      out.push(id);
      continue;
    }
    const [a, b] = rows[id - n]!;
    stack.push(b, a); // a first
  }
  return out;
}

/** Cluster genes by their profiles across groups (`profiles[gene][group]`):
 *  the linkage rows and the gene order they give. */
export function clusterGenes(profiles: readonly (readonly number[])[]): { tree: LinkageRow[]; order: number[] } {
  const n = profiles.length;
  const dist = profiles.map((a, i) => profiles.map((b, j) => (i === j ? 0 : correlationDistance(a, b))));
  const tree = averageLinkage(dist);
  return { tree, order: leafOrder(tree, n) };
}
