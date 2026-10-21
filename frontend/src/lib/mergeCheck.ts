/** Helpers for the merge check (components/unified/MergeCheck). */
import type { MergePair } from "@/api/clustering";

export const pairKey = (p: { a: string; b: string }) => `${p.a}\u0000${p.b}`;

/** The pairs ticked at first: the merge candidates. */
export function defaultSelection(pairs: readonly MergePair[]): Set<string> {
  return new Set(pairs.filter((p) => p.candidate).map(pairKey));
}

/** How many clusters remain after merging `pairs` (chains join: 1+2 and 2+3 make one). */
export function clustersAfter(nClusters: number, pairs: readonly { a: string; b: string }[]): number {
  const parent = new Map<string, string>();
  const find = (x: string): string => {
    let r = x;
    while (parent.has(r) && parent.get(r) !== r) r = parent.get(r)!;
    parent.set(x, r);
    return r;
  };
  let joined = 0;
  for (const { a, b } of pairs) {
    if (!parent.has(a)) parent.set(a, a);
    if (!parent.has(b)) parent.set(b, b);
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) {
      parent.set(rb, ra);
      joined += 1;
    }
  }
  return nClusters - joined;
}

/** The verdict in words, for a table cell's tooltip. */
export function describePair(p: MergePair, maxGenes: number): string {
  if (p.evidence === "together") {
    return "One training cluster holds most of both: the training half never split them, so no held-out test compares them.";
  }
  if (p.evidence === "untested") {
    return "No training cluster stands for one of them clearly enough to test this pair.";
  }
  const genes = `${p.n_genes} gene${p.n_genes === 1 ? "" : "s"} distinguish them on the held-out half`;
  const via = p.train_pairs.length ? ` (training pairs ${p.train_pairs.join(", ")})` : "";
  return p.candidate
    ? `${genes}${via}: fewer than ${maxGenes}, a merge candidate.`
    : `${genes}${via}: kept apart.`;
}
