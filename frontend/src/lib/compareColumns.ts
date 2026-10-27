/** Which column of each dataset the two-dataset composition compares (Compare > Composition >
 *  Between datasets). Label transfer writes `<prefix>_transfer` into the query, so the columns that
 *  hold the same labels need not share a name. */
import type { ObsColumnInfo } from "@/api/types";
import type { RankedGrouping } from "@/lib/obsColumns";

/** Labels the two columns have in common (case-insensitive): the count and the Jaccard share.
 *  Cluster numbers ("0", "3") are left out: two clusterings share those by chance. */
export function labelOverlap(
  a: ObsColumnInfo | undefined,
  b: ObsColumnInfo | undefined,
): { shared: number; share: number } {
  const names = (c: ObsColumnInfo | undefined) =>
    new Set((c?.values ?? []).map((v) => v.trim().toLowerCase()).filter((v) => !/^[\d\s.,+-]+$/.test(v)));
  const x = names(a);
  const y = names(b);
  if (!x.size || !y.size) return { shared: 0, share: 0 };
  let shared = 0;
  for (const v of x) if (y.has(v)) shared++;
  return { shared, share: shared / (x.size + y.size - shared) };
}

// Columns whose labels name cells: what a composition is compared by.
const LABEL_KINDS = new Set(["user labels", "cell type"]);
// Fewer shared labels than this is chance ("True"/"False", "CTRL"/"STIM").
const MIN_SHARED = 3;

/**
 * The default pair: the two columns, one per dataset, whose labels overlap most (at least
 * MIN_SHARED labels and a quarter of them; true/false flags never), earlier in each ranking on
 * a tie; else a cell-type or hand-label column of the same name in both; else each dataset's
 * best-ranked column. `rankedA`/`rankedB` are best first (lib/obsColumns.ts).
 */
export function suggestColumnPair(
  rankedA: readonly RankedGrouping[],
  rankedB: readonly RankedGrouping[],
  colsA: readonly ObsColumnInfo[],
  colsB: readonly ObsColumnInfo[],
): [string, string] {
  const byA = new Map(colsA.map((c) => [c.name, c]));
  const byB = new Map(colsB.map((c) => [c.name, c]));
  const flag = (c: ObsColumnInfo | undefined) => c?.dtype === "bool";
  let best: [string, string] | null = null;
  let bestScore = 0.25;
  rankedA.forEach((a, i) =>
    rankedB.forEach((b, j) => {
      const ca = byA.get(a.name);
      const cb = byB.get(b.name);
      if (flag(ca) || flag(cb)) return;
      const { shared, share } = labelOverlap(ca, cb);
      if (shared < MIN_SHARED) return;
      const score = share - 1e-3 * (i + j);
      if (score > bestScore) {
        best = [a.name, b.name];
        bestScore = score;
      }
    }),
  );
  if (best) return best;
  const namesB = new Set(rankedB.filter((g) => LABEL_KINDS.has(g.kind)).map((g) => g.name));
  const same = rankedA.find((g) => LABEL_KINDS.has(g.kind) && namesB.has(g.name));
  if (same) return [same.name, same.name];
  return [rankedA[0]?.name ?? "", rankedB[0]?.name ?? ""];
}
