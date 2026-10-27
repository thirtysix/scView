/** Which column of each dataset the two-dataset composition compares (Compare > Composition >
 *  Between datasets). Label transfer writes `<prefix>_transfer` into the query, so the columns that
 *  hold the same labels need not share a name. */
import type { ObsColumnInfo } from "@/api/types";

/** Share of labels the two columns have in common (Jaccard, case-insensitive). */
export function labelOverlap(a: ObsColumnInfo | undefined, b: ObsColumnInfo | undefined): number {
  const x = new Set((a?.values ?? []).map((v) => v.toLowerCase()));
  const y = new Set((b?.values ?? []).map((v) => v.toLowerCase()));
  if (!x.size || !y.size) return 0;
  let shared = 0;
  for (const v of x) if (y.has(v)) shared++;
  return shared / (x.size + y.size - shared);
}

/**
 * The default pair: the two columns, one per dataset, whose labels overlap most (at least a
 * quarter of them), earlier in each ranking on a tie; else the same name in both; else each
 * dataset's best-ranked column. `rankedA`/`rankedB` are best first (lib/obsColumns.ts).
 */
export function suggestColumnPair(
  rankedA: readonly string[],
  rankedB: readonly string[],
  colsA: readonly ObsColumnInfo[],
  colsB: readonly ObsColumnInfo[],
): [string, string] {
  const byA = new Map(colsA.map((c) => [c.name, c]));
  const byB = new Map(colsB.map((c) => [c.name, c]));
  let best: [string, string] | null = null;
  let bestScore = 0.25;
  rankedA.forEach((a, i) =>
    rankedB.forEach((b, j) => {
      const score = labelOverlap(byA.get(a), byB.get(b)) - 1e-3 * (i + j);
      if (score > bestScore) {
        best = [a, b];
        bestScore = score;
      }
    }),
  );
  if (best) return best;
  const same = rankedA.find((a) => rankedB.includes(a));
  if (same) return [same, same];
  return [rankedA[0] ?? "", rankedB[0] ?? ""];
}
