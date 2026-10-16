import type { ObsColumnInfo } from "@/api/types";

/** Most-specific first: a donor column beats a per-sample one, which may also
 *  encode the condition (sample = donor x condition still works, unpaired). */
const REPLICATE_PATTERNS: RegExp[] = [
  /donor/i,
  /patient/i,
  /individual/i,
  /^ind$/i,
  /subject/i,
  /participant/i,
  /replicate/i,
  /sample/i,
  /orig\.ident/i,
];

/** The backend refuses more pseudobulk samples than this (core/pseudobulk.py). */
const MAX_REPLICATE_LEVELS = 300;

/** Columns that could hold donors/samples: 2..300 distinct values, excluding the
 *  columns already used to split or scope the comparison. */
export function replicateColumns(
  cols: ObsColumnInfo[] | undefined,
  exclude: string[],
): ObsColumnInfo[] {
  return (cols ?? []).filter(
    (c) =>
      !exclude.includes(c.name) &&
      c.n_unique >= 2 &&
      c.n_unique <= MAX_REPLICATE_LEVELS &&
      !/float|bool/i.test(c.dtype),
  );
}

/** Pre-select the most donor-like column by name, or "" when none looks like one. */
export function suggestReplicateColumn(cols: ObsColumnInfo[]): string {
  for (const re of REPLICATE_PATTERNS) {
    const hit = cols.find((c) => re.test(c.name));
    if (hit) return hit.name;
  }
  return "";
}
