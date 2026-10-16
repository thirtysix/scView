import type { ObsColumnInfo } from "@/api/types";

/** Pseudotime computed in-app is stored in this obs column (core/trajectory.py). */
export const DPT_COLUMN = "scview_dpt_pseudotime";

const CELL_TYPE_RE = /cell_?type|celltypeanno|annotation|majority_voting|predicted_labels/i;

/** Columns a root group can be picked from: categorical, 2..100 listed values. */
export function rootGroupColumns(cols: ObsColumnInfo[] | undefined): ObsColumnInfo[] {
  return (cols ?? []).filter(
    (c) => (c.values?.length ?? 0) >= 2 && (c.values?.length ?? 0) <= 100 && !/bool/i.test(c.dtype),
  );
}

/** Default root-group column: a cell-type annotation when there is one (a start
 *  state is usually named), else the active clustering, else the first column. */
export function defaultRootColumn(cols: ObsColumnInfo[], activeClustering?: string | null): string {
  return (
    cols.find((c) => CELL_TYPE_RE.test(c.name))?.name ??
    cols.find((c) => c.name === activeClustering)?.name ??
    cols[0]?.name ??
    ""
  );
}
