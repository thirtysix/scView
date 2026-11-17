import { apiFetch } from "./client";
import type { CellFilter } from "@/lib/cellFilter";
import { errorMessage } from "@/lib/errorMessage";

/** How the cells are chosen: exactly one of these (backend core/subset.py SubsetRequest). */
export type SubsetSelector =
  | { filter: CellFilter }
  | { groups: { column: string; values: string[] } }
  /** Packed base64 bitmask over the dataset's cells (lib/session encodeMask) */
  | { mask: string };

export interface SubsetResult {
  /** The new dataset's id; null for a dry run */
  dataset_id: string | null;
  name: string;
  n_cells: number;
  n_parent_cells: number;
  n_genes?: number | null;
  described: string;
  counts_source?: string | null;
  carried: string[];
  not_carried: string[];
  renamed: Record<string, string>;
  dry_run: boolean;
  /** The run the new dataset needs; filtering and doublet detection left out on cells the parent filtered */
  steps?: string[];
  steps_note?: string | null;
  next: string;
}

/** A new dataset of some of a dataset's cells (`POST /datasets/{id}/subset`); `dryRun`
 *  says what it would hold and makes nothing. Errors carry the server's sentence. */
export async function makeSubset(
  datasetId: string,
  selector: SubsetSelector,
  opts: { name?: string; dryRun?: boolean; signal?: AbortSignal } = {},
): Promise<SubsetResult> {
  try {
    return await apiFetch<SubsetResult>(`/datasets/${datasetId}/subset`, {
      method: "POST",
      body: JSON.stringify({ ...selector, ...(opts.name ? { name: opts.name } : {}), dry_run: !!opts.dryRun }),
      signal: opts.signal,
    });
  } catch (e) {
    throw new Error(errorMessage(e));
  }
}
