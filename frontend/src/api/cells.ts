import { apiFetch } from "./client";
import { decodeMask, type CellFilter } from "@/lib/cellFilter";
import { errorMessage } from "@/lib/errorMessage";

export interface CellFilterResult {
  n_cells: number;
  n_matching: number;
  /** Packed bitmask (base64), when asked for */
  mask: string | null;
}

/** The server's reason for a refused request (FastAPI `detail`, a 422's messages). */
function reason(e: unknown): Error {
  return new Error(errorMessage(e));
}

/** How many cells match `filter` and, with `mask`, which (`POST /cells/filter`). */
export async function filterCells(
  datasetId: string,
  filter: CellFilter,
  mask: boolean,
  signal?: AbortSignal,
): Promise<CellFilterResult> {
  try {
    return await apiFetch<CellFilterResult>(`/datasets/${datasetId}/cells/filter`, {
      method: "POST",
      body: JSON.stringify({ filter, mask }),
      signal,
    });
  } catch (e) {
    throw reason(e);
  }
}

/** The cells matching `filter`, as cell indices. */
export async function cellsMatching(datasetId: string, filter: CellFilter): Promise<Set<number>> {
  const res = await filterCells(datasetId, filter, true);
  return res.mask ? decodeMask(res.mask, res.n_cells) : new Set();
}
