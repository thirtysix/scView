import { apiFetch } from "@/api/client";
import { streamEvents } from "@/api/clustering";

/** How a dataset's matrices are stored and what storing them by rows costs
 *  (GET /datasets/{id}/storage; also `mode.by_columns` of a run preflight). */
export interface StoragePlan {
  dataset_id: string;
  /** Matrices stored by columns (CSC): empty for a dataset stored by rows, or dense */
  column_stored: string[];
  /** Of those, the ones large-dataset mode reads in blocks of cells and cannot */
  blocks_large_mode: string[];
  /** Files the rewrite writes: 0 when there is nothing to do */
  files: number;
  seconds: number;
  disk_bytes: number;
  disk_free_bytes: number;
  /** Disk scView leaves free whatever the need */
  disk_floor_bytes?: number;
  enough_disk: boolean;
  memory_bytes: number;
  /** "about 30 s, 1.7 GB of disk (508 GB free)"; missing when there is nothing to do */
  text?: string;
}

export interface StoreByRowsResult {
  converted: string[];
  seconds: number;
  bytes_before: number;
  bytes_after: number;
}

export interface StoreByRowsProgress {
  done: number;
  total: number;
}

export async function getStoragePlan(datasetId: string, signal?: AbortSignal): Promise<StoragePlan> {
  return apiFetch(`/datasets/${datasetId}/storage`, { signal });
}

/** Store the dataset's column-stored matrices by rows (POST .../storage/by-rows, a stream).
 *  Only ever called from a click on the offer (components/common/StoreByRowsOffer.tsx). */
export async function storeByRows(
  datasetId: string,
  onProgress?: (p: StoreByRowsProgress) => void,
): Promise<StoreByRowsResult> {
  const out: { result?: StoreByRowsResult } = {};
  await streamEvents(
    `/datasets/${datasetId}/storage/by-rows`,
    {},
    (event, data) => {
      if (event === "progress") onProgress?.(data as StoreByRowsProgress);
      else if (event === "result") out.result = data as StoreByRowsResult;
    },
    "Store by rows",
  );
  if (!out.result) throw new Error("Store by rows: it ended without a result.");
  return out.result;
}

const GB = 1024 ** 3;

function gb(bytes: number): string {
  const v = bytes / GB;
  return v >= 10 ? v.toFixed(0) : v >= 0.1 ? v.toFixed(1) : v.toFixed(2);
}

function secondsText(seconds: number): string {
  if (seconds < 90) return `${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  return `${Math.round(seconds / 60)} min`;
}

/** The offered action's label: "Store this dataset by rows (about 30 s, 1.7 GB of disk)". */
export function storeByRowsLabel(plan: Pick<StoragePlan, "seconds" | "disk_bytes">): string {
  return `Store this dataset by rows (about ${secondsText(plan.seconds)}, ${gb(plan.disk_bytes)} GB of disk)`;
}

/** Why the rewrite cannot start, or null: the disk is short (both numbers). */
export function shortDiskText(
  plan: Pick<StoragePlan, "enough_disk" | "disk_bytes" | "disk_free_bytes" | "disk_floor_bytes">,
): string | null {
  if (plan.enough_disk) return null;
  // The copy alone would fit: it is the disk left free that does not.
  const floor =
    plan.disk_bytes <= plan.disk_free_bytes && plan.disk_floor_bytes
      ? ` (scView leaves ${(plan.disk_floor_bytes / GB).toFixed(2)} GB free)`
      : "";
  return `It writes a second copy: that needs about ${gb(plan.disk_bytes)} GB of disk and ${gb(plan.disk_free_bytes)} GB is free${floor}. Free some disk space first.`;
}
