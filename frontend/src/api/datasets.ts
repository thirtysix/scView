import { apiFetch, apiUpload } from "./client";
import type { DatasetInfo } from "./types";

export async function uploadDataset(file: File) {
  return apiUpload("/datasets/upload", file);
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  return apiFetch<DatasetInfo[]>("/datasets");
}

export async function getDataset(id: string): Promise<DatasetInfo> {
  return apiFetch<DatasetInfo>(`/datasets/${id}`);
}

export async function deleteDataset(id: string): Promise<void> {
  await apiFetch(`/datasets/${id}`, { method: "DELETE" });
}

export async function pruneDatasets(): Promise<{
  removed: { id: string; name: string }[];
  count: number;
}> {
  return apiFetch("/datasets/prune", { method: "POST" });
}

/**
 * Rename a category in a categorical obs column (e.g. correct a cell-type label).
 * Persists to the derived layer; never touches the original upload. Returns once
 * the backend has rewritten + reloaded the dataset.
 */
export async function renameObsCategory(
  id: string,
  column: string,
  oldName: string,
  newName: string,
): Promise<{ column: string; old: string; new: string }> {
  return apiFetch(`/datasets/${id}/metadata/${encodeURIComponent(column)}/rename`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old: oldName, new: newName }),
  });
}
