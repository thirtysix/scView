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

/** Rename a dataset (display name only). A rejected name throws with the server's reason. */
export async function renameDataset(id: string, name: string): Promise<{ id: string; name: string }> {
  try {
    return await apiFetch(`/datasets/${id}`, { method: "PATCH", body: JSON.stringify({ name }) });
  } catch (e) {
    const body = e instanceof Error ? e.message.replace(/^API error \d+: /, "") : "";
    let detail = "";
    try {
      detail = String(JSON.parse(body).detail ?? "");
    } catch {
      /* not JSON: keep the original error */
    }
    throw detail ? new Error(detail) : e;
  }
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
