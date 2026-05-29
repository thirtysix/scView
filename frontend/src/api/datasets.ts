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
