import { API_BASE } from "@/lib/constants";

import { apiFetch, readError } from "./client";

/** Backend api/v1/public_import.py: datasets from CELLxGENE Discover, pinned to a version. */

export interface PublicImportStatus {
  enabled: boolean;
  source: string;
  api: string;
  files: string;
  license: string;
  license_url: string;
}

export interface DiscoverDataset {
  dataset_id: string;
  dataset_version_id: string;
  collection_id: string;
  collection_name: string | null;
  collection_doi: string | null;
  collection_doi_label: string | null;
  title: string;
  citation: string;
  organism: string[];
  tissue: string[];
  disease: string[];
  assay: string[];
  n_cell_types: number;
  suspension_type: string[];
  cell_count: number;
  size_bytes: number;
  schema_version: string | null;
  published_at: string | null;
  explorer_url: string | null;
  license: string;
}

export interface DiscoverDetail extends DiscoverDataset {
  download_url: string;
  license_url: string;
  disk: { free_bytes: number | null; need_bytes: number; ok: boolean };
}

export interface DiscoverSearch {
  total: number;
  results: DiscoverDataset[];
  facets: { organism: [string, number][]; tissue: [string, number][] };
  fetched_at: string;
  n_datasets: number;
}

export interface DiscoverQuery {
  q?: string;
  organism?: string;
  tissue?: string;
  max_mb?: number;
  sort?: "cells" | "size" | "title" | "newest";
  limit?: number;
  offset?: number;
}

/** Where an imported dataset came from (metadata public_source; uns about_*). */
export interface PublicSource {
  about_title: string;
  about_source: string;
  about_citation: string;
  about_publication: string;
  about_doi: string;
  about_collection: string;
  about_collection_id: string;
  about_dataset_id: string;
  about_dataset_version_id: string;
  about_schema_version: string;
  about_license: string;
  about_url: string;
  about_downloaded_at: string;
  about_sha256: string;
  about_size_bytes: string;
  citation_text: string;
  author_annotation?: string | null;
}

export function getPublicImportStatus(): Promise<PublicImportStatus> {
  return apiFetch("/public-import/status");
}

export function searchDiscover(query: DiscoverQuery): Promise<DiscoverSearch> {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  return apiFetch(`/public-import/discover/datasets?${params.toString()}`);
}

export function getDiscoverDataset(versionId: string): Promise<DiscoverDetail> {
  return apiFetch(`/public-import/discover/datasets/${encodeURIComponent(versionId)}`);
}

export function cancelPublicImport(jobId: string): Promise<{ cancelled: boolean }> {
  return apiFetch(`/public-import/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
}

export interface ImportEvent {
  event: string;
  data: Record<string, unknown>;
}

/** The detail of an error body (FastAPI's {"detail": ...}), else the text. */
function detailOf(text: string): string {
  try {
    const d = (JSON.parse(text) as { detail?: unknown }).detail;
    return typeof d === "string" ? d : text;
  } catch {
    return text;
  }
}

/**
 * Download one Discover dataset version into the library, calling `onEvent` for
 * each SSE event (job, stage, progress, memory). Resolves with the new dataset id;
 * rejects on `error` or `cancelled` (message from the server).
 */
export async function importDiscoverDataset(
  versionId: string,
  onEvent: (e: ImportEvent) => void,
  name?: string,
): Promise<string> {
  const res = await fetch(`${API_BASE}/public-import/discover/import-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_version_id: versionId, name: name || null }),
  });
  if (!res.ok) throw new Error(detailOf(await readError(res)));
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let datasetId: string | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ") && event) {
        const data = JSON.parse(line.slice(6)) as Record<string, unknown>;
        if (event === "error") throw new Error(String(data.error));
        if (event === "cancelled") throw new Error(String(data.message ?? "Import cancelled."));
        if (event === "complete") datasetId = String(data.dataset_id);
        onEvent({ event, data });
        event = "";
      }
    }
  }
  if (!datasetId) throw new Error("The import ended without completing.");
  return datasetId;
}

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}
