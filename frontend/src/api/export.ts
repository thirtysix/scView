import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** What GET /datasets/{id}/export/h5ad would send. */
export interface H5adInfo {
  filename: string;
  size_bytes: number;
  /** Storage layer: derived | ingested | converted | uploads */
  source: string;
  /** Plain-language version of `source`, e.g. "processed" or "as uploaded" */
  source_label: string;
  /** A large dataset: the download is put together from its files, so its size is close, not exact */
  large?: boolean;
}

export async function getH5adInfo(id: string): Promise<H5adInfo> {
  return apiFetch<H5adInfo>(`/datasets/${id}/export/h5ad/info`);
}

/** Plain link to the .h5ad: the browser streams it to disk (no multi-GB blob in memory). */
export function h5adDownloadUrl(id: string): string {
  return `${API_BASE}/datasets/${id}/export/h5ad`;
}

/** Start a browser download of a same-origin URL. */
export function downloadUrl(url: string, filename?: string): void {
  const a = document.createElement("a");
  a.href = url;
  if (filename) a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** The filename from a Content-Disposition header, or `fallback`. */
export function filenameFromDisposition(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const star = header.match(/filename\*=(?:UTF-8|utf-8)''([^;]+)/);
  if (star?.[1]) {
    try {
      return decodeURIComponent(star[1].trim());
    } catch {
      /* fall through to the plain filename */
    }
  }
  const plain = header.match(/filename="([^"]+)"/) ?? header.match(/filename=([^;]+)/);
  return plain?.[1]?.trim() || fallback;
}

export type ExportTable = "metadata" | "markers";

/**
 * Download a table built by POST /datasets/{id}/export (cell metadata, or the
 * marker genes of every grouping, or of `params.column` only). Tables are a few
 * MB at most, so going through a Blob is fine here, unlike the .h5ad.
 */
export async function downloadTable(
  id: string,
  dataType: ExportTable,
  format: "csv" | "xlsx",
  params: Record<string, unknown> = {},
): Promise<void> {
  const res = await fetch(`${API_BASE}/datasets/${id}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, data_type: dataType, params }),
  });
  if (!res.ok) {
    let detail = await readError(res);
    try {
      detail = String(JSON.parse(detail).detail ?? detail);
    } catch {
      /* plain-text error */
    }
    throw new Error(detail || `Export failed (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    downloadUrl(url, filenameFromDisposition(res.headers.get("content-disposition"), `${dataType}.${format}`));
  } finally {
    // Revoke after the click has been handled; revoking synchronously can cancel it in some browsers.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

export type RecipeFile = "script" | "manifest";

/**
 * Download the replay script (.py) or RUN_MANIFEST (.json) written from the
 * dataset's recorded steps (GET /datasets/{id}/recipe/{script|manifest}).
 */
export async function downloadRecipeFile(id: string, file: RecipeFile): Promise<void> {
  const res = await fetch(`${API_BASE}/datasets/${id}/recipe/${file}`);
  if (!res.ok) {
    let detail = await readError(res);
    try {
      detail = String(JSON.parse(detail).detail ?? detail);
    } catch {
      /* plain-text error */
    }
    throw new Error(detail || `Download failed (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const fallback = file === "script" ? "replay.py" : "RUN_MANIFEST.json";
  try {
    downloadUrl(url, filenameFromDisposition(res.headers.get("content-disposition"), fallback));
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}
