import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** Exports for other viewers (backend api/v1/export_formats.py). */
export type ExportFormat = "cellxgene" | "vitessce";

export interface ExportFormatOptions {
  /** obsm keys; undefined = every 2D/3D layout */
  embeddings?: string[];
  /** obs columns; undefined = all (CELLxGENE) or the label columns (Vitessce) */
  columns?: string[];
  /** CELLxGENE: raw counts as .raw */
  include_raw?: boolean;
  /** Vitessce: all genes or the variable genes only */
  genes?: "all" | "variable";
}

export interface EmbeddingChoice {
  key: string;
  label: string;
  n_dims: number;
  default: boolean;
  export_name: string;
}

export interface ReadinessItem {
  key: string;
  label: string;
  status: "ok" | "missing" | "invalid" | "warn";
  detail: string;
}

export interface Readiness {
  schema_version: string;
  schema_url: string;
  items: ReadinessItem[];
  n_ok: number;
  n_items: number;
  ready: boolean;
  hints: string[];
}

export interface ExportFormatInfo {
  format: ExportFormat;
  filename: string;
  size_bytes: number;
  large: boolean;
  n_cells: number;
  n_genes: number;
  embeddings: EmbeddingChoice[];
  label_columns: { name: string; n_values: number; default: boolean }[];
  has_variable_genes: boolean;
  has_counts?: boolean;
  x_from: string;
  raw_from?: string | null;
  obs_columns: string[];
  exported_embeddings: string[];
  dropped: Record<string, string[]>;
  notes: string[];
  renamed?: { cells: number; genes: number };
  readiness?: Readiness;
}

/** What an export with these options holds, its size and (CELLxGENE) Discover readiness. */
export function getExportFormatInfo(
  id: string,
  format: ExportFormat,
  opts: ExportFormatOptions = {},
): Promise<ExportFormatInfo> {
  const q = new URLSearchParams({ format });
  opts.embeddings?.forEach((e) => q.append("embedding", e));
  opts.columns?.forEach((c) => q.append("column", c));
  if (opts.include_raw) q.set("raw", "true");
  if (opts.genes) q.set("genes", opts.genes);
  return apiFetch<ExportFormatInfo>(`/datasets/${id}/export/h5ad/info?${q.toString()}`);
}

export interface BuiltExport {
  file: string;
  filename: string;
  size_bytes: number;
}

/** URL of a finished export (streamed from disk by the browser). */
export function exportFileUrl(id: string, file: string): string {
  return `${API_BASE}/datasets/${id}/export/files/${encodeURIComponent(file)}`;
}

/**
 * Write the export on the server, reporting each step to `onProgress`; resolves with
 * the file to download. Aborting `signal` stops the write.
 */
export async function buildExport(
  id: string,
  format: ExportFormat,
  opts: ExportFormatOptions,
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<BuiltExport> {
  const res = await fetch(`${API_BASE}/datasets/${id}/export/${format}/build-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
    signal,
  });
  if (!res.ok) {
    let detail = await readError(res);
    try {
      detail = String(JSON.parse(detail).detail ?? detail);
    } catch {
      /* plain text */
    }
    throw new Error(detail || `Export failed (${res.status})`);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ") && event) {
        const data = JSON.parse(line.slice(6));
        if (event === "error") throw new Error(String(data.error));
        if (event === "complete") return data as BuiltExport;
        if (event === "progress") onProgress?.(String(data.message));
        event = "";
      }
    }
  }
  throw new Error("The export stopped before it finished.");
}
