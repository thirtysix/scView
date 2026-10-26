import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** A dataset as an R object (backend api/v1/export_rds.py, converter/export_rds.R). */
export type RFormat = "seurat" | "sce";

export interface RExportInfo {
  format: RFormat;
  label: string;
  /** The R package the file needs, e.g. "SeuratObject >= 5" */
  needs: string;
  filename: string;
  n_cells: number;
  n_genes: number;
  /** Where the log-normalised values come from (null: none, e.g. not normalised yet) */
  data_from: string | null;
  counts_from: string | null;
  embeddings: string[];
  n_proteins: number;
  cluster_column: string | null;
  notes: string[];
  dropped: Record<string, string[]>;
  needs_bytes: number;
  limit_bytes: number | null;
  fits: boolean;
  too_big?: string;
  /** ok | busy (converting another file) | down (not running) */
  converter: "ok" | "busy" | "down";
  running_job: string | null;
}

export interface RExportReport {
  kept: string[];
  skipped: string[];
  renamed_features?: number;
  r_packages?: Record<string, string>;
  file_bytes?: number;
}

export interface RExportJob {
  job: string;
  format: RFormat;
  status: "writing" | "converting" | "done" | "error";
  message: string;
  filename: string;
  error?: string;
  report?: RExportReport;
  size_bytes?: number;
}

/** What the R object will hold and leave out, and whether the converter can build it. */
export function getRExportInfo(id: string, format: RFormat): Promise<RExportInfo> {
  return apiFetch<RExportInfo>(`/datasets/${id}/export/rds/info?format=${format}`);
}

/** The plain sentence of a refused request (`{"detail": ...}`), else the body. */
async function detail(res: Response): Promise<string> {
  const text = await readError(res);
  try {
    return String(JSON.parse(text).detail ?? text);
  } catch {
    return text || `Export failed (${res.status})`;
  }
}

/** Start writing the .rds on the server; the job to poll. */
export async function startRExport(id: string, format: RFormat): Promise<RExportJob> {
  const res = await fetch(`${API_BASE}/datasets/${id}/export/rds?format=${format}`, { method: "POST" });
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

export async function getRExportStatus(id: string, job: string): Promise<RExportJob> {
  const res = await fetch(`${API_BASE}/datasets/${id}/export/rds/${job}/status`);
  if (!res.ok) throw new Error(await detail(res));
  return res.json();
}

/** URL of the finished .rds (sent once, then removed on the server). */
export function rExportFileUrl(id: string, job: string): string {
  return `${API_BASE}/datasets/${id}/export/rds/${job}`;
}
