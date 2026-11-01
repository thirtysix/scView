import { API_BASE } from "@/lib/constants";
import { parseOutlines, type CellOutlines } from "@/lib/cellOutlines";
import { parseTranscripts, type TranscriptPoints } from "@/lib/transcripts";
import { apiFetch, readError } from "./client";
import { streamEvents } from "./clustering";
import { uploadErrorText } from "./atacFragments";
import type { WatchdogEvent } from "./pipelineRun";

// --- Outlines and transcripts of a Xenium run (backend core/cell_outlines.py, core/transcripts.py) ---

export type OutlineKind = "cell" | "nucleus";

export async function fetchOutlines(datasetId: string, kind: OutlineKind, signal?: AbortSignal): Promise<CellOutlines> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/spatial/outlines?kind=${kind}`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  return parseOutlines(await res.arrayBuffer());
}

export interface TranscriptEstimate {
  rows: number;
  disk_bytes: number;
  peak_bytes: number;
  seconds: number;
  seconds_range: [number, number];
  upload_bytes?: number;
}

export interface TranscriptCheck {
  available: boolean;
  note?: string;
  n_genes?: number;
  r?: number | null;
  median_abs_diff?: number;
  max_abs_diff?: number;
  max_diff_gene?: string;
  n_equal?: number;
  total_points?: number;
  total_counts?: number;
  only_in_transcripts?: string[];
  only_in_matrix?: string[];
}

export interface TranscriptSummary {
  source: string;
  n_rows: number;
  n_kept: number;
  n_low_quality: number;
  n_control: number;
  min_qv: number;
  n_genes: number;
  n_assigned: number;
  bounds: [number, number, number, number];
  check: TranscriptCheck;
  seconds: number;
}

export interface TranscriptStatus {
  attached: boolean;
  uploaded: boolean;
  max_genes: number;
  default_limit: number;
  min_qv: number;
  summary?: TranscriptSummary;
  genes?: { name: string; n: number }[];
  disk_bytes?: number;
  upload_bytes?: number;
  estimate?: TranscriptEstimate;
  estimate_for_size?: TranscriptEstimate;
  disk_free?: number;
  running?: boolean;
}

export function getTranscriptStatus(datasetId: string, size?: number): Promise<TranscriptStatus> {
  return apiFetch(`/datasets/${datasetId}/spatial/transcripts/status${size ? `?size=${size}` : ""}`);
}

export function deleteTranscripts(datasetId: string): Promise<{ deleted: boolean }> {
  return apiFetch(`/datasets/${datasetId}/spatial/transcripts`, { method: "DELETE" });
}

function rawUpload<T>(url: string, file: File, onProgress?: (fraction: number) => void, signal?: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText) as T);
      else reject(new Error(uploadErrorText(xhr.status, xhr.responseText)));
    };
    xhr.onerror = () => reject(new Error(`The upload of ${file.name} stopped before the end: send it again.`));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

export function uploadTranscripts(
  datasetId: string,
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<TranscriptStatus> {
  const url = `${API_BASE}/datasets/${datasetId}/spatial/transcripts?name=${encodeURIComponent(file.name)}`;
  return rawUpload(url, file, onProgress, signal);
}

export interface OutlineFit {
  kind: OutlineKind;
  n_cells: number;
  n_file: number;
  n_matched: number;
  n_placed: number;
  n_vertices: number;
  outlines: OutlineKind[];
}

export function uploadOutlines(
  datasetId: string,
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<OutlineFit> {
  const url = `${API_BASE}/datasets/${datasetId}/spatial/outlines?name=${encodeURIComponent(file.name)}`;
  return rawUpload(url, file, onProgress, signal);
}

export interface TranscriptIndexEvents {
  start?: (data: { estimate: TranscriptEstimate; row_groups: number }) => void;
  progress?: (data: { phase: string; done: number; total: number; message: string }) => void;
  watchdog?: (data: WatchdogEvent) => void;
  result?: (data: TranscriptSummary) => void;
  error?: (message: string) => void;
}

/** The build (SSE). Aborting the request stops it after its row group; nothing is kept. */
export function runTranscriptIndex(datasetId: string, on: TranscriptIndexEvents, signal?: AbortSignal): Promise<void> {
  return streamEvents(
    `/datasets/${datasetId}/spatial/transcripts/index-stream`,
    {},
    (event, data) => {
      if (event === "start") on.start?.(data as never);
      else if (event === "progress") on.progress?.(data as never);
      else if (event === "watchdog") on.watchdog?.(data as WatchdogEvent);
      else if (event === "result") on.result?.(data as TranscriptSummary);
      else if (event === "error") on.error?.((data as { error?: string }).error ?? "Indexing the transcripts failed.");
    },
    "Indexing the transcripts",
    signal,
  );
}

export interface TranscriptView extends TranscriptPoints {
  total: number;
  perGene: number[];
}

/** The points of up to 8 genes in a box of API coordinates (y down). */
export async function fetchTranscripts(
  datasetId: string,
  genes: string[],
  box: { x0: number; x1: number; y0: number; y1: number } | null,
  signal?: AbortSignal,
): Promise<TranscriptView> {
  const q = new URLSearchParams({ genes: genes.join(",") });
  if (box) for (const [k, v] of Object.entries(box)) q.set(k, String(v));
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/spatial/transcripts?${q}`, { signal });
  if (!res.ok) throw new Error(await readError(res));
  const points = parseTranscripts(await res.arrayBuffer());
  return {
    ...points,
    total: Number(res.headers.get("x-transcripts-total") ?? points.n),
    perGene: (res.headers.get("x-transcripts-per-gene") ?? "").split(",").filter(Boolean).map(Number),
  };
}
