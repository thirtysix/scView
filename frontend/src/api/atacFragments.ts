import { API_BASE } from "@/lib/constants";
import { apiFetch, BACKEND_DOWN_MESSAGE } from "./client";
import { streamEvents } from "./clustering";
import type { WatchdogEvent } from "./pipelineRun";

// --- ATAC fragments (backend core/atac_fragments.py, api/v1/atac_fragments.py) ---

export interface FragmentEstimate {
  peak_bytes: number;
  seconds: number;
  seconds_range: [number, number];
  note: string;
}

export interface FragmentQc {
  created_at: string;
  n_cells: number;
  n_cells_with_fragments: number;
  n_lines: number;
  n_fragments_in_cells: number;
  medians: { tss_enrichment: number | null; nucleosome_signal: number | null; n_fragments: number | null };
  /** Fragments of each length from 0 bp; the last entry holds longer ones */
  length_histogram: { counts: number[]; max: number; n: number };
  tss_profile: { start: number; bin: number; fold: number[]; n_tss: number; chromosomes: number };
  params: { genome: string | null; annotation: string | null; center_bp: number; flank_bp: number; file: string | null };
  seconds: number;
  notes: string[];
}

export interface FragmentStatus {
  attached: boolean;
  fragments: boolean;
  index: boolean;
  /** The sentence on what is still to upload */
  missing: string | null;
  name: string | null;
  size_bytes: number | null;
  barcode_share: number | null;
  lines_checked: number | null;
  cells_seen: number | null;
  genome: string | null;
  estimate?: FragmentEstimate;
  qc?: FragmentQc | null;
  n_cells?: number;
  max_window?: number;
  gene_pad?: number;
  running?: boolean;
}

export interface QcValuesGroup {
  name: string;
  n: number;
  n_shown: number;
  tss_enrichment: (number | null)[];
  nucleosome_signal: (number | null)[];
  n_fragments: (number | null)[];
}

export interface QcValues {
  group_by: string | null;
  groups: QcValuesGroup[];
  max_per_group: number;
}

export interface FragmentLengths {
  group_by: string | null;
  bin: number;
  x: number[];
  groups: { name: string; n_cells: number; n_fragments: number; share: number[] }[];
}

export interface CoverageTrack {
  name: string;
  n_cells: number;
  n_fragments_in_window: number;
  total_fragments: number | null;
  values: number[];
}

export interface CoverageGene {
  name: string;
  id: string;
  start: number;
  end: number;
  strand: string;
  type: string;
}

export interface Coverage {
  region: { chrom: string; start: number; end: number };
  bins: number;
  bin_size: number;
  x: number[];
  group_by: string | null;
  tracks: CoverageTrack[];
  ymax: number;
  /** Which scale the values are on, in one sentence */
  normalisation: string;
  notes: string[];
  n_fragments: number;
  max_window: number;
  peaks: { start: number; end: number }[];
  genes: CoverageGene[];
  gene: CoverageGene | null;
  genes_note: string | null;
  seconds: number;
}

export function getFragments(datasetId: string): Promise<FragmentStatus> {
  return apiFetch(`/datasets/${datasetId}/atac/fragments`);
}

export function deleteFragments(datasetId: string): Promise<{ deleted: boolean }> {
  return apiFetch(`/datasets/${datasetId}/atac/fragments`, { method: "DELETE" });
}

/** The detail sentence of an error body, or the body itself. */
export function uploadErrorText(status: number, body: string): string {
  if ([502, 503, 504].includes(status) && (!body.trim() || /^\s*<(!doctype|html)/i.test(body))) return BACKEND_DOWN_MESSAGE;
  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* not JSON */
  }
  return body || `The upload failed (HTTP ${status}).`;
}

/** One file as the raw request body: the browser reads it from disk as it sends, never whole
 *  into memory. An XMLHttpRequest, for its upload progress on a 2.4 GB file. */
export function uploadFragmentFile(
  datasetId: string,
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<FragmentStatus> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/datasets/${datasetId}/atac/fragments?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText) as FragmentStatus);
      else reject(new Error(uploadErrorText(xhr.status, xhr.responseText)));
    };
    xhr.onerror = () => reject(new Error(`The upload of ${file.name} stopped before the end: send it again.`));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

export interface FragmentQcEvents {
  start?: (data: { estimate: FragmentEstimate; n_cells: number; genome: string }) => void;
  progress?: (data: { fraction: number; message: string }) => void;
  watchdog?: (data: WatchdogEvent) => void;
  result?: (data: FragmentQc) => void;
  error?: (message: string) => void;
}

/** The QC pass (SSE). Aborting the request stops it after its block of text; nothing is saved. */
export function runFragmentQc(datasetId: string, on: FragmentQcEvents, signal?: AbortSignal): Promise<void> {
  return streamEvents(
    `/datasets/${datasetId}/atac/fragments/qc-stream`,
    {},
    (event, data) => {
      if (event === "start") on.start?.(data as never);
      else if (event === "progress") on.progress?.(data as never);
      else if (event === "watchdog") on.watchdog?.(data as WatchdogEvent);
      else if (event === "result") on.result?.(data as FragmentQc);
      else if (event === "error") on.error?.((data as { error?: string }).error ?? "The fragment QC failed.");
    },
    "The fragment QC",
    signal,
  );
}

function withGroup(path: string, groupBy: string | null | undefined): string {
  return groupBy ? `${path}?group_by=${encodeURIComponent(groupBy)}` : path;
}

export function getQcValues(datasetId: string, groupBy?: string | null): Promise<QcValues> {
  return apiFetch(withGroup(`/datasets/${datasetId}/atac/fragments/values`, groupBy));
}

export function getFragmentLengths(datasetId: string, groupBy?: string | null): Promise<FragmentLengths> {
  return apiFetch(withGroup(`/datasets/${datasetId}/atac/fragments/lengths`, groupBy));
}

export interface CoverageRequest {
  gene?: string;
  region?: string;
  groupBy?: string | null;
  groups?: string[];
  bins?: number;
}

export function coverageQuery(req: CoverageRequest): string {
  const q = new URLSearchParams();
  if (req.gene) q.set("gene", req.gene);
  if (req.region) q.set("region", req.region);
  if (req.groupBy) q.set("group_by", req.groupBy);
  for (const g of req.groups ?? []) q.append("groups", g);
  if (req.bins) q.set("bins", String(req.bins));
  return q.toString();
}

export function getCoverage(datasetId: string, req: CoverageRequest, signal?: AbortSignal): Promise<Coverage> {
  return apiFetch(`/datasets/${datasetId}/atac/coverage?${coverageQuery(req)}`, { signal });
}
