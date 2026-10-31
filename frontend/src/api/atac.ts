import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";
import { streamEvents } from "./clustering";
import type { WatchdogEvent } from "./pipelineRun";

/** What a run would use and need (backend core/atac.py AtacPlan.to_dict). */
export interface AtacPlan {
  n_cells: number;
  n_peaks: number;
  nnz: number;
  genome: string | null;
  n_lsi: number;
  dims_to: number;
  drop_first: "auto" | "yes" | "no";
  percentile: number;
  resolution: number;
  activity: boolean;
  estimate_seconds: number;
  estimate_bytes: number;
  estimate_disk_bytes: number;
}

export interface AtacLsi {
  run_id: string;
  created_at: string;
  sdev: number[];
  variance_ratio: number[];
  /** Pearson r of each component with the cell's total counts (Signac's DepthCor) */
  depth_r: number[];
  depth_limit: number;
  dropped_first: boolean;
  /** 1-based, as Signac writes them */
  dims_used: number[];
  n_selected: number;
  n_clusters: number;
  seconds: number;
  method: string;
  stale: boolean;
  clusters_column: string | null;
}

export interface AtacWnnPlan {
  n_cells: number;
  rep: string;
  d_rna: number;
  lsi_dims: number[];
  notes: string[];
  estimate_seconds: number;
  estimate_bytes: number;
}

export interface AtacWnn {
  run_id: string;
  n_clusters: number;
  mean_rna_weight: number;
  plan: AtacWnnPlan;
  method: string;
  stale: boolean;
  clusters_column: string | null;
  by_group?: { column: string; rows: { group: string; n_cells: number; mean_rna_weight: number; median_rna_weight: number }[] } | null;
  by_group_error?: string;
}

export interface AtacActivitySummary {
  available: boolean;
  reason?: string;
  n_genes?: number;
  n_genes_with_peaks?: number;
  n_dataset_genes?: number;
  genome?: string;
  source?: string;
  caption?: string;
}

export interface AtacInfo {
  n_peaks?: number;
  n_cells?: number;
  nnz?: number;
  genome?: string | null;
  genome_assumed?: boolean;
  source?: string;
  computed?: string[];
  metrics_files?: string[];
}

/** GET /datasets/{id}/atac */
export interface AtacState {
  available: boolean;
  reason: string | null;
  n_cells?: number;
  info?: AtacInfo;
  lsi?: AtacLsi | null;
  wnn?: AtacWnn | null;
  activity?: AtacActivitySummary | null;
  plan?: AtacPlan | null;
  plan_reason?: string | null;
  wnn_plan?: AtacWnnPlan | null;
  wnn_reason?: string | null;
  qc_columns?: string[];
  running?: "run" | "wnn" | "differential" | null;
}

export interface AtacStage {
  stage: string;
  label: string;
  index: number;
  total: number;
  elapsed_s: number;
}

export interface AtacQc {
  columns: { key: string; label: string; values: (number | null)[] }[];
  n_cells: number;
  groups: (string | null)[] | null;
  group_order: string[] | null;
  column: string | null;
}

export interface AtacActivity {
  gene: string;
  caption: string;
  n_cells: number;
  n_nonzero: number;
  /** log1p per 10k per cell, in the dataset's cell order */
  values: number[];
  violin: Record<string, number[]> | null;
  column: string | null;
  violin_error?: string;
  /** Pearson r with RNA over all cells; null when RNA is constant or the gene has no RNA */
  r: number | null;
  rna: { rna: number[]; activity: number[]; n_shown: number } | null;
}

export interface AtacDaRow {
  group: string;
  peak: string;
  z: number;
  log2fc: number;
  pct_in: number;
  pct_rest: number;
  pval: number;
  pval_adj: number;
  nearest_gene: string;
  distance_to_tss: number | null;
  peak_class: string;
}

export interface AtacDa {
  id: string;
  column: string;
  groups: string[];
  n_tested: Record<string, number>;
  n_significant: Record<string, number>;
  rows: Record<string, AtacDaRow[]>;
  method: string;
  kept_per_group: number;
  created_at: string;
  note?: string | null;
}

export type AtacRunBody = { n_lsi?: number; dims_to?: number; drop_first?: "auto" | "yes" | "no"; resolution?: number; activity?: boolean };
export type AtacWnnBody = { lsi_first?: number; lsi_last?: number; resolution?: number };
export type AtacDaBody = { column: string; groups?: string[]; min_pct?: number };

export function getAtac(datasetId: string, groupby?: string): Promise<AtacState> {
  const q = new URLSearchParams({ allow_empty: "true" });
  if (groupby) q.set("groupby", groupby);
  return apiFetch(`/datasets/${datasetId}/atac?${q.toString()}`);
}

export function getAtacQc(datasetId: string, groupby?: string): Promise<AtacQc> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/atac/qc${q}`);
}

export function getAtacActivity(datasetId: string, gene: string, groupby?: string): Promise<AtacActivity> {
  const q = new URLSearchParams({ gene });
  if (groupby) q.set("groupby", groupby);
  return apiFetch(`/datasets/${datasetId}/atac/activity?${q.toString()}`);
}

export function getLatestDa(datasetId: string): Promise<AtacDa | null> {
  return apiFetch(`/datasets/${datasetId}/atac/differential`);
}

export function getDaPage(datasetId: string, id: string, group: string, limit: number): Promise<AtacDa> {
  const q = new URLSearchParams({ group, limit: String(limit) });
  return apiFetch(`/datasets/${datasetId}/atac/differential/${id}?${q.toString()}`);
}

export function daCsvUrl(datasetId: string, id: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/differential/${id}/csv`;
}

export function atacExportUrl(datasetId: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/export`;
}

export function attachAtacMetrics(datasetId: string, file: File): Promise<{ n_cells: number; n_matched: number; file: string }> {
  const form = new FormData();
  form.append("file", file);
  return apiFetch(`/datasets/${datasetId}/atac/metrics`, { method: "POST", body: form });
}

export interface AtacStreamHandlers {
  onStage?: (s: AtacStage) => void;
  onProgress?: (message: string) => void;
  onWatchdog?: (w: WatchdogEvent) => void;
  signal?: AbortSignal;
}

async function runStream<T>(path: string, body: unknown, what: string, h: AtacStreamHandlers): Promise<T> {
  let result: T | null = null;
  await streamEvents(
    path,
    body,
    (event, data) => {
      if (event === "stage") h.onStage?.(data as AtacStage);
      else if (event === "progress") h.onProgress?.((data as { message: string }).message);
      else if (event === "watchdog") h.onWatchdog?.(data as WatchdogEvent);
      else if (event === "result") result = data as T;
    },
    what,
    h.signal,
  );
  if (!result) throw new Error(`${what} ended without a result; nothing was saved.`);
  return result;
}

/** LSI, graph, UMAP, Leiden, gene activity; aborting `signal` stops it, nothing saved. */
export function runAtac(datasetId: string, body: AtacRunBody, h: AtacStreamHandlers = {}): Promise<AtacState> {
  return runStream(`/datasets/${datasetId}/atac/run-stream`, body, "The ATAC run", h);
}

export function runAtacWnn(datasetId: string, body: AtacWnnBody, h: AtacStreamHandlers = {}): Promise<AtacState> {
  return runStream(`/datasets/${datasetId}/atac/wnn/run-stream`, body, "The WNN run", h);
}

export function runAtacDa(datasetId: string, body: AtacDaBody, h: AtacStreamHandlers = {}): Promise<AtacDa> {
  return runStream(`/datasets/${datasetId}/atac/differential`, body, "Differential accessibility", h);
}
