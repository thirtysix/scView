import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";
import { announceSavedScore } from "@/lib/savedScores";
import { streamEvents } from "./clustering";

/** The scan's numbers (backend core/atac_motifs.py build). */
export interface MotifScanMeta {
  genome: string;
  n_peaks: number;
  n_motifs: number;
  n_scanned: number;
  n_left_out: number;
  n_unknown_bases: number;
  p_cutoff: number;
  pseudocount: number;
  background: number[];
  n_hits: number;
  hit_rate: number;
  n_best_word_only: number;
  seconds: number;
  peak_mb?: number;
}

export interface MotifRankRow {
  motif_id: string;
  motif: string;
  variability: number;
  p: number | null;
  padj: number | null;
  n_hits: number;
  n_nan: number;
  gc_rich: boolean;
}

export interface ChromVarState {
  run_id: string;
  n_cells: number;
  n_peaks_used: number;
  n_peaks_no_counts: number;
  n_peaks_left_out: number;
  seed: number;
  iterations: number;
  w: number;
  bs: number;
  genome: string;
  seconds: number;
  peak_mb?: number;
  created_at?: string;
  method?: string;
  stale: boolean;
  ranking: MotifRankRow[];
  caveats: string;
  citation: string;
}

export interface MotifState {
  available: boolean;
  reason?: string;
  genome?: string;
  key?: string;
  scan?: {
    ready: boolean;
    running: boolean;
    progress: { stage?: string; message?: string; done?: number; total?: number } | null;
    n_peaks: number;
    n_bases: number;
    estimate_seconds: number;
    estimate_bytes: number;
    meta: MotifScanMeta | null;
    p_cutoff: number;
  };
  files?: {
    genome: { genome: string; source: string; url: string; size: number; downloaded: boolean };
    genome_auto: boolean;
    jaspar: { name: string; url: string; size: number; downloaded: boolean };
    jaspar_auto: boolean;
  };
  chromvar?: ChromVarState | null;
  chromvar_estimate?: { n_cells: number; estimate_seconds: number; estimate_bytes: number; disk_bytes: number; iterations: number };
  enrichment_defaults?: EnrichmentBody;
  caveats?: string;
  citation?: string;
  running?: string | null;
}

export interface EnrichmentBody {
  da_id?: string;
  groups?: string[];
  p_column?: "pval" | "pval_adj";
  p_max?: number;
  min_pct?: number;
  min_log2fc?: number;
  background?: number;
  seed?: number;
}

export interface EnrichmentRow {
  motif_id: string;
  motif: string;
  tfs: string;
  observed: number;
  n_query: number;
  background: number;
  n_background: number;
  pct_observed: number;
  pct_background: number;
  fold_enrichment: number | null;
  pval: number;
  pval_adj: number;
  gc: number;
  gc_rich: boolean;
  tf_gene: string;
  tf_pct: number | null;
  tf_log2fc: number | null;
  tf_expressed: boolean;
}

export interface EnrichmentGroup {
  n_query: number;
  n_left_out: number;
  n_background: number;
  n_significant?: number;
  capped: boolean;
  note: string | null;
}

export interface MotifEnrichment {
  id: string;
  da_id: string;
  column: string;
  groups: string[];
  summary: Record<string, EnrichmentGroup>;
  rows: Record<string, EnrichmentRow[]>;
  params: EnrichmentBody & { da_test?: string };
  method: string;
  genome: string;
  scan: { n_motifs: number; n_peaks: number; n_left_out: number; n_best_word_only: number; p_cutoff: number };
  caveats: string;
  citation: string;
  created_at: string;
}

export interface MotifLogo {
  motif_id: string;
  motif: string;
  length: number;
  consensus: string;
  svg: string;
}

export interface ChromVarGroupRow {
  motif_id: string;
  motif: string;
  tfs: string[];
  mean_z: number;
  mean_diff: number;
  pval: number;
  pval_adj: number;
  gc_rich: boolean;
}

export interface ChromVarGroups {
  column: string;
  groups: string[];
  n_cells: number;
  n_rows: number;
  heatmap: { motif_ids: string[]; motifs: string[]; variability: (number | null)[]; mean_z: (number | null)[][] };
  rows: Record<string, ChromVarGroupRow[]>;
}

export interface ChromVarMotif {
  motif_id: string;
  motif: string;
  n_cells: number;
  n_nan: number;
  values: (number | null)[];
  violin: Record<string, number[]> | null;
  column: string | null;
  violin_error?: string;
  /** Asked to save: the obs column (chromvar_<motif>_<id>), or null with save_note saying why */
  saved_column?: string | null;
  save_note?: string | null;
}

export function getMotifState(datasetId: string): Promise<MotifState> {
  return apiFetch(`/datasets/${datasetId}/atac/motifs`);
}

export interface ScanHandlers {
  onProgress?: (p: { stage?: string; message?: string; done?: number; total?: number }) => void;
  signal?: AbortSignal;
}

/** Starts the scan, or joins the running one. Resolves "done" or "cancelled"; a client going away leaves it running. */
export async function scanPeaks(datasetId: string, h: ScanHandlers = {}): Promise<"done" | "cancelled"> {
  let out: "done" | "cancelled" = "done";
  await streamEvents(
    `/datasets/${datasetId}/atac/motifs/scan`,
    {},
    (event, data) => {
      if (event === "progress") h.onProgress?.(data as { message?: string });
      else if (event === "cancelled") out = "cancelled";
    },
    "The motif scan",
    h.signal,
  );
  return out;
}

export function cancelScan(datasetId: string): Promise<{ cancelled: boolean }> {
  return apiFetch(`/datasets/${datasetId}/atac/motifs/scan/cancel`, { method: "POST" });
}

export function runEnrichment(datasetId: string, body: EnrichmentBody): Promise<MotifEnrichment> {
  return apiFetch(`/datasets/${datasetId}/atac/motifs/enrichment`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getLatestEnrichment(datasetId: string): Promise<MotifEnrichment | null> {
  return apiFetch(`/datasets/${datasetId}/atac/motifs/enrichment`);
}

export function getEnrichmentPage(datasetId: string, id: string, group: string, limit: number): Promise<MotifEnrichment> {
  const q = new URLSearchParams({ group, limit: String(limit) });
  return apiFetch(`/datasets/${datasetId}/atac/motifs/enrichment/${id}?${q.toString()}`);
}

export function enrichmentCsvUrl(datasetId: string, id: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/motifs/enrichment/${id}/csv`;
}

export function enrichmentExportUrl(datasetId: string, id: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/motifs/enrichment/${id}/export`;
}

export function getMotifLogo(datasetId: string, motifId: string): Promise<MotifLogo> {
  return apiFetch(`/datasets/${datasetId}/atac/motifs/logo/${encodeURIComponent(motifId)}`);
}

export function getChromVarGroups(datasetId: string, column: string): Promise<ChromVarGroups> {
  const q = new URLSearchParams({ column });
  return apiFetch(`/datasets/${datasetId}/atac/motifs/chromvar/groups?${q.toString()}`);
}

/** One motif's chromVAR z per cell. `save` (the default: the user drew it) also keeps it as
 *  an obs column (backend core/saved_scores.py); a violin regroup or a session drawing it
 *  again passes false. */
export async function getChromVarMotif(
  datasetId: string,
  motifId: string,
  groupby?: string,
  save = true,
): Promise<ChromVarMotif> {
  const q = new URLSearchParams({ motif_id: motifId });
  if (groupby) q.set("groupby", groupby);
  if (save) q.set("save", "true");
  const res = await apiFetch<ChromVarMotif>(`/datasets/${datasetId}/atac/motifs/chromvar/motif?${q.toString()}`);
  if (res.saved_column) announceSavedScore(datasetId, res.saved_column);
  return res;
}

export function chromVarCsvUrl(datasetId: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/motifs/chromvar/csv`;
}
