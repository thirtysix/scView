import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** Upstream TF candidates from promoter motifs: backend core/tf_motifs.py. */
export type Assembly = "hg38" | "mm10" | "mm39";

export interface TfMotifAssemblyStatus {
  assembly: Assembly;
  species: "human" | "mouse";
  label: string;
  prepared: boolean;
  estimate_seconds: number;
  building: boolean;
  progress: TfMotifProgress | null;
  files: { name: string; url: string; sha256: string; bytes: number; downloaded: boolean }[];
  meta?: { n_genes: number; n_motifs: number; n_unreachable: number; seconds: number; peak_mb?: number };
}

export interface TfMotifStatus {
  assemblies: TfMotifAssemblyStatus[];
  auto_download: boolean;
  citation: string;
  licence: string;
  caveats: string;
  p_threshold: number;
  promoter_bp: number;
  dataset?: { species: string; assembly: Assembly | null; prepared: boolean };
}

export interface TfMotifProgress {
  stage: "download" | "read" | "scan" | "done";
  message: string;
  done?: number;
  total?: number;
}

export interface TfExpressed {
  gene: string;
  /** Share of the group's cells expressing it. */
  pct: number;
  logfc: number | null;
  expressed: boolean;
}

export interface TfMotifRow {
  motif_id: string;
  motif: string;
  /** The TF symbols the motif's name stands for (a dimer gives two). */
  tfs: string[];
  k: number;
  n: number;
  K: number;
  N: number;
  odds_ratio: number;
  p: number;
  padj: number;
  gc: number;
  gc_rich: boolean;
  overlap: string[];
  n_overlap: number;
  tf_expressed: TfExpressed | null;
  /** null: no TF activity result, or the TF is not a CollecTRI TF. */
  activity_agrees: boolean | null;
}

export interface TfMotifGroup {
  group: string;
  n_markers: number;
  n: number;
  n_cells: number;
  n_tested: number;
  n_significant: number;
  /** Motifs with k >= 3 (the rows are the first 100 of them, lowest p first). */
  n_reported: number;
  rows: TfMotifRow[];
  note?: string;
}

export interface TfMotifResult {
  result_id: string;
  column: string;
  assembly: Assembly;
  species: string;
  n_markers_requested: number;
  N: number;
  n_motifs: number;
  n_motifs_tested: number;
  n_unreachable: number;
  groups: TfMotifGroup[];
  p_threshold: number;
  pseudocount: number;
  min_overlap: number;
  promoter_bp: number;
  activity: { available: boolean; hint: string | null; network: string };
  matrix_source: string;
  citation: string;
  licence: string;
  caveats: string;
  elapsed_seconds: number;
  created_at: string;
}

export function getTfMotifStatus(datasetId?: string): Promise<TfMotifStatus> {
  const q = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : "";
  return apiFetch<TfMotifStatus>(`/tf-motifs/status${q}`);
}

/** Download the inputs and build the motif table (SSE); resolves with the table's summary. */
export async function prepareTfMotifs(
  body: { species: "human" | "mouse"; assembly?: Assembly },
  onProgress?: (p: TfMotifProgress) => void,
  signal?: AbortSignal,
): Promise<{ assembly: Assembly; meta: TfMotifAssemblyStatus["meta"] }> {
  let result: { assembly: Assembly; meta: TfMotifAssemblyStatus["meta"] } | null = null;
  await streamEvents(
    "/tf-motifs/prepare",
    body,
    (event, data) => {
      if (event === "progress") onProgress?.(data as TfMotifProgress);
      else if (event === "result") result = data as typeof result;
    },
    "Building the motif table",
    signal,
  );
  if (!result) throw new Error("Building the motif table ended without a result.");
  return result;
}

export function runTfMotifs(datasetId: string, body: { column: string; n_markers?: number }): Promise<TfMotifResult> {
  return apiFetch<TfMotifResult>(`/datasets/${datasetId}/tf-motifs/run`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** The last run on the dataset, or null when none has run (404). */
export async function getLastTfMotifs(datasetId: string): Promise<TfMotifResult | null> {
  try {
    return await apiFetch<TfMotifResult>(`/datasets/${datasetId}/tf-motifs`);
  } catch (e) {
    if (e instanceof Error && /API error 404/.test(e.message)) return null;
    throw e;
  }
}
