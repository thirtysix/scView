import { apiFetch } from "./client";
import { streamEvents } from "./clustering";
import type { WatchdogEvent } from "./pipelineRun";

/** What a WNN run uses (backend core/wnn.py WnnPlan). */
export interface WnnPlan {
  n_cells: number;
  proteins: string[];
  /** Isotype controls left out */
  controls: string[];
  n_proteins: number;
  /** The RNA representation (X_pca, or X_pca_harmony) and its PCs used */
  rep: string;
  d_rna: number;
  d_protein: number;
  /** Harmony batch column the protein PCs are corrected on, like the RNA */
  batch_key: string | null;
  /** Cells with no antibody counts at all */
  n_zero_protein: number;
  k: number;
  knn_range: number;
  resolution: number;
  exact: boolean;
  warnings: string[];
  notes: string[];
  estimate_seconds: number;
}

export interface WnnGroupRow {
  group: string;
  n_cells: number;
  mean_rna_weight: number;
  median_rna_weight: number;
}

export interface WnnInfo {
  available: boolean;
  runnable: { ok: boolean; reason: string | null; plan: WnnPlan | null };
  running: boolean;
  n_cells_total: number;
  // A saved run (available):
  run_id?: string;
  created_at?: string;
  plan?: Omit<WnnPlan, "estimate_seconds">;
  stale?: boolean;
  stale_reason?: string | null;
  n_clusters?: number;
  mean_rna_weight?: number;
  seconds?: number;
  method?: string;
  /** X_umap, X_umap_protein, X_umap_wnn: the ones the dataset has */
  embeddings?: string[];
  clusters_column?: string | null;
  by_group?: { column: string; rows: WnnGroupRow[] } | null;
  by_group_error?: string;
}

export interface WnnStage {
  stage: string;
  label: string;
  index: number;
  total: number;
  elapsed_s: number;
}

export type WnnStart = WnnPlan & { stages: { stage: string; label: string }[] };

export interface WnnWeights {
  run_id: string;
  /** RNA weight per cell, in the dataset's cell order (protein weight = 1 - it) */
  values: number[];
  violin: Record<string, number[]> | null;
  column: string | null;
  violin_error?: string;
}

export function getWnn(datasetId: string, groupby?: string): Promise<WnnInfo> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/wnn${q}`);
}

export function getWnnWeights(datasetId: string, groupby?: string): Promise<WnnWeights> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/wnn/weights${q}`);
}

export interface WnnRunBody {
  protein_pcs?: number;
  resolution?: number;
}

/**
 * Run WNN, calling the handlers as the stream arrives; resolves with the saved
 * result. Rejects with the server's reason; aborting `signal` stops the run before
 * its next stage (nothing is saved).
 */
export async function runWnn(
  datasetId: string,
  body: WnnRunBody,
  handlers: {
    onStart?: (s: WnnStart) => void;
    onStage?: (s: WnnStage) => void;
    onWatchdog?: (w: WatchdogEvent) => void;
    signal?: AbortSignal;
  } = {},
): Promise<WnnInfo> {
  let result: WnnInfo | null = null;
  await streamEvents(
    `/datasets/${datasetId}/wnn/run-stream`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as WnnStart);
      else if (event === "stage") handlers.onStage?.(data as WnnStage);
      else if (event === "watchdog") handlers.onWatchdog?.(data as WatchdogEvent);
      else if (event === "result") result = data as WnnInfo;
    },
    "The WNN run",
    handlers.signal,
  );
  if (!result) throw new Error("The WNN run ended without a result; nothing was saved.");
  return result;
}
