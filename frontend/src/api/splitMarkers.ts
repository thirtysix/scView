import { apiFetch } from "@/api/client";
import { streamEvents, type HvgRecorded } from "@/api/clustering";

/** Count-split markers (backend core/split_markers.py): marker p-values on held-out counts. */

export interface SplitMarkersInfo {
  available: boolean;
  reason: string | null;
  n_cells: number;
  first_run_seconds: number;
  reference?: string;
  n_groups?: number;
  recorded?: HvgRecorded;
  /** The last run on this column is kept: no waiting */
  cached?: boolean;
  estimate_seconds?: number;
}

export interface SplitMarkersStart {
  n_cells: number;
  reference: string;
  estimate_seconds: number;
  cached: boolean;
}

export interface SplitMarkersProgress {
  step: string;
  label: string;
}

/** One gene of a group: the ordinary test (all counts, the group's own cells)
 *  and the split test (its training cluster vs the rest, on the held-out half). */
export interface SplitGene {
  gene: string;
  logfoldchange: number | null;
  pval: number | null;
  pval_adj: number | null;
  split_logfoldchange: number | null;
  split_pval: number | null;
  split_pval_adj: number | null;
  split_pct_in: number | null;
  split_pct_out: number | null;
  /** An ordinary marker (up, detected, adjusted p < 0.05) */
  marker: boolean;
  /** ...that the split test also has up at adjusted p < 0.05 */
  kept: boolean;
}

export interface SplitGroup {
  group: string;
  n_cells: number;
  /** The training cluster that matches it best (Jaccard) */
  train_cluster: string | null;
  jaccard: number | null;
  /** Another group matches the same training cluster */
  shared: boolean;
  tested: boolean;
  /** Why it was not tested (too few cells, no matching training cluster) */
  reason: string | null;
  n_ordinary: number;
  n_kept: number;
  n_lost: number;
  n_split: number;
  n_split_only: number;
  kept: string[];
  lost: string[];
  split_only: string[];
  /** The group's top genes by ordinary Wilcoxon score */
  genes: SplitGene[];
}

export interface SplitMarkersResult {
  reference: string;
  reference_exists: boolean;
  method: string;
  resolution: number;
  train_resolution: number;
  n_groups: number;
  n_tested_groups: number;
  n_keep_enough: number;
  n_ordinary: number;
  n_kept: number;
  train: { n_clusters: number; ari: number; ami: number };
  groups: SplitGroup[];
  notes: string[];
  epsilon: number;
  seed: number;
  thresholds: { padj: number; min_logfc: number; min_pts: number; keeps_enough: number; genes_per_group: number; min_jaccard: number };
  seconds: { train?: number; test?: number; total?: number };
}

function columnQuery(column: string | undefined): string {
  return column ? `?column=${encodeURIComponent(column)}` : "";
}

export function getSplitMarkersInfo(datasetId: string, column?: string): Promise<SplitMarkersInfo> {
  return apiFetch(`/datasets/${datasetId}/markers/count-split${columnQuery(column)}`);
}

/**
 * Split the counts, re-cluster the training half and test each cluster on the
 * held-out half, without changing the dataset. Rejects with the server's reason;
 * aborting `signal` stops reading and, on the server, the run.
 */
export async function runSplitMarkers(
  datasetId: string,
  column: string | undefined,
  handlers: {
    onStart?: (s: SplitMarkersStart) => void;
    onProgress?: (p: SplitMarkersProgress) => void;
    onResult: (r: SplitMarkersResult) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/markers/count-split`,
    { column: column || null },
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as SplitMarkersStart);
      else if (event === "progress") handlers.onProgress?.(data as SplitMarkersProgress);
      else if (event === "result") handlers.onResult(data as SplitMarkersResult);
    },
    "The count-split check",
    handlers.signal,
  );
}

/** Keep one group's split test with the DE results (GSEA, DE overlap). */
export function keepSplitResult(
  datasetId: string,
  column: string | undefined,
  group: string,
): Promise<{ result_id: string; label: string }> {
  return apiFetch(`/datasets/${datasetId}/markers/count-split/keep`, {
    method: "POST",
    body: JSON.stringify({ column: column || null, group }),
  });
}
