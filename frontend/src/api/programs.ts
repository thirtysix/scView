import { apiFetch } from "./client";
import { streamEvents } from "./clustering";
import type { EnrichmentResult } from "./enrichment";
import type { WatchdogEvent } from "./pipelineRun";

/** One gene of a program: its score (backend core/gene_programs.gene_scores). */
export interface ProgramGene {
  gene: string;
  /** The z-score the gene is predicted to have in a cell made entirely of this program. */
  score: number;
}

/** One saved gene program (backend core/gene_programs.py). */
export interface GeneProgram {
  /** 1-based: P1..Pk, the most used first. */
  index: number;
  name: string;
  /** Replicate spectra grouped into this program. */
  n_members: number;
  /** Replicates (seeds) with a member in it. */
  runs_found: number;
  /** Mean Pearson correlation of its member spectra (null: one member). */
  correlation: number | null;
  /** Mean silhouette of its members in the k-means grouping. */
  silhouette: number | null;
  /** Found in every replicate, members correlating >= the stable threshold. */
  stable: boolean;
  /** Mean usage share over all cells. */
  mean_usage: number;
  /** Share of cells (with any usage) where it is the largest program. */
  pct_top: number;
  top_genes: ProgramGene[];
  /** Its best enriched term (adjusted p < 0.05), once its enrichment has run. */
  top_term?: { term: string; adjusted_pvalue: number; overlap_count: number; gene_count: number } | null;
}

/** The numbers a k sweep compares. */
export interface ProgramsKStats {
  k: number;
  /** Silhouette of the k-means grouping of the replicate spectra (cNMF's stability). */
  silhouette: number | null;
  mean_correlation: number | null;
  n_stable: number;
  /** ||X - WH||^2 / ||X||^2 on the scaled input. */
  error: number;
  seconds: number;
}

export interface ProgramsByGroup {
  column: string;
  groups: string[];
  n_cells: number[];
  /** [program][group]: mean usage share. */
  mean: number[][];
}

export interface ProgramsRunnable {
  ok: boolean;
  reason: string | null;
  n_flagged: number | null;
  source: string | null;
}

export interface ProgramsDefaults {
  k: number;
  seeds: number;
  sweep: number[];
  min_k: number;
  max_k: number;
  max_seeds: number;
  max_sweep: number;
  min_genes: number;
  n_top: number;
  stable_correlation: number;
}

/** GET /programs: the saved run (when `available`), whether one can run, the defaults. */
export interface ProgramsSummary {
  available: boolean;
  runnable: ProgramsRunnable;
  defaults: ProgramsDefaults;
  running: boolean;
  n_cells_total: number;
  run_id?: string;
  created_at?: string;
  k?: number;
  seeds?: number[];
  n_cells?: number;
  n_genes?: number;
  n_flagged?: number;
  n_hvg?: number | null;
  source?: string;
  notes?: string[];
  silhouette?: number | null;
  error?: number;
  seconds?: number;
  sweep?: ProgramsKStats[];
  programs?: GeneProgram[];
  method?: string;
  by_group?: ProgramsByGroup | null;
  by_group_error?: string;
}

export interface ProgramsStart {
  k: number;
  seeds: number[];
  ks: number[];
  n_cells: number;
  n_genes: number;
  n_flagged: number;
  source: string;
  notes: string[];
  estimate_seconds: number;
}

export interface ProgramsProgress {
  k: number;
  run: number;
  runs: number;
  done: number;
  total: number;
  elapsed_s: number;
}

export interface ProgramsRunBody {
  k?: number;
  seeds?: number;
  n_hvg?: number;
  sweep?: number[];
}

export function getPrograms(datasetId: string, groupby?: string): Promise<ProgramsSummary> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/programs${q}`);
}

/**
 * Run consensus NMF, calling the handlers as the stream arrives; resolves with the
 * saved programs. Rejects with the server's reason; aborting `signal` stops the run
 * (nothing is saved).
 */
export async function runPrograms(
  datasetId: string,
  body: ProgramsRunBody,
  handlers: {
    onStart?: (s: ProgramsStart) => void;
    onProgress?: (p: ProgramsProgress) => void;
    onK?: (s: ProgramsKStats) => void;
    onWatchdog?: (w: WatchdogEvent) => void;
    signal?: AbortSignal;
  } = {},
): Promise<ProgramsSummary> {
  let result: ProgramsSummary | null = null;
  await streamEvents(
    `/datasets/${datasetId}/programs/run-stream`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as ProgramsStart);
      else if (event === "progress") handlers.onProgress?.(data as ProgramsProgress);
      else if (event === "k_result") handlers.onK?.(data as ProgramsKStats);
      else if (event === "watchdog") handlers.onWatchdog?.(data as WatchdogEvent);
      else if (event === "result") result = data as ProgramsSummary;
    },
    "The gene programs run",
    handlers.signal,
  );
  if (!result) throw new Error("The gene programs run ended without a result; nothing was saved.");
  return result;
}

export interface ProgramUsage {
  index: number;
  name: string;
  raw: boolean;
  run_id: string;
  top_genes: string[];
  /** Per cell, in the dataset's cell order. */
  values: number[];
  /** The values split by the `groupby` column, for the violin. */
  violin: Record<string, number[]> | null;
}

export function getProgramUsage(datasetId: string, index: number, groupby?: string): Promise<ProgramUsage> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/programs/${index}/usage${q}`);
}

export interface ProgramEnrichment {
  index: number;
  name: string;
  run_id: string;
  collections: string[];
  query: string[];
  n_query: number;
  n_excluded_ribo_mito: number;
  background_n: number;
  n_terms: number;
  n_significant: number;
  results: EnrichmentResult[];
  warning: string;
}

/** ORA of a program's top genes (MSigDB Hallmark by default) against the genes NMF used. */
export function enrichProgram(
  datasetId: string,
  index: number,
  body: { collections?: string[]; n_genes?: number; exclude_ribo_mito?: boolean } = {},
): Promise<ProgramEnrichment> {
  return apiFetch(`/datasets/${datasetId}/programs/${index}/enrichment`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
