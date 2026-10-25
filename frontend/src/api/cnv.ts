import { apiFetch } from "./client";
import { streamEvents } from "./clustering";
import type { WatchdogEvent } from "./pipelineRun";

/** Whether a copy-number run can start (backend core/cnv.runnable). */
export interface CnvRunnable {
  ok: boolean;
  reason: string | null;
  genome: "human" | "mouse" | null;
  assembly: string | null;
  genome_why: string | null;
  source: string | null;
  /** Genes of the matrix with a position on the genome (before the detection filter). */
  n_placed: number | null;
  match: { n_genes: number; by_id: number; by_symbol: number; unplaced: number; excluded: number } | null;
}

export interface CnvDefaults {
  window: number;
  step: number;
  exclude: string[];
  min_frac: number;
  threshold: number;
  min_window: number;
  max_window: number;
  max_step: number;
  min_ref_cells: number;
  high_ratio: number;
}

/** One group's CNV score box (quartiles, Tukey whiskers). */
export interface CnvGroupScore {
  group: string;
  n: number;
  mean: number;
  q1: number;
  median: number;
  q3: number;
  lo: number;
  hi: number;
  reference: boolean;
  /** Mean score / the reference cells' mean score. */
  ratio: number | null;
}

/** One CNV cluster (obs cnv_leiden). */
export interface CnvCluster {
  cluster: string;
  n: number;
  mean: number;
  ratio: number | null;
  reference_share: number;
  gains: string[];
  losses: string[];
  /** Score >= high_ratio x the reference and under half reference cells. */
  high: boolean;
}

/** GET /cnv: the saved result (when `available`), whether a run can start, defaults. */
export interface CnvSummary {
  available: boolean;
  runnable: CnvRunnable;
  defaults: CnvDefaults;
  running: boolean;
  n_cells_total: number;
  estimate?: { need_bytes: number; available_bytes: number | null; level: string; seconds: number };
  run_id?: string;
  genome?: string;
  assembly?: string;
  reference_column?: string;
  reference_groups?: string[];
  n_reference_cells?: number;
  n_cells?: number;
  n_genes?: number;
  n_windows?: number;
  n_low?: number;
  source?: string;
  notes?: string[];
  params?: { window: number; step: number; exclude: string[]; min_frac: number; threshold: number | null };
  reference_score?: number | null;
  seconds?: number;
  method?: string;
  by_group?: { column: string; groups: CnvGroupScore[] } | null;
  by_group_error?: string;
  clusters?: CnvCluster[];
}

export interface CnvStart {
  n_cells: number;
  n_genes: number;
  n_windows: number;
  genome: string;
  assembly: string;
  reference_column: string;
  reference_groups: string[];
  n_reference_cells: number;
  source: string;
  n_low: number;
  notes: string[];
  estimate_seconds: number;
}

export interface CnvProgress {
  stage: "reference" | "windows" | "clusters";
  done: number;
  total: number;
  elapsed_s: number;
}

export interface CnvRunBody {
  reference_column: string;
  reference_groups: string[];
  window?: number;
  step?: number;
  exclude_chromosomes?: string[];
  min_frac?: number;
  threshold?: number | null;
  genome?: string;
}

/** GET /cnv/heatmap: sampled cells x genome as int8 codes (value = code x scale). */
export interface CnvHeatmap {
  column: string;
  groups: { group: string; n_sampled: number; n_total: number; reference: boolean }[];
  n_rows: number;
  n_cols: number;
  chromosomes: { chromosome: string; start: number; end: number }[];
  vmax: number;
  scale: number;
  values: string;
  clusters: string[];
  n_cells_total: number;
}

export function getCnv(datasetId: string, groupby?: string): Promise<CnvSummary> {
  const q = groupby ? `?${new URLSearchParams({ groupby }).toString()}` : "";
  return apiFetch(`/datasets/${datasetId}/cnv${q}`);
}

export function getCnvHeatmap(datasetId: string, groupby?: string, maxCells?: number): Promise<CnvHeatmap> {
  const p = new URLSearchParams();
  if (groupby) p.set("groupby", groupby);
  if (maxCells) p.set("max_cells", String(maxCells));
  const q = p.toString();
  return apiFetch(`/datasets/${datasetId}/cnv/heatmap${q ? `?${q}` : ""}`);
}

/**
 * Infer copy number, calling the handlers as the stream arrives; resolves with the
 * saved summary. Rejects with the server's reason; aborting `signal` stops the run
 * (nothing is saved).
 */
export async function runCnv(
  datasetId: string,
  body: CnvRunBody,
  handlers: {
    onStart?: (s: CnvStart) => void;
    onProgress?: (p: CnvProgress) => void;
    onWatchdog?: (w: WatchdogEvent) => void;
    signal?: AbortSignal;
  } = {},
): Promise<CnvSummary> {
  let result: CnvSummary | null = null;
  await streamEvents(
    `/datasets/${datasetId}/cnv/run`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as CnvStart);
      else if (event === "progress") handlers.onProgress?.(data as CnvProgress);
      else if (event === "watchdog") handlers.onWatchdog?.(data as WatchdogEvent);
      else if (event === "result") result = data as CnvSummary;
    },
    "The copy-number run",
    handlers.signal,
  );
  if (!result) throw new Error("The copy-number run ended without a result; nothing was saved.");
  return result;
}
