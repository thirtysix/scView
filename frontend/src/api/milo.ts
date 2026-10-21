import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

export interface MiloRequest {
  condition_key: string;
  /** Donor / sample column; left out, the backend picks the most donor-like one. */
  sample_key?: string;
  /** [reference, tested]; default: the column's first two values. */
  conditions?: string[];
  /** Per-sample columns (batch, sex, site) added to the design. */
  covariates?: string[];
  /** Left out: paired when every donor is in both conditions. */
  paired?: boolean;
  /** Column (cell types, clusters) to label each neighbourhood by. */
  label_key?: string;
  /** Share of cells sampled as index cells (default 0.1; 0.05 above 30k cells). */
  prop?: number;
}

export interface MiloPlan {
  condition_key: string;
  sample_key: string;
  sample_key_detected: boolean;
  /** [reference, tested] */
  conditions: string[];
  reference: string;
  design: "paired" | "unpaired";
  covariates: string[];
  formula: string;
  n_cells: number;
  n_cells_total?: number;
  n_samples: Record<string, number>;
  prop: number;
  /** Index cells sampled; refinement merges some, so the run has at most this many. */
  n_nhoods_estimate: number;
  estimate_seconds: number;
  representation: string;
  warnings: string[];
  notes: string[];
  dropped_samples: string[];
}

export interface MiloLabelSummary {
  label: string;
  n_nhoods: number;
  n_up: number;
  n_down: number;
  median_logfc: number | null;
}

/** Per-neighbourhood columns, all the same length (n_nhoods). */
export interface MiloNhoods {
  index_cell: string[];
  /** Row index of each index cell. */
  cell: number[];
  size: number[];
  /** Commonest label ("Mixed" below `mixed_fraction`); null without a label column. */
  label: string[] | null;
  label_fraction: (number | null)[] | null;
  /** log2 fold change, tested condition over reference. */
  log_fc: (number | null)[];
  pval: (number | null)[];
  spatial_fdr: (number | null)[];
  /** Condition -> cells of that condition in each neighbourhood. */
  counts: Record<string, number[]>;
}

export interface MiloResult {
  condition_key: string;
  sample_key: string;
  sample_key_detected: boolean;
  label_key: string | null;
  conditions: string[];
  reference: string;
  design: "paired" | "unpaired";
  covariates: string[];
  formula: string;
  representation: string;
  prop: number;
  seed: number;
  n_cells: number;
  n_samples: Record<string, number>;
  n_nhoods: number;
  n_tested: number;
  mean_nhood_size: number;
  alpha: number;
  mixed_fraction: number;
  n_significant: number;
  n_up: number;
  n_down: number;
  nhoods: MiloNhoods;
  labels: MiloLabelSummary[];
  /** [i, j, shared cells], most shared first (capped). */
  edges: [number, number, number][];
  n_edges_total: number;
  /** Embedding -> [x, y] of each index cell. */
  layouts: Record<string, [number | null, number | null][]>;
  warnings: string[];
  notes: string[];
  dropped_samples: string[];
  timings: Record<string, number>;
}

export interface MiloProgress {
  step: "nhoods" | "counts" | "test" | "annotate";
  message: string;
}

/** The design a run would use and its time estimate; rejects with the reason a run can't. */
export function planMilo(datasetId: string, body: MiloRequest): Promise<MiloPlan> {
  return apiFetch<MiloPlan>(`/datasets/${datasetId}/milo/plan`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/**
 * Run Milo (neighbourhoods, counts, a negative binomial test per neighbourhood,
 * spatial FDR) and resolve with the result, reporting each step as it starts.
 * Aborting `signal` stops reading.
 */
export async function runMilo(
  datasetId: string,
  body: MiloRequest,
  handlers: { onStart?: (p: MiloPlan) => void; onProgress?: (p: MiloProgress) => void; signal?: AbortSignal } = {},
): Promise<MiloResult> {
  let result: MiloResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/milo/run`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as MiloPlan);
      else if (event === "progress") handlers.onProgress?.(data as MiloProgress);
      else if (event === "result") result = data as MiloResult;
    },
    "Milo",
    handlers.signal,
  );
  if (!result) throw new Error("Milo finished without a result.");
  return result;
}

/** Cells (row indices) in the given neighbourhoods of the dataset's last run. */
export function miloCells(datasetId: string, nhoods: number[]): Promise<{ cells: number[]; n_cells: number }> {
  return apiFetch(`/datasets/${datasetId}/milo/cells`, {
    method: "POST",
    body: JSON.stringify({ nhoods }),
  });
}
