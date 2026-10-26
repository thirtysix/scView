import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

export type MiloDEGenes = "hvg" | "expressed";

export interface MiloDERequest {
  condition_key: string;
  /** Donor / sample column; left out, the backend picks the most donor-like one. */
  sample_key?: string;
  /** [reference, tested]; default: the column's first two values. */
  conditions?: string[];
  covariates?: string[];
  /** Left out: paired when every donor is in both conditions. */
  paired?: boolean;
  /** Column (cell types, clusters) to label each neighbourhood by. */
  label_key?: string;
  /** Neighbours per cell of the graph the neighbourhoods are built on (default 25). */
  k?: number;
  /** Share of cells sampled as neighbourhood centres (default 0.2). */
  prop?: number;
  /** Highly variable genes (default) or every expressed gene. */
  genes?: MiloDEGenes;
}

export interface MiloDEPlan {
  condition_key: string;
  sample_key: string;
  sample_key_detected: boolean;
  conditions: string[];
  reference: string;
  design: "paired" | "unpaired";
  covariates: string[];
  formula: string;
  n_cells: number;
  n_cells_total?: number;
  n_samples: Record<string, number>;
  representation: string;
  k: number;
  order: number;
  prop: number;
  genes_mode: MiloDEGenes;
  n_genes: number;
  counts_source: string;
  n_nhoods: number;
  n_nhoods_sampled: number;
  /** Neighbourhoods with 2+ samples of each condition holding 3+ cells there. */
  n_testable: number;
  median_nhood_size: number;
  median_samples_per_nhood: number;
  min_cells_per_sample: number;
  estimate_seconds: number;
  warnings: string[];
  notes: string[];
  dropped_samples: string[];
}

/** Per-neighbourhood columns, all n_nhoods long. */
export interface MiloDENhoods {
  index_cell: string[];
  cell: number[];
  size: number[];
  /** Samples with 3+ cells in the neighbourhood. */
  n_samples: number[];
  tested: boolean[];
  /** "paired" | "unpaired" | "" (not tested) */
  design: string[];
  /** Genes DE at spatial FDR < alpha. */
  n_de: number[];
  /** Genes DE by BH within the neighbourhood. */
  n_de_within: number[];
  label: string[] | null;
  label_fraction: (number | null)[] | null;
  /** Condition -> its cells in each neighbourhood. */
  cells: Record<string, number[]>;
  /** The smaller condition's share of the cells, both scaled to the same size. */
  minority_share: (number | null)[];
}

export interface MiloDEGeneRow {
  gene: string;
  n_tested: number;
  n_de: number;
  n_up: number;
  n_down: number;
  /** n_de / n_tested */
  share: number;
  median_logfc: number | null;
  min_spatial_fdr: number | null;
}

export interface MiloDEResult {
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
  k: number;
  order: number;
  prop: number;
  genes_mode: MiloDEGenes;
  counts_source: string;
  n_cells: number;
  n_samples: Record<string, number>;
  n_nhoods: number;
  n_tested: number;
  n_genes: number;
  n_genes_de: number;
  alpha: number;
  min_cells_per_sample: number;
  one_sided: number;
  nhoods: MiloDENhoods;
  /** Most widespread DE first. */
  genes: MiloDEGeneRow[];
  /** Embedding -> [x, y] of each index cell. */
  layouts: Record<string, [number | null, number | null][]>;
  warnings: string[];
  notes: string[];
  dropped_samples: string[];
  timings: Record<string, number>;
}

export interface MiloDEProgress {
  step: "counts" | "test" | "fdr";
  message: string;
  done?: number;
  total?: number;
}

/** One gene in every neighbourhood of the last run. */
export interface MiloDEGeneDetail {
  gene: string;
  alpha: number;
  log_fc: (number | null)[];
  pval: (number | null)[];
  spatial_fdr: (number | null)[];
  significant: boolean[];
}

/** One neighbourhood's genes in the last run, smallest p first (capped). */
export interface MiloDENhoodDetail {
  nhood: number;
  index_cell: string;
  size: number;
  n_samples: number;
  tested: boolean;
  design: string;
  label: string | null;
  n_genes_tested: number;
  n_de: number;
  n_de_within: number;
  alpha: number;
  n_rows: number;
  genes: { gene: string; log_fc: number | null; pval: number | null; fdr: number | null; spatial_fdr: number | null }[];
}

/** The design, the neighbourhoods and the time estimate; rejects with the reason a run can't. */
export function planMiloDE(datasetId: string, body: MiloDERequest): Promise<MiloDEPlan> {
  return apiFetch<MiloDEPlan>(`/datasets/${datasetId}/milode/plan`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Run neighbourhood DE and resolve with the result. Aborting `signal` closes the
 *  stream, which stops the run on the server at its next chunk. */
export async function runMiloDE(
  datasetId: string,
  body: MiloDERequest,
  handlers: { onProgress?: (p: MiloDEProgress) => void; signal?: AbortSignal } = {},
): Promise<MiloDEResult> {
  let result: MiloDEResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/milode/run`,
    body,
    (event, data) => {
      if (event === "progress") handlers.onProgress?.(data as MiloDEProgress);
      else if (event === "result") result = data as MiloDEResult;
    },
    "Neighbourhood DE",
    handlers.signal,
  );
  if (!result) throw new Error("Neighbourhood DE finished without a result.");
  return result;
}

export function miloDEGene(datasetId: string, gene: string): Promise<MiloDEGeneDetail> {
  return apiFetch(`/datasets/${datasetId}/milode/gene?gene=${encodeURIComponent(gene)}`);
}

export function miloDENhood(datasetId: string, nhood: number): Promise<MiloDENhoodDetail> {
  return apiFetch(`/datasets/${datasetId}/milode/nhood/${nhood}`);
}

/** Cells (row indices) in the given neighbourhoods of the last run. */
export function miloDECells(datasetId: string, nhoods: number[]): Promise<{ cells: number[]; n_cells: number }> {
  return apiFetch(`/datasets/${datasetId}/milode/cells`, {
    method: "POST",
    body: JSON.stringify({ nhoods }),
  });
}
