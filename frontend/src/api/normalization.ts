import { apiFetch } from "./client";

/** Per-gene mean vs variance before and after normalisation (backend core/normalization.py). */
export interface MeanVariance {
  n_cells: number;
  /** Cells sampled for the diagnostic (at most 20,000). */
  n_cells_used: number;
  n_genes: number;
  n_genes_undetected: number;
  theta: number;
  clip: number;
  counts_source: string;
  /** How scView last processed the dataset ("log" | "pearson_residuals"), null if never. */
  normalization_method: string | null;
  genes: string[];
  /** Mean of the counts. */
  mean: number[];
  var_counts: number[];
  /** Variance after the shifted log (normalize_total 1e4 + log1p). */
  var_log: number[];
  /** Variance of the analytic Pearson residuals. */
  var_residuals: number[];
  /** The dataset's current variable genes, aligned with `genes`; null if none chosen. */
  hvg: boolean[] | null;
}

export async function getMeanVariance(datasetId: string): Promise<MeanVariance> {
  return apiFetch<MeanVariance>(`/datasets/${datasetId}/qc/mean-variance`);
}
