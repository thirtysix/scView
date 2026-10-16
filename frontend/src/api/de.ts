import { apiFetch } from "./client";

export interface DEGene {
  gene: string;
  logfoldchange: number;
  pval: number;
  pval_adj: number;
}

export interface DEResponse {
  n_selected: number;
  n_rest: number;
  label: string;
  genes: DEGene[];
  /** Replicate guard (/de/compare only). */
  replicate_key?: string | null;
  n_donors_a?: number | null;
  n_donors_b?: number | null;
  top_donor_share_a?: number | null;
  top_donor_share_b?: number | null;
  underreplicated?: boolean;
}

/**
 * One-vs-rest Wilcoxon differential expression for an arbitrary set of selected
 * cell indices (a lasso or a clicked cluster), over all genes — for a volcano plot.
 */
export async function computeDE(
  datasetId: string,
  indices: number[],
  label = "selection",
): Promise<DEResponse> {
  return apiFetch<DEResponse>(`/datasets/${datasetId}/de`, {
    method: "POST",
    body: JSON.stringify({ indices, label }),
  });
}

export interface DECompareRequest {
  split_by: string;
  group_a: string;
  group_b: string;
  within_column?: string;
  within_value?: string;
  replicate_key?: string;
}

/**
 * Wilcoxon DE between two groups of one obs column (A vs B), optionally within
 * one subgroup — e.g. dataset STIM vs CTRL within one cell type. In the response,
 * `n_selected` / `n_rest` are the A / B cell counts.
 */
export async function computeDECompare(
  datasetId: string,
  body: DECompareRequest,
): Promise<DEResponse> {
  return apiFetch<DEResponse>(`/datasets/${datasetId}/de/compare`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
