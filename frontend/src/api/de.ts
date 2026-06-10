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
