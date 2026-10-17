import { apiFetch } from "./client";

export interface DEGene {
  gene: string;
  logfoldchange: number;
  pval: number;
  pval_adj: number;
  /** Test statistic (Wilcoxon z, DESeq2 Wald): TF / pathway activity is scored on
   *  it, and GSEA ranks a Wilcoxon result by it. */
  stat?: number | null;
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
  /** Pseudobulk (method "pseudobulk") only. */
  method?: "wilcoxon" | "pseudobulk";
  n_samples_a?: number | null;
  n_samples_b?: number | null;
  n_genes_tested?: number | null;
  design?: string | null;
  counts_source?: string | null;
  dropped_samples?: string[];
  /** /de/compare: the two sides are also different batches (datasets, runs). */
  confounding?: DEConfounding | null;
  /** Names this result for GSEA, which could not send every gene back. */
  result_id?: string | null;
  /** Wilcoxon: the detection cut-off applied (null: none, as for pseudobulk or
   *  scaled expression) and the genes it left out; `n_genes_tested` counts the rest. */
  min_pct?: number | null;
  n_genes_skipped?: number | null;
  /** Every gene's larger detection fraction of the two sides, binned at 0.01 %:
   *  `detection[k]` genes were detected in [k, k + 1) / 10,000 of the cells. */
  detection?: number[] | null;
}

export interface DEConfounding {
  batch: string;
  condition: string;
  kind: "confounded" | "nested";
  message: string;
}

/**
 * One-vs-rest Wilcoxon differential expression for an arbitrary set of selected
 * cell indices (a lasso or a clicked cluster), over all genes — for a volcano plot.
 */
export async function computeDE(
  datasetId: string,
  indices: number[],
  label = "selection",
  minPct?: number,
): Promise<DEResponse> {
  return apiFetch<DEResponse>(`/datasets/${datasetId}/de`, {
    method: "POST",
    body: JSON.stringify({ indices, label, ...(minPct != null ? { min_pct: minPct } : {}) }),
  });
}

export interface DECompareRequest {
  split_by: string;
  group_a: string;
  group_b: string;
  within_column?: string;
  within_value?: string;
  replicate_key?: string;
  /** "pseudobulk": per-donor count sums tested with DESeq2 (needs replicate_key). */
  method?: "wilcoxon" | "pseudobulk";
  /** Wilcoxon: test only genes detected in this fraction of either group's cells. */
  min_pct?: number;
}

/**
 * DE between two groups of one obs column (A vs B), optionally within one
 * subgroup — e.g. dataset STIM vs CTRL within one cell type: cell-level Wilcoxon,
 * or donor-aware pseudobulk DESeq2. In the response, `n_selected` / `n_rest` are
 * the A / B cell counts.
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
