import { apiFetch } from "./client";
import type { CellMethod } from "@/lib/deMethods";

export interface DEGene {
  gene: string;
  logfoldchange: number;
  /** null for logistic regression, which gives no p-values. */
  pval: number | null;
  pval_adj: number | null;
  /** Test statistic (Wilcoxon z, Welch t, logistic regression coefficient, DESeq2
   *  Wald): TF / pathway activity is scored on it, and GSEA ranks a cell-level
   *  result by it. */
  stat?: number | null;
}

/** The DE test: cell-level (wilcoxon, Welch "t-test", "logreg") or pseudobulk. */
export type DEMethod = CellMethod | "pseudobulk";

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
  /** The test that ran (absent from older servers: Wilcoxon). */
  method?: DEMethod;
  /** Pseudobulk (method "pseudobulk") only. */
  n_samples_a?: number | null;
  n_samples_b?: number | null;
  n_genes_tested?: number | null;
  design?: string | null;
  counts_source?: string | null;
  dropped_samples?: string[];
  /** /de/compare: the two sides are also different batches (datasets, runs). */
  confounding?: DEConfounding | null;
  /** /de/compare: each side's share of every cluster/state (absent from older servers). */
  composition?: DEComposition | null;
  /** Names this result for GSEA, which could not send every gene back. */
  result_id?: string | null;
  /** Wilcoxon: the detection cut-off applied (null: none, as for pseudobulk or
   *  scaled expression) and the genes it left out; `n_genes_tested` counts the rest. */
  min_pct?: number | null;
  n_genes_skipped?: number | null;
  /** Every gene's larger detection fraction of the two sides, binned at 0.01 %:
   *  `detection[k]` genes were detected in [k, k + 1) / 10,000 of the cells. */
  detection?: number[] | null;
  /** The two sides' RNA content; null when the dataset holds no raw counts. */
  transcriptome_size?: TranscriptomeSize | null;
}

/** Median raw total counts per cell of the two sides (backend core/transcriptome_size.py). */
export interface TranscriptomeSize {
  /** The obs column, or where the counts were summed from. */
  source: string;
  median_a: number;
  median_b: number;
  /** Larger median over smaller; null when one is 0. */
  ratio: number | null;
  /** The ratio is `threshold` or more: fold changes of broadly expressed genes shift. */
  caution: boolean;
  threshold: number;
  /** A sample of each side's per-cell totals, for a violin. */
  values_a: number[];
  values_b: number[];
}

/** One group of a composition check: each side's share of it among the compared cells. */
export interface DECompositionGroup {
  group: string;
  n_a: number;
  n_b: number;
  /** Of side A's labelled compared cells (0-1). */
  share_a: number;
  share_b: number;
  /** (share_a - share_b) x 100. */
  diff_points: number;
  p?: number | null; // Fisher's exact test, computed for the candidates only
  /** Past `threshold_points`, with `min_cells` cells on a side. */
  flagged: boolean;
  /** Nearly absent from one side: maybe a state the comparison itself creates. */
  one_sided: boolean;
}

/** The two sides' composition by a label column (backend core/de_composition.py). */
export interface DEComposition {
  /** null: not checked; `note` says why. */
  column: string | null;
  label_a: string;
  label_b: string;
  /** The column was the active clustering (none asked for). */
  auto: boolean;
  n_a: number;
  n_b: number;
  /** Largest difference first; at most 50 (`n_groups` counts all). */
  groups: DECompositionGroup[];
  n_groups: number;
  warn: boolean;
  message: string | null;
  note: string | null;
  threshold_points: number;
  min_cells: number;
  max_p?: number; // Fisher's exact p a flagged group must be under
  n_missing?: number;
  /** The sides are different groups of `column`: composition is the comparison. */
  nested?: boolean;
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
  method?: CellMethod,
): Promise<DEResponse> {
  return apiFetch<DEResponse>(`/datasets/${datasetId}/de`, {
    method: "POST",
    body: JSON.stringify({
      indices,
      label,
      ...(minPct != null ? { min_pct: minPct } : {}),
      ...(method && method !== "wilcoxon" ? { method } : {}),
    }),
  });
}

export interface DECompareRequest {
  split_by: string;
  group_a: string;
  group_b: string;
  within_column?: string;
  within_value?: string;
  replicate_key?: string;
  /** "pseudobulk": per-donor count sums tested with DESeq2 (needs replicate_key);
   *  the others are cell-level tests (Wilcoxon when absent). */
  method?: DEMethod;
  /** Cell-level: test only genes detected in this fraction of either group's cells. */
  min_pct?: number;
  /** The label column the composition check uses (the active clustering when absent). */
  composition_by?: string;
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

/** One cell-level test of a methods comparison (backend core/de_methods.py). */
export interface DEMethodSummary {
  method: CellMethod;
  label: string;
  stat_label: string;
  has_pvalues: boolean;
  /** Genes with adjusted p <= 0.05; null without p-values (logistic regression). */
  n_significant: number | null;
  seconds: number;
}

/** Two tests' agreement: Jaccard of their top-N genes on each side and the
 *  Spearman correlation of their statistics. */
export interface DEMethodPair {
  a: CellMethod;
  b: CellMethod;
  jaccard_up: number | null;
  shared_up: number;
  jaccard_down: number | null;
  shared_down: number;
  spearman: number | null;
}

/** A gene in one test's top N on a side and no other test's: each test's rank. */
export interface DEMethodUnique {
  gene: string;
  method: CellMethod;
  side: "up" | "down";
  ranks: Record<string, number | null>;
}

export interface DEMethodsResponse {
  n_selected: number;
  n_rest: number;
  label: string;
  top_n: number;
  methods: DEMethodSummary[];
  pairs: DEMethodPair[];
  unique: DEMethodUnique[];
  /** Tests that cannot run here, with the reason (MAST). */
  unavailable: Record<string, string>;
  min_pct?: number | null;
  n_genes_tested: number;
  n_genes_skipped?: number | null;
}

/** The selection vs the rest under every cell-level test, and their agreement. */
export async function computeDEMethods(
  datasetId: string,
  body: { indices: number[]; label?: string; min_pct?: number; top_n?: number },
): Promise<DEMethodsResponse> {
  return apiFetch<DEMethodsResponse>(`/datasets/${datasetId}/de/methods`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** A vs B (as computeDECompare) under every cell-level test, and their agreement. */
export async function computeDECompareMethods(
  datasetId: string,
  body: Omit<DECompareRequest, "method"> & { top_n?: number },
): Promise<DEMethodsResponse> {
  return apiFetch<DEMethodsResponse>(`/datasets/${datasetId}/de/compare/methods`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** What a composition check is asked about: the /de/compare scope, and the label column. */
export interface DECompositionParams {
  split_by: string;
  group_a: string;
  group_b: string;
  within_column?: string;
  within_value?: string;
  /** The label column; the active clustering when absent. */
  column?: string;
}

/**
 * Each side's share of every cluster/state among the cells a /de/compare of
 * these groups would test, before it is run (obs only, fast).
 */
export async function fetchDEComposition(
  datasetId: string,
  params: DECompositionParams,
  signal?: AbortSignal,
): Promise<DEComposition> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v);
  return apiFetch<DEComposition>(`/datasets/${datasetId}/de/composition?${q}`, { signal });
}
