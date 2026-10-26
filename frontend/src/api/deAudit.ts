import { streamEvents } from "./clustering";

/** A between-groups comparison to audit: a kept DE result, or its scope. */
export interface DEAuditRequest {
  result_id?: string;
  split_by?: string;
  group_a?: string;
  group_b?: string;
  within_column?: string;
  within_value?: string;
  /** Donor / sample column; left out, the result's (or the most donor-like one). */
  replicate_key?: string;
  method?: "wilcoxon" | "t-test";
  min_pct?: number;
  /** Donor shuffles to run (1-100, default 20). */
  n_permutations?: number;
  /** The shuffles stop here with those done (default 300 s). */
  max_seconds?: number;
  pb_method?: "deseq2" | "voom" | "voombygroup";
  seed?: number;
}

export interface DEAuditStart {
  label: string;
  donor_key: string;
  method: string;
  /** paired: half the donors' samples swapped; donors / samples: reassigned. */
  scheme: "paired" | "donors" | "samples";
  n_donors_a: number;
  n_donors_b: number;
  /** Shuffles that will run (0: too few donors). */
  n_permutations: number;
  n_possible: number;
  n_cells_a: number;
  n_cells_b: number;
}

export interface DEAuditProgress {
  step: "observed" | "shuffle" | "pseudobulk";
  message: string;
  i?: number;
  n?: number;
  count?: number;
}

export interface DEAuditPseudobulk {
  method: string;
  name: string;
  n_tested: number;
  n_samples_a: number;
  n_samples_b: number;
  design: string;
  n_sig_cell: number;
  n_sig_pseudobulk: number;
  n_shared: number;
  jaccard: number | null;
  n_genes_common: number;
  spearman_lfc: number | null;
}

/** A gene significant at cell level and not in pseudobulk. */
export interface DEAuditGene {
  gene: string;
  cell_lfc: number;
  cell_padj: number;
  /** null: pseudobulk gave the gene no p-value (filtered out, or an outlier sample). */
  pb_lfc: number | null;
  pb_padj: number | null;
  pb_tested: boolean;
  /** Share of the difference lost when the top donor is left out (0-1). */
  top_donor_share: number | null;
  top_donor: string | null;
}

export interface DEAuditResult {
  label: string;
  comparison: Record<string, unknown>;
  donor_key: string;
  method: string;
  scheme: DEAuditStart["scheme"];
  scheme_text: string;
  n_donors_a: number;
  n_donors_b: number;
  n_cells_a: number;
  n_cells_b: number;
  n_genes_tested: number;
  /** Genes at adjusted p < alpha in the observed comparison. */
  observed: number;
  /** The same count in each donor shuffle. */
  null_counts: number[];
  n_possible: number;
  exhaustive: boolean;
  empirical_p: number | null;
  stopped: string | null;
  permutation_note: string | null;
  /** -log10 p quantiles: uniform expectation, observed, the shuffles pooled. */
  qq: { expected: number[]; observed: number[]; permuted: number[] | null };
  pseudobulk: DEAuditPseudobulk | null;
  pseudobulk_note: string | null;
  cell_only: DEAuditGene[];
  n_cell_only: number;
  verdict: string;
  seconds: number;
  alpha: number;
  null_share: number;
  notes: string[];
}

/**
 * Audit a cell-level comparison: the same test after shuffling the condition
 * across donors, and pseudobulk on the same cells. Resolves with the result;
 * aborting `signal` stops the run between shuffles.
 */
export async function runDEAudit(
  datasetId: string,
  body: DEAuditRequest,
  handlers: {
    onStart?: (s: DEAuditStart) => void;
    onProgress?: (p: DEAuditProgress) => void;
    signal?: AbortSignal;
  } = {},
): Promise<DEAuditResult> {
  let result: DEAuditResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/de/audit`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as DEAuditStart);
      else if (event === "progress") handlers.onProgress?.(data as DEAuditProgress);
      else if (event === "result") result = data as DEAuditResult;
    },
    "The DE audit",
    handlers.signal,
  );
  if (!result) throw new Error("The DE audit finished without a result.");
  return result;
}
