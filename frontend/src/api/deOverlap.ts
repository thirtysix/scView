import { apiFetch } from "./client";

/** How a kept DE result's significant genes were chosen (backend core/de_overlap.py):
 *  the adjusted p it kept, raw p adjusted by the overlap (BH within the result),
 *  or, without p-values (logistic regression), the top N coefficients per direction. */
export type DESetRule = "padj" | "padj_bh_here" | "top_n";

/** One DE result the backend still keeps (GET /de/results). */
export interface DEResultSummary {
  result_id: string;
  /** "CD14 Mono: STIM vs CTRL (Wilcoxon)" */
  label: string;
  method: string;
  /** ISO 8601, UTC */
  created_at: string;
  /** Genes the result tested (after the detection prefilter). */
  n_tested: number;
  has_pvalues: boolean;
  rule: DESetRule;
  /** At the listing's padj_max / lfc_min (logistic regression: the top-N rule). */
  n_sig_up: number;
  n_sig_down: number;
  min_pct?: number | null;
  comparison?: Record<string, unknown>;
}

export interface DEResultsList {
  /** This dataset's, newest first. */
  results: DEResultSummary[];
  /** Results the backend keeps, over all datasets. */
  keep: number;
  padj_max: number;
  lfc_min: number;
  logreg_top_n: number;
}

export type OverlapDirection = "up" | "down" | "both";

export interface DEOverlapRequest {
  result_ids: string[];
  direction: OverlapDirection;
  padj_max: number;
  lfc_min: number;
  restrict_to_common_tested: boolean;
  /** The user's gene lists, compared beside the DE results (no direction). */
  extra_sets?: { name: string; genes: string[] }[];
}

export interface DEOverlapSet {
  index: number;
  name: string;
  kind: "de" | "extra";
  result_id: string | null;
  method: string | null;
  rule: DESetRule | null;
  n: number;
  n_up: number;
  n_down: number;
  /** DE sets: genes the result tested. */
  n_tested: number | null;
  /** Extra sets: genes no result tested. */
  n_not_tested: number | null;
}

export interface DEOverlapGene {
  gene: string;
  /** log2FC in each DE result, in request order; null where the gene was not tested. */
  lfc: (number | null)[];
  /** Discordant genes only: the sets the gene is in. */
  sets?: number[] | null;
}

/** Genes in exactly these sets (indices into `sets`) and no other. */
export interface DEOverlapIntersection {
  sets: number[];
  degree: number;
  n: number;
  /** Higher in A in every DE result of the combination. */
  n_up: number;
  n_down: number;
  /** Largest mean |log2FC| first, at most `max_listed`. */
  genes: DEOverlapGene[];
}

export interface DEOverlapResponse {
  direction: OverlapDirection;
  padj_max: number;
  lfc_min: number;
  restrict_to_common_tested: boolean;
  logreg_top_n: number;
  max_listed: number;
  sets: DEOverlapSet[];
  result_ids: string[];
  n_common_tested: number;
  /** = the intersections' n (hidden ones too) + discordant n. */
  n_union: number;
  /** Largest first. */
  intersections: DEOverlapIntersection[];
  n_intersections: number;
  hidden_intersections: number;
  hidden_genes: number;
  /** "both": genes significant in several results with opposite signs. */
  discordant: { n: number; combinations: { sets: number[]; n: number }[]; genes: DEOverlapGene[] };
  pairs: { a: number; b: number; shared: number; discordant: number; union: number; jaccard: number | null }[];
  notes: string[];
}

/** The DE results the backend still keeps for this dataset, with significant counts at these cut-offs. */
export async function listDEResults(
  datasetId: string,
  padjMax?: number,
  lfcMin?: number,
): Promise<DEResultsList> {
  const q = new URLSearchParams();
  if (padjMax != null) q.set("padj_max", String(padjMax));
  if (lfcMin != null) q.set("lfc_min", String(lfcMin));
  const qs = q.toString();
  return apiFetch<DEResultsList>(`/datasets/${datasetId}/de/results${qs ? `?${qs}` : ""}`);
}

/** The overlap of 2-8 kept DE results' significant genes (and any gene lists). */
export async function computeDEOverlap(datasetId: string, body: DEOverlapRequest): Promise<DEOverlapResponse> {
  return apiFetch<DEOverlapResponse>(`/datasets/${datasetId}/de/overlap`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
