import { apiFetch } from "./client";

/** How a gene set is scored per cell (backend api/v1/genesets.py, core/aucell.py). */
export type ScoreMethod = "score_genes" | "aucell";

export const SCORE_METHODS: { id: ScoreMethod; label: string }[] = [
  { id: "score_genes", label: "score_genes" },
  { id: "aucell", label: "AUCell" },
];

export function methodLabel(method: ScoreMethod | undefined): string {
  return method === "aucell" ? "AUCell" : "score_genes";
}

/** The overlay / legend label: "IFN response (AUCell)"; score_genes (the default
 *  since before there was a choice) keeps the bare name. */
export function scoreLabel(name: string, method: ScoreMethod | undefined): string {
  return method === "aucell" ? `${name} (AUCell)` : name;
}

export interface GeneSetScoreResponse {
  score_name: string;
  n_cells: number;
  scores: number[];
  min_score: number;
  max_score: number;
  genes_found: string[];
  genes_missing: string[];
  method: ScoreMethod;
  /** AUCell: top genes per cell counted; null for score_genes */
  max_rank: number | null;
  /** The matrix scored, e.g. "log-normalised .raw" */
  source: string | null;
  /** AUCell: genes the dataset has but no cell expresses (left out of the set) */
  genes_undetected: string[];
  /** AUCell: share of cells detecting fewer genes than max_rank */
  share_below_max_rank: number | null;
}

/** Score a gene set on every cell. */
export function scoreGeneSet(
  datasetId: string,
  genes: string[],
  name: string,
  method: ScoreMethod = "score_genes",
): Promise<GeneSetScoreResponse> {
  return apiFetch<GeneSetScoreResponse>(`/datasets/${datasetId}/genesets/score`, {
    method: "POST",
    body: JSON.stringify({ gene_set: genes, score_name: name, method }),
  });
}

/** Spearman's rho over all cells and the median within groups (null: none to compute). */
export interface DepthCorrelation {
  overall: number | null;
  within: number | null;
  n_groups: number;
}

export type DepthMetric = "total_counts" | "n_genes";

export interface DepthCheck {
  n_cells: number;
  /** The grouping the within-group rhos use; null when there was none */
  groupby: string | null;
  n_groups: number;
  min_group_cells: number;
  /** Where each depth measure came from (total_counts null: no counts in the dataset) */
  depth: { total_counts: string | null; n_genes: string };
  /** The measure the sentence quotes: the one either method follows most */
  metric: DepthMetric;
  methods: Record<ScoreMethod, Partial<Record<DepthMetric, DepthCorrelation>>>;
  between: DepthCorrelation | null;
  max_rank: number;
  share_below_max_rank: number;
  genes_found: string[];
  genes_missing: string[];
  sentence: string;
}

/** Which method's score follows sequencing depth on this dataset. */
export function checkScoreDepth(datasetId: string, genes: string[], groupby?: string): Promise<DepthCheck> {
  return apiFetch<DepthCheck>(`/datasets/${datasetId}/genesets/score/compare`, {
    method: "POST",
    body: JSON.stringify({ gene_set: genes, ...(groupby ? { groupby } : {}) }),
  });
}
