import { apiFetch } from "./client";

/** How a gene set is scored per cell (backend api/v1/genesets.py, core/aucell.py,
 *  core/ssgsea.py, core/gsva.py). */
export type ScoreMethod = "score_genes" | "aucell" | "ssgsea" | "gsva";

export const SCORE_METHODS: { id: ScoreMethod; label: string }[] = [
  { id: "score_genes", label: "score_genes" },
  { id: "aucell", label: "AUCell" },
  { id: "ssgsea", label: "ssGSEA" },
  { id: "gsva", label: "GSVA" },
];

/** Every method id, for checking a stored or loaded value. */
export const SCORE_METHOD_IDS = SCORE_METHODS.map((m) => m.id);

export function isScoreMethod(v: unknown): v is ScoreMethod {
  return typeof v === "string" && (SCORE_METHOD_IDS as string[]).includes(v);
}

export function methodLabel(method: ScoreMethod | undefined): string {
  return SCORE_METHODS.find((m) => m.id === method)?.label ?? "score_genes";
}

/** The overlay / legend label: "IFN response (AUCell)"; score_genes (the default
 *  since before there was a choice) keeps the bare name. */
export function scoreLabel(name: string, method: ScoreMethod | undefined): string {
  return method && method !== "score_genes" ? `${name} (${methodLabel(method)})` : name;
}

export interface GeneSetScoreResponse {
  score_name: string;
  n_cells: number;
  scores: number[];
  min_score: number;
  max_score: number;
  genes_found: string[];
  genes_missing: string[];
  /** A targeted panel's caution (backend core/targeted_panel.py). */
  panel_caution?: string | null;
  method: ScoreMethod;
  /** AUCell: top genes per cell counted; null for score_genes */
  max_rank: number | null;
  /** The matrix scored, e.g. "log-normalised .raw" */
  source: string | null;
  /** AUCell: genes the dataset has but no cell expresses (left out of the set) */
  genes_undetected: string[];
  /** AUCell: share of cells detecting fewer genes than max_rank */
  share_below_max_rank: number | null;
  /** ssGSEA / GSVA: how the score was computed, in one line */
  note?: string | null;
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
  /** Per method scored (ssGSEA / GSVA left out when in `skipped`) */
  methods: Partial<Record<ScoreMethod, Partial<Record<DepthMetric, DepthCorrelation>>>>;
  /** score_genes vs AUCell */
  between: DepthCorrelation | null;
  /** Each other method vs score_genes */
  agreement?: Partial<Record<ScoreMethod, DepthCorrelation>>;
  /** Methods left out, with the reason (too large a dataset, no gene to rank) */
  skipped?: Partial<Record<ScoreMethod, string>>;
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
