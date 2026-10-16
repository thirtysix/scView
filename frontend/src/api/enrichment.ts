import { apiFetch } from "./client";

/** One enriched gene set (backend api/v1/enrichment.py EnrichmentResult). */
export interface EnrichmentResult {
  term: string;
  pvalue: number;
  adjusted_pvalue: number;
  overlap_count: number;
  gene_count: number;
  genes: string[];
  collection: string;
  /** Set for a custom collection: its name, "(deleted)" once it is gone. */
  collection_name?: string;
}

export interface GeneListEnrichmentRequest {
  up: string[];
  down: string[];
  /** The genes that were tested: the universe both lists were drawn from. */
  background: string[];
  collections: string[];
  exclude_ribo_mito: boolean;
  /** Most significant terms returned per list (backend default 1000). */
  max_terms?: number;
}

/** ORA of one list: how many genes went in, and the enriched terms. */
export interface GeneListSide {
  n_genes_requested: number;
  /** In the background and not filtered out. */
  n_genes_used: number;
  n_excluded_ribo_mito: number;
  /** Terms sharing a gene with the list; `results` keeps the most significant (max_terms). */
  n_terms: number;
  /** Of n_terms, adjusted p < 0.05 (counted before the cap). */
  n_significant: number;
  results: EnrichmentResult[];
}

export interface GeneListEnrichmentResponse {
  /** Background genes the dataset measures. */
  background_n: number;
  collections: string[];
  exclude_ribo_mito: boolean;
  up: GeneListSide;
  down: GeneListSide;
  source: string;
  /** Set when the (human) gene sets may not fit the dataset, e.g. a mouse dataset. */
  warning?: string;
}

/**
 * Over-representation of explicit gene lists (a DE result's up and down genes),
 * each tested on its own against `background`.
 */
export async function enrichGeneLists(
  datasetId: string,
  body: GeneListEnrichmentRequest,
): Promise<GeneListEnrichmentResponse> {
  return apiFetch<GeneListEnrichmentResponse>(`/datasets/${datasetId}/enrichment/gene-lists`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** One gene of a DE result for GSEA: ranked by `score` when every gene has one. */
export interface GseaGene {
  gene: string;
  logfoldchange?: number | null;
  pval?: number | null;
  score?: number | null;
}

export interface GseaRequest {
  /** Every gene a DE tested ... */
  genes?: GseaGene[];
  /** ... or a marker group, ranked by scanpy's score. */
  column?: string;
  group?: string;
  collections: string[];
  exclude_ribo_mito: boolean;
  /** Left unset, the backend picks as many as its cost budget allows. */
  permutations?: number;
  max_terms?: number;
}

/** One gene set's GSEA result (backend GseaTerm). */
export interface GseaTerm {
  term: string;
  es: number;
  /** > 0: the set's genes sit at the top of the ranking. */
  nes: number;
  /** Nominal, from permutations; 0 means below 1/permutations. */
  pvalue: number;
  fdr: number;
  /** The set's genes in the ranking. */
  set_size: number;
  /** The genes driving the score, in ranking order. */
  leading_edge: string[];
  collection: string;
  collection_name?: string;
}

export interface GseaSide {
  n_terms: number;
  /** FDR < 0.05, counted before `results` is capped. */
  n_significant: number;
  results: GseaTerm[];
}

export interface GseaResponse {
  rank_metric: "wilcoxon_score" | "signed_log10_p" | "marker_score";
  rank_metric_label: string;
  n_genes_ranked: number;
  n_excluded_ribo_mito: number;
  n_sets_tested: number;
  permutations: number;
  min_size: number;
  max_size: number;
  collections: string[];
  exclude_ribo_mito: boolean;
  /** NES > 0, and NES < 0. */
  up: GseaSide;
  down: GseaSide;
  /** Seconds the run took; 0 when served from the backend's cache. */
  seconds: number;
  cached: boolean;
  column?: string;
  group?: string;
  warning?: string;
}

/** Preranked GSEA (gseapy prerank) of a DE result or a marker group. */
export async function runGsea(datasetId: string, body: GseaRequest): Promise<GseaResponse> {
  return apiFetch<GseaResponse>(`/datasets/${datasetId}/enrichment/gsea`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
