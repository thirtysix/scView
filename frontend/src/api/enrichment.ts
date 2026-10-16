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
