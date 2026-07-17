import { apiFetch } from "@/api/client";

/** One enriched term. Mirrors a node from core/enrichment_network.py. */
export interface NetworkNode {
  id: number;
  term: string;
  adjusted_pvalue: number;
  overlap_count: number;
  gene_count: number;
  collection: string;
  genes: string[];
  cluster: number;
  /** True for the cluster's most significant term, which supplies its label. */
  is_hub: boolean;
  degree: number;
  x: number;
  y: number;
}

export interface NetworkEdge {
  source: number;
  target: number;
  weight: number;
  jaccard: number;
  n_shared: number;
  same_cluster: boolean;
}

/** A group of redundant terms: the unit the network view actually shows. */
export interface NetworkCluster {
  cluster: number;
  /** The cluster's most significant term. */
  label: string;
  n_terms: number;
  best_adjusted_pvalue: number;
  /** Union of the member terms' overlap genes. */
  genes: string[];
  terms: string[];
}

export interface NetworkParams {
  fdr: number;
  metric: string;
  min_similarity: number;
  resolution: number;
  seed: number;
  min_term_size: number;
  max_term_size: number;
}

export interface EnrichmentNetworkResponse {
  dataset_id: string;
  group: string;
  groupby: string;
  n_genes_used: number;
  n_significant: number;
  n_terms: number;
  n_clusters: number;
  n_edges: number;
  /** Terms dropped for falling outside the gene-count bounds. */
  n_size_filtered: number;
  /** Terms dropped by the max_terms cap; reported so a bounded view is never mistaken for a complete one. */
  truncated: number;
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  clusters: NetworkCluster[];
  params: NetworkParams;
}

export interface EnrichmentNetworkRequest {
  column: string;
  group: string;
  n_genes: number;
  collections: string[];
  resolution?: number;
  includeEdges?: boolean;
  /** Drop ribosomal/mitochondrial genes from the query (default true). */
  excludeRiboMito?: boolean;
}

export function fetchEnrichmentNetwork(
  datasetId: string,
  req: EnrichmentNetworkRequest,
): Promise<EnrichmentNetworkResponse> {
  return apiFetch<EnrichmentNetworkResponse>(
    `/datasets/${datasetId}/enrichment/network`,
    {
      method: "POST",
      body: JSON.stringify({
        column: req.column,
        group: req.group,
        n_genes: req.n_genes,
        collections: req.collections,
        resolution: req.resolution ?? 0.4,
        // Edges are only needed to draw the graph. Turning this off drops tens of
        // thousands of rows from the payload if the graph is ever removed.
        include_edges: req.includeEdges ?? true,
        exclude_ribo_mito: req.excludeRiboMito ?? true,
      }),
    },
  );
}
