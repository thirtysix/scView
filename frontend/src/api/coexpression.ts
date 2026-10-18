import { apiFetch } from "./client";

export type CoexpressionMethod = "pearson" | "spearman";

export interface CoexpressedGene {
  gene: string;
  r: number;
  /** Cells (of the chosen ones) with the gene above zero. */
  n_expressing: number;
  pct_expressing: number;
}

/** Genes correlated with a query gene across cells (backend core/coexpression.py). */
export interface CoexpressionResponse {
  gene: string;
  method: CoexpressionMethod;
  /** "all cells", "selection" or "<column> = <value>" */
  scope: string;
  n_cells: number;
  n_cells_total: number;
  query_n_expressing: number;
  /** Expressing cells a gene needs to be correlated; null for scaled values. */
  min_expressing: number | null;
  n_genes_tested: number;
  n_skipped_rare: number;
  n_skipped_constant: number;
  /** The matrix read: "raw" or "X". */
  layer: string;
  values: "log-normalised" | "log-normalised from counts" | "scaled";
  positive: CoexpressedGene[];
  negative: CoexpressedGene[];
}

export interface CoexpressionRequest {
  gene: string;
  method?: CoexpressionMethod;
  /** A selection of cells; or `column` + `value` for one group; neither = every cell. */
  indices?: number[];
  column?: string;
  value?: string;
  top_n?: number;
}

export async function fetchCoexpression(
  datasetId: string,
  body: CoexpressionRequest,
): Promise<CoexpressionResponse> {
  return apiFetch<CoexpressionResponse>(`/datasets/${datasetId}/expression/coexpression`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
