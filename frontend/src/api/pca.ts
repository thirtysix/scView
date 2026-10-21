import { apiFetch } from "./client";

/** One gene's loading on a PC (backend core/pca_diagnostics.py). */
export interface LoadingGene {
  gene: string;
  loading: number;
}

export interface PcLoadings {
  /** 1-based */
  pc: number;
  /** Largest positive loadings first; zero loadings (genes outside the variable set) left out. */
  positive: LoadingGene[];
  /** Most negative first. */
  negative: LoadingGene[];
}

export interface PcaNeighbors {
  /** The representation the neighbour graph was built from (X_pca, X_pca_harmony, ...). */
  use_rep: string;
  /** PCs it used: the recorded n_pcs, else every column of use_rep; null when unknown. */
  n_pcs: number | null;
  n_neighbors: number | null;
  /** use_rep is the PCA or a correction of it with the same columns, so n_pcs marks the scree plot. */
  from_pca: boolean;
}

export interface PcaParams {
  n_comps: number;
  use_highly_variable: boolean | null;
  mask_var: string | null;
  zero_center: boolean | null;
}

export type PcaDiagnostics =
  | { available: false; reason: string }
  | {
      available: true;
      n_cells: number;
      n_pcs: number;
      n_genes_total: number;
      /** Genes with a non-zero loading (the variable genes); null without loadings. */
      n_genes_used: number | null;
      params: PcaParams;
      variance: number[];
      /** Each PC's share of the variance: of the total ("total") or, for a PCA without stored
       *  ratios, of the stored PCs' sum ("retained"). */
      variance_ratio: number[];
      cumulative: number[];
      ratio_basis: "total" | "retained";
      /** Sum of variance_ratio: what the stored PCs keep together. */
      retained: number;
      /** Kneedle elbow of the scree curve, 1-based. */
      elbow: number;
      /** Leading PCs reaching 90% of the retained variance. */
      pcs_90: number;
      neighbors: PcaNeighbors | null;
      loadings: PcLoadings[];
      /** How each number is computed, in one paragraph. */
      method: string;
    };

/** Variance per PC, elbow, loadings and the neighbour graph's PC count. */
export async function getPcaDiagnostics(datasetId: string, topK?: number): Promise<PcaDiagnostics> {
  const q = topK ? `?top_k=${topK}` : "";
  return apiFetch<PcaDiagnostics>(`/datasets/${datasetId}/pca/diagnostics${q}`);
}
