import { apiFetch } from "./client";

/** One cluster's QC numbers + the flags it earned against the other clusters. */
export interface ClusterQc {
  group: string;
  n_cells: number;
  fraction: number;
  median_genes: number | null;
  median_counts: number | null;
  median_pct_mt: number | null;
  median_pct_ribo: number | null;
  doublet_fraction: number | null;
  median_doublet_score: number | null;
  flags: ClusterFlag[];
  reasons: string[]; // one plain-language sentence per flag, with the numbers
}

export type ClusterFlag = "low_quality" | "high_mito" | "low_genes" | "doublet_rich" | "small";

/** Per-cluster QC for one grouping column (deterministic, no LLM). */
export interface ClusterQcReport {
  column: string | null;
  n_cells: number;
  n_groups: number;
  metrics: string[];
  typical: Record<string, number>; // median across clusters, keyed like ClusterQc fields
  clusters: ClusterQc[];
  flag_help: Record<string, string>;
  computed_on_demand: boolean;
  note: string;
}

export async function getClusterQc(datasetId: string, groupby?: string): Promise<ClusterQcReport> {
  const q = groupby ? `?groupby=${encodeURIComponent(groupby)}` : "";
  return apiFetch<ClusterQcReport>(`/datasets/${datasetId}/cluster-qc${q}`);
}
