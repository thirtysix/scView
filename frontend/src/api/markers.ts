/** One marker gene of one group (backend GET /datasets/{id}/markers?format=json). */
export interface MarkerGene {
  gene: string;
  group: string;
  /** null when scanpy's fold change was NaN (e.g. a gene absent from both sides). */
  logfoldchange: number | null;
  pval: number;
  pval_adj: number;
  /** Fraction of the group's cells that detect the gene. */
  pct_in?: number;
  /** Fraction of all other cells, pooled. */
  pct_out?: number;
  score?: number;
  /** Tau over the groups' mean expression: 0 even across groups, 1 one group only. */
  tau?: number | null;
  /** Highest detection fraction among the other groups, and which group that is. */
  pct_max_other?: number | null;
  max_other_group?: string | null;
  /** pct_in / pct_max_other; null when infinite (no other group detects it) or unknown. */
  pct_ratio?: number | null;
}

export interface MarkersResponse {
  groups: string[];
  markers: Record<string, MarkerGene[]>;
}
