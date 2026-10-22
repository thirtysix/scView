/** Markers the backend keeps per group (`GET /markers` default `n_genes`): the
 *  highest by scanpy's rank_genes_groups Wilcoxon score. Every gene is tested. */
export const MARKERS_PER_GROUP = 100;

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
  /** Set when there are none (200, no groups): why, in a sentence the UI shows
   *  (an unprocessed dataset: "No marker genes yet: run the pipeline or pick a column."). */
  reason?: string;
}

/** Shown when an empty marker response carries no reason of its own (older backends 404'd). */
export const NO_MARKERS_REASON = "No marker genes yet: run the pipeline or pick a column.";

/** The reason a marker response has no markers, or null when it has some. */
export function noMarkersReason(data: MarkersResponse | null | undefined): string | null {
  if (!data || !Array.isArray(data.groups)) return NO_MARKERS_REASON;
  return data.groups.length === 0 ? (data.reason ?? NO_MARKERS_REASON) : null;
}
