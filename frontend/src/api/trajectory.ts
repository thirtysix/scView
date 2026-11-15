import { apiFetch } from "./client";

/** A saved PAGA graph between the groups of one column (core/trajectory.py). */
export interface PagaRecord {
  groupby: string;
  groups: string[];
  /** Cells per group, in `groups` order */
  sizes: number[];
  /** [i, j, connectivity] for every pair with a non-zero connectivity, i < j */
  edges: [number, number, number][];
  /** [i, j] edges of PAGA's maximum spanning tree */
  tree: [number, number][];
  /** Which neighbour graph it was computed on */
  graph: string;
  n_cells: number;
}

export interface PagaView {
  paga: PagaRecord | null;
  /** Each group's median position on the embedding asked for ([x, y] or [x, y, z]) */
  positions: (number[] | null)[] | null;
  /** Why the graph no longer matches its column (relabelled, re-clustered), else null */
  stale: string | null;
  /** The group holding the pseudotime root cell, where a branch starts */
  root_group: string | null;
}

export function getPaga(datasetId: string, embedding: string): Promise<PagaView> {
  return apiFetch(`/datasets/${datasetId}/trajectory/paga?embedding=${encodeURIComponent(embedding)}`);
}

export function computePaga(datasetId: string, groupby: string): Promise<PagaRecord> {
  return apiFetch(`/datasets/${datasetId}/trajectory/paga`, {
    method: "POST",
    body: JSON.stringify({ groupby }),
  });
}

/** One gene whose expression changes along pseudotime (core/pseudotime_genes.py). */
export interface TrendGene {
  gene: string;
  f_stat: number | null;
  pval: number | null;
  pval_adj: number | null;
  r2: number | null;
  /** Fitted max - min, log-normalised units */
  amplitude: number | null;
  /** Pseudotime of the bin where the fitted trend peaks */
  peak_pseudotime: number | null;
  pattern: "rising" | "falling" | "transient";
  n_detected: number;
  /** Mean expression per bin (`bins`) */
  mean: number[];
  /** The spline's fitted mean per bin */
  fitted: number[];
  /** Branch-specific ranking: how far the branch's own cells sit above the other branches' curve (log) */
  branch_diff?: number | null;
  /** Mean expression of the branch's own cells, and of the other branches' cells */
  mean_branch?: number | null;
  mean_others?: number | null;
}

/** How a branch was split for the branch-specific ranking (core/pseudotime_genes.branch_split). */
export interface BranchSplit {
  column: string;
  path: string[];
  start: string;
  end: string;
  /** Groups the branch shares with the paths to the tree's other ends */
  trunk: string[];
  /** The branch's own groups */
  own: string[];
  /** The other branches' groups */
  others: string[];
  n_own_cells: number;
  n_trunk_cells: number;
  n_other_cells: number;
}

export type RankBy = "trend" | "branch_specific";

export interface TrendGenesResult {
  column: string;
  n_cells: number;
  n_genes_tested: number;
  min_detected: number;
  /** Where the expression came from (log-normalised .raw, ...) */
  expression: string;
  method: string;
  restrict: { column: string; groups: string[] } | null;
  rank_by?: RankBy;
  /** Branch-specific ranking: genes tested that sit higher on the branch, and the split */
  n_higher?: number;
  branch?: BranchSplit;
  /** Equal-count bins along pseudotime: each bin's median pseudotime and size */
  bins: { pseudotime: number[]; n_cells: number[] };
  /** Most changing first */
  genes: TrendGene[];
}

export interface TrendGenesRequest {
  n_top?: number;
  n_bins?: number;
  /** Only the cells in `groups` of this column (a branch) */
  groups_column?: string;
  groups?: string[];
  /** trend (default): change along the cells; branch_specific: higher on the branch than the others */
  rank_by?: RankBy;
}

export function rankTrendGenes(
  datasetId: string,
  column: string,
  body: TrendGenesRequest,
  signal?: AbortSignal,
): Promise<TrendGenesResult> {
  return apiFetch(`/datasets/${datasetId}/trajectory/${encodeURIComponent(column)}/trend-genes`, {
    method: "POST",
    body: JSON.stringify(body),
    signal,
  });
}
