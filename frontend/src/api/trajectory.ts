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
}

export interface TrendGenesResult {
  column: string;
  n_cells: number;
  n_genes_tested: number;
  min_detected: number;
  /** Where the expression came from (log-normalised .raw, ...) */
  expression: string;
  method: string;
  restrict: { column: string; groups: string[] } | null;
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
