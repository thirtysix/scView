import { apiFetch } from "./client";

/** One group's placement (its scored cells). */
export interface ReliabilityGroup {
  group: string;
  n: number;
  median: number;
  q1: number;
  q3: number;
  mean: number;
  /** Share of the group's cells below `low_threshold` */
  share_low: number;
  /** Median strict score: share of the k graph neighbours that are the k nearest on the plot */
  exact_median?: number | null;
}

export interface ReliabilityStats {
  n: number;
  median: number | null;
  q1: number | null;
  q3: number | null;
  mean: number | null;
  share_low: number | null;
}

/** backend core/embedding_reliability.py: summarise() */
export interface ReliabilitySummary {
  embedding: string;
  /** The columns scored, as plotted */
  dims: number[];
  n_dims: number;
  /** Graph neighbours per cell */
  k: number;
  /** Placement window: each cell's nearest `cells` cells on the plot (`share` of all cells) */
  window: {
    share: number;
    cells: number;
    /** "estimated": measured against a seeded sample of `ref_cells` cells (large datasets) */
    radius: "exact" | "estimated";
    ref_cells: number | null;
  };
  graph: {
    /** "stored": the graph the clustering used; "computed": an exact kNN built for this */
    source: "stored" | "computed";
    key: string | null;
    k: number;
    use_rep: string | null;
    n_pcs: number | null;
    metric: string;
  };
  n_cells: number;
  n_scored: number;
  /** Above the cap a seeded sample was scored; the other cells have no value */
  sampled: boolean;
  seed: number;
  /** window / (n - 1): the placement of a cell whose neighbours are a random draw */
  chance: number;
  /** k / (n - 1): the same for the strict exact-neighbour score */
  exact_chance: number;
  low_threshold: number;
  low_reason: string;
  notes: string[];
  method: string;
  seconds: number;
  /** Placement over the scored cells */
  overall: ReliabilityStats;
  /** The strict score (k graph neighbours among the k nearest on the plot), for context */
  exact: { median: number | null; q1: number | null; q3: number | null; mean: number | null; chance: number };
  /** The same score after one seeded shuffle of the coordinates */
  shuffled: { median: number | null; mean: number | null; seed: number };
  group_by: string | null;
  group_reason: string | null;
  /** Worst first */
  groups: ReliabilityGroup[];
  cached: boolean;
}

export interface ReliabilityResponse extends ReliabilitySummary {
  /** Placement, one per cell, in cell order; null for a cell left out of a sample */
  scores: (number | null)[];
}

export interface ReliabilityQuery {
  /** Two columns of a many-column embedding, as plotted (a PCA pair) */
  dims?: [number, number];
  groupBy?: string;
  k?: number;
}

function query(q: ReliabilityQuery): string {
  const p = new URLSearchParams();
  if (q.dims) p.set("dims", `${q.dims[0]},${q.dims[1]}`);
  if (q.groupBy) p.set("group_by", q.groupBy);
  if (q.k != null) p.set("k", String(q.k));
  return p.size ? `?${p}` : "";
}

export async function getEmbeddingReliability(
  datasetId: string,
  embedding: string,
  q: ReliabilityQuery = {},
): Promise<ReliabilityResponse> {
  return apiFetch<ReliabilityResponse>(
    `/datasets/${datasetId}/embeddings/${encodeURIComponent(embedding)}/reliability${query(q)}`,
  );
}

export async function getReliabilitySummary(
  datasetId: string,
  embedding: string,
  q: ReliabilityQuery = {},
): Promise<ReliabilitySummary> {
  return apiFetch<ReliabilitySummary>(
    `/datasets/${datasetId}/embeddings/${encodeURIComponent(embedding)}/reliability/summary${query(q)}`,
  );
}
