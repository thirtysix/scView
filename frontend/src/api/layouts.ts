import { streamEvents } from "./clustering";
import { apiFetch } from "./client";

// --- Layout stability: which group relations survive other seeds and neighbour
// counts (backend core/layout_stability.py, api/v1/layouts.py) ---

export interface LayoutRecorded {
  use_rep: string;
  n_pcs: number | null;
  metric: string;
  n_neighbors: number;
  min_dist: number | null;
  /** The stored UMAP curve, used when min_dist is not recorded */
  a: number | null;
  b: number | null;
  notes: string[];
}

export interface LayoutSpec {
  index: number;
  seed: number;
  n_neighbors: number;
  /** obsm key it is saved under ("X_umap_s1_k10") */
  key: string;
  label: string;
}

export interface LayoutPlan {
  recorded: LayoutRecorded;
  seeds: number[];
  n_neighbors: number[];
  /** Display order: one row per neighbour count, one column per seed */
  layouts: LayoutSpec[];
  n_cells: number;
  n_laid_out: number;
  sampled: boolean;
  group_by: string | null;
  /** The 2D embedding compared as the user's plot, or null */
  current: string | null;
  notes: string[];
}

export interface LayoutDone extends LayoutSpec {
  seconds: number;
  /** Equal to the user's plot (same coordinates) */
  same_as_current: boolean;
  /** Already saved as an embedding */
  exists: boolean;
}

export type RelationStatus = "every" | "some" | "never";
export type RelationFlag = "rare_in_plot" | "hidden_in_plot";

export interface LayoutRelation {
  a: string;
  b: string;
  n_a: number;
  n_b: number;
  /** Touching in each layout, in the plan's order */
  layouts: boolean[];
  n_touch: number;
  n_layouts: number;
  share: number | null;
  /** Touching in the user's plot; null without one */
  current: boolean | null;
  status: RelationStatus;
  flag: RelationFlag | null;
  /** Contact cells of a (towards b) and of b (towards a), per layout */
  contact_a: number[];
  contact_b: number[];
  current_contact: [number, number] | null;
}

export interface LayoutGroup {
  group: string;
  n_cells: number;
  /** Median placement per layout */
  placement: number[];
  current: number | null;
  min: number | null;
  max: number | null;
  range: number | null;
}

export interface LayoutSummary extends LayoutPlan {
  complete: boolean;
  stopped: boolean;
  n_done: number;
  seconds: number;
  done: LayoutDone[];
  rule: string;
  plot_k: number;
  min_contact: number;
  body_share: number;
  relations: LayoutRelation[];
  n_pairs: number;
  n_listed: number;
  counts: Partial<Record<RelationStatus | RelationFlag, number>>;
  groups: LayoutGroup[];
  n_groups?: number;
}

export interface LayoutsInfo {
  available: boolean;
  reason: string | null;
  n_cells: number;
  max_cells: number;
  max_layouts: number;
  first_run_seconds: number;
  rule: string;
  running: boolean;
  plan?: LayoutPlan;
  estimate_seconds?: number;
  /** The latest run's summary for this grouping, when one is held */
  result: LayoutSummary | null;
}

export interface LayoutStart extends LayoutPlan {
  estimate_seconds: number;
  first_run_seconds: number;
}

export interface LayoutEvent extends LayoutSpec {
  seconds: number;
  exists: boolean;
}

export interface LayoutThumb {
  index: number | "current";
  /** For the user's plot: which embedding */
  embedding?: string | null;
  seed: number | null;
  n_neighbors: number | null;
  key: string | null;
  label: string;
  /** obs indices of the cells drawn (the same in every layout) */
  cells: number[];
  x: number[];
  y: number[];
  n_laid_out: number;
  n_shown: number;
}

function query(params: Record<string, string | null | undefined>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v);
  return q.size ? `?${q}` : "";
}

/** Whether the check can run, its defaults and, when held, the latest run's summary. */
export function getLayoutsInfo(
  datasetId: string,
  opts: { groupBy?: string | null; embedding?: string | null } = {},
): Promise<LayoutsInfo> {
  return apiFetch(`/datasets/${datasetId}/layouts${query({ group_by: opts.groupBy, embedding: opts.embedding })}`);
}

/**
 * Lay the cells out under each seed x neighbour count, calling `onStart` once,
 * `onLayout` per layout and `onSummary` at the end. Rejects with the server's
 * reason; aborting `signal` stops reading and, on the server, the run.
 */
export async function runLayouts(
  datasetId: string,
  body: { group_by?: string | null; embedding?: string | null; seeds?: number[]; n_neighbors?: number[] },
  handlers: {
    onStart?: (s: LayoutStart) => void;
    onLayout?: (e: LayoutEvent) => void;
    onSummary?: (s: LayoutSummary) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/layouts/run-stream`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as LayoutStart);
      else if (event === "layout") handlers.onLayout?.(data as LayoutEvent);
      else if (event === "summary") handlers.onSummary?.(data as LayoutSummary);
    },
    "The layout check",
    handlers.signal,
  );
}

/** One layout's coordinates on the run's thumbnail cells (`"current"`: the user's plot). */
export function getLayoutThumb(
  datasetId: string,
  index: number | "current",
  opts: { maxPoints?: number; embedding?: string | null } = {},
): Promise<LayoutThumb> {
  const q = query({ max_points: opts.maxPoints ? String(opts.maxPoints) : null, embedding: opts.embedding });
  return apiFetch(`/datasets/${datasetId}/layouts/${index}${q}`);
}

/** Save one layout as an embedding (obsm X_umap_s<seed>_k<k>), once. */
export function keepLayout(
  datasetId: string,
  index: number,
): Promise<{ key: string; created: boolean; n_cells: number; recomputed: boolean; note: string | null }> {
  return apiFetch(`/datasets/${datasetId}/layouts/${index}/keep`, { method: "POST" });
}
