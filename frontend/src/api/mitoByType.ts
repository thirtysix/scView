import { apiFetch } from "./client";

/** One cell type's % mito and what each bound would remove (backend core/mito_by_type.py). */
export interface TypeMito {
  group: string;
  n_cells: number;
  median: number | null;
  mad: number | null;
  /** "q05", "q25", "q75", "q95" of % mito. */
  quantiles: Record<string, number>;
  /** Its own bound, median + 3 MADs; null: too few cells, no spread, or damaged. */
  bound: number | null;
  removed_global: number;
  removed_global_frac: number;
  removed_bound: number | null;
  removed_bound_frac: number | null;
  /** The floor: its own bound never below the ceiling (max of the two), and the cells above
   *  it; null without a bound of its own. Older backends omit them. */
  floor_bound?: number | null;
  removed_floor?: number | null;
  removed_floor_frac?: number | null;
  /** The global ceiling removes more than `loss_flag` of it, or sits below its median. */
  flagged: boolean;
  /** Median at or above 50 %: likely damaged cells, not a high-baseline type. */
  damaged: boolean;
  /** Filtered per type on this column: judged by its own bound ("own") or the global rule. */
  applied_rule?: "own" | "global" | null;
  applied_bound?: number | null;
  /** Filtered per type with the floor: which bound it used. */
  applied_floor?: "type" | "global" | "mixed" | null;
  /** Cells the filtering step's % mito rule actually removed (its record); null: not filtered. */
  removed_filter?: number | null;
  removed_filter_frac?: number | null;
  reasons: string[];
  note: string;
}

/** Per-type % mito bounds vs the global ceiling, and the rule filtering used. */
export interface MitoByType {
  column: string | null;
  reason: string | null;
  /** False when the dataset has no MT- genes: % mito can't be measured. */
  measurable: boolean;
  n_cells: number;
  n_labelled: number;
  ceiling: number | null;
  ceiling_source: "requested" | "recorded" | "default" | null;
  filter_mode: string | null;
  /** The recorded % mito rule ("global" | "per_type"), and the column per-type filtering used. */
  filter_mito_mode?: "global" | "per_type" | null;
  filter_type_column?: string | null;
  /** The per-type filtering was never stricter than the global rule. */
  filter_type_floor?: boolean;
  already_filtered: boolean;
  includes_removed?: number; // cells the filtering step removed, added back from its record
  /** Of those, cells given a group by expression (a clustering made after filtering). */
  assigned_removed?: number;
  mad_k: number;
  loss_flag: number;
  min_cells: number;
  types: TypeMito[];
  n_flagged: number;
  removed_global_total: number;
  removed_bound_total: number;
  removed_floor_total?: number;
  summary: string;
  note: string;
  rule: string;
  citations: string;
}

export async function getMitoByType(datasetId: string, column?: string, ceiling?: number): Promise<MitoByType> {
  const q = new URLSearchParams();
  if (column) q.set("column", column);
  if (ceiling !== undefined) q.set("ceiling", String(ceiling));
  const qs = q.toString();
  return apiFetch<MitoByType>(`/datasets/${datasetId}/qc/mito-by-type${qs ? `?${qs}` : ""}`);
}
