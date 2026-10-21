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
  /** The global ceiling removes more than `loss_flag` of it, or sits below its median. */
  flagged: boolean;
  /** Median at or above 50 %: likely damaged cells, not a high-baseline type. */
  damaged: boolean;
  reasons: string[];
  note: string;
}

/** Per-type % mito bounds vs the global ceiling (advisory: filtering is unchanged). */
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
  already_filtered: boolean;
  mad_k: number;
  loss_flag: number;
  min_cells: number;
  types: TypeMito[];
  n_flagged: number;
  removed_global_total: number;
  removed_bound_total: number;
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
