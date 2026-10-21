import { apiFetch } from "./client";

/** One mixture component of a group (core/qc_states.py). */
export interface QcStateComponent {
  /** "main", "low-RP high-mito", "low-RP" (no mito measured) or "second state". */
  label: string;
  n_cells: number;
  weight: number;
  /** Mean ribosomal-protein share of non-mito counts, %. */
  mean_rp_share: number | null;
  /** Mean % mito; null when the dataset has no MT- genes. */
  mean_pct_mt: number | null;
}

export interface QcStateGroup {
  name: string;
  n_cells: number;
  /** Cells with counts and finite shares (the ones fitted). */
  n_fitted: number;
  fitted: boolean;
  n_states: 1 | 2;
  /** BIC(1 component) - BIC(2): positive favours two. */
  delta_bic: number | null;
  /** Ashman's D between the two components. */
  separation: number | null;
  /** Label of the flagged state (two-state groups only). */
  flagged_label: string | null;
  /** Share of the group's fitted cells in the flagged state. */
  flagged_share: number | null;
  minor_weight: number | null;
  components: QcStateComponent[];
  /** Per condition (or sample) level: cells of this group, and the flagged share (null: one state). */
  per_condition: Record<string, { n: number; share: number | null }>;
  /** The rule that decided, with the numbers. */
  reason: string;
}

export interface QcStates {
  column: string | null;
  reason: "requested" | "celltype" | "clustering" | null;
  n_cells: number;
  n_unlabelled?: number;
  has_mt?: boolean;
  /** "rp_mito": 2-D fit; "rp_only": no MT- genes, fitted on the RP share alone. */
  fit?: "rp_mito" | "rp_only";
  note: string;
  rules?: { min_cells: number; min_delta_bic: number; min_separation: number; min_weight: number };
  condition?: { column: string; kind: "condition" | "sample"; levels: string[] } | null;
  groups: QcStateGroup[];
  two_state_groups: string[];
  overall?: { n_fitted: number; n_flagged: number; share: number | null };
  scatter?: {
    x: number[];
    /** null without mito: the plot is RP share by group. */
    y: number[] | null;
    group: number[];
    /** 0 = main, 1 = the group's flagged state. */
    state: number[];
    x_label: string;
    y_label: string | null;
    n_shown: number;
    max_per_group: number;
  };
  seconds?: number;
}

export interface QcStateLabelSaved {
  name: string;
  column: string;
  counts: Record<string, number>;
  two_state_groups: string[];
  fit: "rp_mito" | "rp_only";
}

/** GET /qc/states: 1 vs 2 states per group of `column` (default: cell type, else clustering). */
export function getQcStates(datasetId: string, column?: string): Promise<QcStates> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  return apiFetch(`/datasets/${datasetId}/qc/states${q}`);
}

/** POST /qc/states/label: save every cell's state as categorical obs column `name`. */
export function saveQcStateLabel(datasetId: string, column: string, name: string): Promise<QcStateLabelSaved> {
  return apiFetch(`/datasets/${datasetId}/qc/states/label`, {
    method: "POST",
    body: JSON.stringify({ column, name }),
  });
}
