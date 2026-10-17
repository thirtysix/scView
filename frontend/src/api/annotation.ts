import { apiFetch } from "./client";

/** A cell's flag in an annotation confidence check (core/annotation_confidence.py). */
export type ConfidenceFlag = "confident" | "ambiguous" | "unsure" | "doesn't fit";

export interface ConfidenceLabelRow {
  label: string;
  n_cells: number;
  /** Shares of the label's cells with each flag (0-1) */
  confident: number;
  ambiguous: number;
  unsure: number;
  not_fit: number;
  /** Share of its held-out cells whose label is in their set; not guaranteed per label */
  coverage_held_out: number | null;
  /** The labels its flagged cells are nearest to, commonest first */
  confused_with: { label: string; n_cells: number; share: number }[];
}

export interface ConfidenceResult {
  column: string;
  alpha: number;
  score: string;
  classifier: string;
  representation: string;
  n_pcs: number;
  n_cells: number;
  n_checked: number;
  labels_not_checked: { label: string; n_cells: number }[];
  thresholds: number[];
  coverage: {
    target: number;
    held_out: number;
    n_held_out: number;
    calibration: number;
    n_calibration: number;
    /** Sampling spread expected of held_out */
    sd: number;
    within_sampling_error: boolean;
  };
  /** Share of cells whose most probable label is their own */
  accuracy: number;
  mean_set_size: number;
  counts: Record<ConfidenceFlag, number>;
  labels: ConfidenceLabelRow[];
  confusions: { label: string; other: string; n_cells: number; share: number }[];
  /** obs columns the check wrote */
  columns: { flag: string; fit: string; set_size: string; alt: string };
  unknown_column: string | null;
  /** The label column changed since the check ran */
  stale: boolean;
  note: string;
  elapsed_seconds: number;
  timestamp: string;
}

export interface ConfidenceInfo {
  available: boolean;
  /** Why not, when it can't run (e.g. no PCA) */
  reason: string | null;
  representation: string | null;
  n_pcs: number | null;
  default_alpha: number;
  min_alpha: number;
  max_alpha: number;
  min_label_cells: number;
  max_labels: number;
  classifiers: string[];
  note: string;
  /** What a check scored by CellTypist's probabilities measures */
  reference_note: string;
  /** Columns CellTypist annotated, which its probabilities can score: column -> model */
  reference_columns: Record<string, string>;
  checked_columns: string[];
  /** The last check of the asked-for column, or null */
  result: ConfidenceResult | null;
}

/** Whether the check can run, and the last check of `column` if there is one. */
export function getAnnotationConfidence(datasetId: string, column?: string): Promise<ConfidenceInfo> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  return apiFetch(`/datasets/${datasetId}/annotation/confidence${q}`);
}

/**
 * Check how well each cell's label in `column` fits, at error rate `alpha`,
 * scored by a classifier on the PCA (default) or by "celltypist", the
 * probabilities CellTypist kept when it made the column.
 */
export function runAnnotationConfidence(
  datasetId: string,
  column: string,
  alpha: number,
  classifier?: string,
): Promise<ConfidenceResult> {
  return apiFetch(`/datasets/${datasetId}/annotation/confidence`, {
    method: "POST",
    body: JSON.stringify(classifier ? { column, alpha, classifier } : { column, alpha }),
  });
}

/** Copy `column` with the cells its last check rejected set to "Unknown". */
export function makeUnknownColumn(
  datasetId: string,
  column: string,
): Promise<{ column: string; unknown_column: string; n_unknown: number; n_cells: number }> {
  return apiFetch(`/datasets/${datasetId}/annotation/confidence/unknown`, {
    method: "POST",
    body: JSON.stringify({ column }),
  });
}
