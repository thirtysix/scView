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

/** A CellTypist catalog model; `species` is read from its name and description. */
export interface CellTypistModel {
  model: string;
  description: string;
  species: "human" | "mouse" | "other";
}

export interface CellTypistModels {
  default: string;
  models: CellTypistModel[];
}

export function getCellTypistModels(): Promise<CellTypistModels> {
  return apiFetch("/annotation/celltypist-models");
}

/** One label of a CellTypist run (core/celltypist_fit.py). */
export interface CellTypistLabelRow {
  label: string;
  n_cells: number;
  /** Median of the cells' highest class probability */
  median_confidence: number | null;
  /** Percent of the label's cells below the low-confidence cut */
  pct_below: number | null;
  low_confidence: boolean;
  /** Percent of its cells whose own best class is this label (majority-vote runs) */
  pct_agree?: number;
}

export interface CellTypistSummary {
  column: string;
  confidence_column: string;
  percell_column: string | null;
  model: string | null;
  model_species: string | null;
  dataset_species: string | null;
  /** Share (0-1) of the model's genes in the dataset; null for runs before it was kept */
  gene_overlap: number | null;
  n_model_genes: number | null;
  n_shared_genes: number | null;
  n_cells: number;
  median_confidence: number | null;
  pct_below: number | null;
  n_low_labels: number;
  labels: CellTypistLabelRow[];
  /** Species mismatch, low gene overlap, low median confidence: plain sentences */
  cautions: string[];
  low_confidence: number;
  min_overlap: number;
  note: string;
}

export interface CellTypistInfo {
  /** Columns with a recorded CellTypist run */
  columns: string[];
  dataset_species: "human" | "mouse" | "unknown";
  result: CellTypistSummary | null;
}

/** The CellTypist runs of a dataset, and the summary of `column`'s if asked. */
export function getCellTypistSummary(datasetId: string, column?: string): Promise<CellTypistInfo> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  return apiFetch(`/datasets/${datasetId}/annotation/celltypist${q}`);
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

// ---------------------------------------------------------------------------
// Annotation cross-check (core/annotation_crosscheck.py): every map is keyed by label
// ---------------------------------------------------------------------------

/** Ref label -> other label -> value; zero entries are left out. */
export type LabelTable = Record<string, Record<string, number>>;

export interface CrosscheckContingency {
  /** Reference labels, in the column's order */
  rows: string[];
  /** The other column's labels, in its order */
  cols: string[];
  counts: LabelTable;
  /** Counts as a share of the reference label's cells */
  row_fraction: LabelTable;
}

export interface CrosscheckBestMatch {
  n_cells: number;
  jaccard: number;
  match: string;
  contained: number;
  purity: number;
  parts: number;
  status: "stable" | "splits" | "merges" | "dissolves";
}

export interface CrosscheckComparison {
  ari: number;
  ami: number;
  /** Cells with a label in both columns */
  n_cells: number;
  /** Share of those cells whose mapped label is the reference label */
  mapped_agreement: number | null;
  /** Other label -> the reference label most of its cells carry, and that share */
  mapping: Record<string, { to: string; share: number }>;
  exact_name: { n_shared_labels: number; shared_labels: string[]; agreement: number | null };
  /** Per reference label: its best match by Jaccard, and what became of it */
  best_match: Record<string, CrosscheckBestMatch>;
  contingency: CrosscheckContingency;
}

export interface CrosscheckAnnotator {
  /** Share of the label's cells this column maps to the label */
  agreement: number;
  disagrees: boolean;
  /** What the column calls the label's cells, commonest first */
  calls: { label: string; share: number; maps_to: string | null }[];
}

export interface CrosscheckLabelRow {
  n_cells: number;
  mean_agreement: number | null;
  disagreeing: string[];
  annotators: Record<string, CrosscheckAnnotator>;
}

export interface CrosscheckResult {
  columns: string[];
  reference: string;
  /** The reference's labels in its order (`labels` is keyed by them) */
  reference_labels: string[];
  n_cells: number;
  n_compared: number;
  pairwise: {
    ari: Record<string, Record<string, number | null>>;
    ami: Record<string, Record<string, number | null>>;
  };
  mean_agreement: number | null;
  /** Agreement value ("1", "0.5", ...) -> cells */
  agreement_counts: Record<string, number>;
  /** Same-name agreement after normalising case, spaces, "+" and plurals; apart from the mapped one */
  exact_name_agreement: number | null;
  consensus_counts: Record<string, number>;
  n_unresolved: number;
  /** Other column -> the reference against it */
  comparisons: Record<string, CrosscheckComparison>;
  /** Reference label -> its row */
  labels: Record<string, CrosscheckLabelRow>;
  note: string;
  unresolved_label: string;
  elapsed_seconds: number;
}

export interface CrosscheckInfo {
  /** Columns that look like annotations -> the method that made them */
  suggested: Record<string, string>;
  active_clustering: string | null;
  min_columns: number;
  max_columns: number;
  max_categories: number;
  default_name: string;
  saved: Record<
    string,
    { columns: { consensus: string; agreement: string }; compared: string[]; reference: string }
  >;
  note: string;
}

export function getCrosscheckInfo(datasetId: string): Promise<CrosscheckInfo> {
  return apiFetch(`/datasets/${datasetId}/annotation/crosscheck`);
}

/** Compare 2-6 label columns against `reference` (one of them). Changes nothing. */
export function runCrosscheck(datasetId: string, columns: string[], reference: string): Promise<CrosscheckResult> {
  return apiFetch(`/datasets/${datasetId}/annotation/crosscheck`, {
    method: "POST",
    body: JSON.stringify({ columns, reference }),
  });
}

export interface CrosscheckSaved {
  name: string;
  columns: { consensus: string; agreement: string };
  reference: string;
  compared: string[];
  n_unresolved: number;
  mean_agreement: number | null;
}

/** Write `{name}_consensus` and `{name}_agreement` to obs. */
export function saveCrosscheck(
  datasetId: string,
  columns: string[],
  reference: string,
  name: string,
): Promise<CrosscheckSaved> {
  return apiFetch(`/datasets/${datasetId}/annotation/crosscheck/save`, {
    method: "POST",
    body: JSON.stringify({ columns, reference, name }),
  });
}

// ---------------------------------------------------------------------------
// Marker-database check (core/marker_db.py): PanglaoDB + CellMarker 2.0
// ---------------------------------------------------------------------------

export interface MarkerDbSource {
  name: string;
  label: string;
  release: string;
  home: string;
  paper: string;
  doi: string;
  url: string;
  downloaded: boolean;
  fetched_at: string | null;
  /** The URL that answered (a mirror when the site was down), or "local file" */
  served_from: string | null;
  sha256: string | null;
  rows: number | null;
  types: { human: number; mouse: number } | null;
  download_bytes: number | null;
  /** Why the last download failed, while it is still missing */
  error: string | null;
  failed_at: string | null;
  /** False for CellMarker 2.0 while 3.0 is installed: not fetched, and not missing */
  needed?: boolean;
  /** Bytes so far while it downloads */
  progress?: { done: number; total: number } | null;
}

export type MarkerDbState = "ready" | "partial" | "missing" | "failed" | "disabled" | "downloading";

export interface MarkerDbStatus {
  state: MarkerDbState;
  /** The CellMarker release in use: "CellMarker 3.0", or "CellMarker 2.0" when 3.0 could not be fetched */
  cellmarker?: string | null;
  auto_download: boolean;
  sources: MarkerDbSource[];
  ontology: { downloaded: boolean; url: string; version: string | null; fetched_at: string | null; error: string | null };
  dataset_species?: "human" | "mouse" | "unknown";
}

export type MarkerDbVerdict = "supported" | "no_support" | "too_few" | "not_in_db" | "no_markers";

export interface MarkerDbType {
  cell_type: string;
  cl_id: string | null;
  sources: string[];
  p: number;
  /** Genes in both the group's markers and the type's database set (the dataset's spelling) */
  overlap: string[];
  /** The type's database genes measured here (its top 50 for the verdict, all for suggestions) */
  n_db_genes: number;
}

export interface MarkerDbLabel {
  label: string;
  /** The group's top 25 markers */
  markers: string[];
  match: {
    types: string[];
    how: "exact" | "cl" | "subset" | "cl-ancestor" | null;
    how_text: string | null;
    via_parent: boolean;
  };
  db_type: string | null;
  cl_id: string | null;
  sources: string[];
  p: number | null;
  overlap: string[];
  n_db_genes: number;
  verdict: MarkerDbVerdict;
  verdict_text: string;
  /** The 3 database types its markers fit best */
  suggestions: MarkerDbType[];
}

export interface MarkerDbResult {
  column: string;
  species: "human" | "mouse";
  species_note: string | null;
  labels: MarkerDbLabel[];
  verdict_counts: Record<MarkerDbVerdict, number>;
  n_measured: number;
  n_db_types: number;
  /** Databases behind the verdicts (PanglaoDB, CellMarker 2.0, ACT when downloaded) */
  verdict_sources: string[];
  /** Databases the suggestions rank (PanglaoDB + CellMarker 2.0) */
  suggestion_sources: string[];
  ontology: boolean;
  cellmarker?: string | null;
  sources: MarkerDbSource[];
  params: Record<string, number>;
  note: string;
  elapsed_seconds: number;
}

export function getMarkerDbStatus(datasetId: string): Promise<MarkerDbStatus> {
  return apiFetch(`/datasets/${datasetId}/annotation/marker-db/status`);
}

/** Check `column`'s labels against the marker databases (downloaded on first use). Changes nothing. */
export function runMarkerDbCheck(
  datasetId: string,
  column: string,
  opts: { species?: "human" | "mouse"; retry?: boolean } = {},
): Promise<MarkerDbResult> {
  return apiFetch(`/datasets/${datasetId}/annotation/marker-db/check`, {
    method: "POST",
    body: JSON.stringify({ column, ...opts }),
  });
}
