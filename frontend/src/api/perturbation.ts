import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** Where each cell's guide comes from (backend core/crispr.py). */
export type GuideSource = "cellranger" | "counts" | "column";

export interface GuideColumnCandidate {
  column: string;
  n_values: number;
  /** The control-like value found in it, if any */
  control: string | null;
  named_like: boolean;
}

export interface AssignmentCounts {
  assigned: number;
  control: number;
  multiple: number;
  none: number;
}

export interface SavedAssignment {
  source: GuideSource;
  source_label: string;
  column: string | null;
  params: Record<string, number | string | null>;
  control: string;
  /** Guide -> target where the user changed the automatic mapping */
  mapping: Record<string, string>;
  columns: { guide_call: string; perturbation: string; n_guides: string };
  when: string;
  n_cells_at_call: number;
  counts: AssignmentCounts;
  n_targets: number;
  per_target: { target: string; n_cells: number }[];
}

export interface PosteriorHist {
  edges: number[];
  ko: number[];
  np: number[];
}

export interface MixscapeTarget {
  target: string;
  n_cells: number;
  n_ko: number;
  n_np: number;
  ko_fraction: number;
  n_de_genes: number;
  top_de_genes: { gene: string; logfc: number; pval_adj: number }[];
  rounds: number;
  note?: string;
  posterior_hist?: PosteriorHist;
}

export interface MixscapeResult {
  params: Record<string, unknown>;
  labels: string;
  control: string;
  split_by: string | null;
  n_cells: number;
  n_control: number;
  n_targets: number;
  n_genes: number;
  gene_source: string;
  expression_source: string;
  targets: MixscapeTarget[];
  warnings: string[];
  lda: { x: number[]; y: number[]; label: string[]; n_cells: number; n_shown: number; dims: number } | null;
  seconds: number;
  created: string;
  n_cells_total: number;
  summary: { n_target_cells: number; n_ko: number; ko_fraction: number; n_targets_with_ko: number };
}

export interface PerturbationSummary {
  n_cells: number;
  guides: string[];
  has_counts: boolean;
  has_cellranger: boolean;
  columns: GuideColumnCandidate[];
  obs_columns: string[];
  suggested: { source: GuideSource | null; column: string | null };
  defaults: { min_umi: number; min_ratio: number };
  sources: { id: GuideSource; label: string }[];
  call: SavedAssignment | null;
  mixscape: MixscapeResult | null;
  /** The run saved in the dataset, when its full result is no longer kept */
  mixscape_saved: Omit<MixscapeResult, "lda"> | null;
  mixscape_defaults: Record<string, number>;
  running: boolean;
}

export interface AssignBody {
  source?: GuideSource | null;
  column?: string | null;
  min_umi?: number;
  min_ratio?: number;
  mapping?: Record<string, string>;
  control?: string | null;
}

export interface AssignmentPreview {
  source: GuideSource;
  source_label: string;
  column: string | null;
  n_cells: number;
  guides: string[];
  mapping: Record<string, string>;
  auto_mapping: Record<string, string>;
  control: string | null;
  targets: string[];
  counts: AssignmentCounts;
  per_target: { target: string; n_cells: number; control: boolean; guides: string[] }[];
}

export interface MixscapeBody {
  labels?: string;
  control?: string;
  split_by?: string | null;
  lda?: boolean;
}

export interface MixscapeProgress {
  step: string;
  done: number;
  total: number;
  message: string;
}

export function getPerturbation(datasetId: string): Promise<PerturbationSummary> {
  return apiFetch(`/datasets/${datasetId}/perturbation`);
}

export function previewAssignment(datasetId: string, body: AssignBody): Promise<AssignmentPreview> {
  return apiFetch(`/datasets/${datasetId}/perturbation/preview`, { method: "POST", body: JSON.stringify(body) });
}

export function saveAssignment(datasetId: string, body: AssignBody): Promise<PerturbationSummary> {
  return apiFetch(`/datasets/${datasetId}/perturbation/assign`, { method: "POST", body: JSON.stringify(body) });
}

/** Run Mixscape (streamed); resolves with the saved result. Aborting `signal` stops
 *  the run and nothing is saved. */
export async function runMixscape(
  datasetId: string,
  body: MixscapeBody,
  handlers: { onProgress?: (p: MixscapeProgress) => void; signal?: AbortSignal } = {},
): Promise<MixscapeResult> {
  let result: MixscapeResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/perturbation/mixscape/run-stream`,
    body,
    (event, data) => {
      if (event === "progress") handlers.onProgress?.(data as MixscapeProgress);
      else if (event === "result") result = data as MixscapeResult;
    },
    "Mixscape",
    handlers.signal,
  );
  if (!result) throw new Error("Mixscape ended without a result; nothing was saved.");
  return result;
}
