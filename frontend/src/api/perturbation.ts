import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** Where each cell's guide comes from (backend core/crispr.py). */
export type GuideSource = "cellranger" | "counts" | "column";

/** Which genes Mixscape tests per target (backend mixscape.DE_GENE_CHOICES). */
export type MixscapeDeGenes = "all" | "variable";

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
  /** Targets in the guide mapping that no cell had when the call was saved */
  targets_without_cells?: string[];
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

/** One row of a KO-share table: a target's cells with one guide, or in one replicate. */
export interface KoShareRow {
  target: string;
  guide?: string;
  replicate?: string;
  n_cells: number;
  n_ko: number;
  ko_fraction: number;
}

/** One group's shift of a protein against the controls (mean CLR difference, cell-level Wilcoxon). */
export interface ProteinShift {
  n: number;
  diff: number;
  p: number;
  padj: number | null;
}

/** One target x protein row of the protein check (backend core/perturbation_protein.py). */
export interface ProteinCheckRow {
  target: string;
  protein: string;
  ko_fraction: number;
  all: ProteinShift | null;
  ko: ProteinShift | null;
  np: ProteinShift | null;
  replicates?: { replicate: string; n: number; diff: number }[];
  n_replicates?: number;
  same_sign?: number;
  p_replicate?: number | null;
  padj_replicate?: number;
  note?: string;
  /** replicate: judged on the replicates' agreement; cell: on cells alone */
  unit: "replicate" | "cell";
  significant: boolean;
  /** a significant shift with (almost) no cells called KO on RNA */
  flag: boolean;
}

/** The protein check's values (backend perturbation_protein.LAYERS); auto picks by panel size. */
export type ProteinLayer = "protein_clr" | "protein_counts" | "protein_dsb";

/** One layer the check can use on this dataset, with why it cannot. */
export interface ProteinLayerOption {
  layer: ProteinLayer;
  label: string;
  available: boolean;
  reason: string | null;
  /** Compositional on this panel (per-cell CLR or DSB step II on a small panel) */
  compositional: boolean;
}

export interface ProteinCheck {
  /** The values tested; results before v0.80 have no other than protein_clr */
  layer: string;
  layer_label?: string;
  /** The shift's unit in words ("CLR", "log1p counts", "DSB") */
  unit_name?: string;
  /** What was asked: auto or a named layer (v0.80+) */
  layer_choice?: string;
  /** auto's pick on this dataset (v0.80+) */
  default_layer?: ProteinLayer;
  layers?: ProteinLayerOption[];
  /** CLR only: each target's shift of the per-cell CLR reference (mean log count over the panel) */
  reference_shift?: Record<string, number> | null;
  proteins: string[];
  n_proteins: number;
  control: string;
  n_control: number;
  replicate_column: string | null;
  min_cells: number;
  ko_share_none: number;
  alpha: number;
  n_pairs: number;
  rows: ProteinCheckRow[];
  flags: { target: string; protein: string; diff: number; unit: string; ko_fraction: number; text: string }[];
  notes: string[];
  default_protein: string | null;
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
  /** Labels left out as not targets ("Unassigned", "multiple", a second control label) */
  excluded?: Record<string, { n_cells: number; reason: string }>;
  /** The obs columns behind per_guide / per_replicate (null: none found or given) */
  guide_column?: string | null;
  replicate_column?: string | null;
  /** KO share per target x guide, and per target x replicate (target cells only) */
  per_guide?: KoShareRow[];
  per_replicate?: KoShareRow[];
  /** Protein shift per target and protein-only flags (null: no protein modality) */
  protein?: ProteinCheck | null;
  lda:{ x: number[]; y: number[]; label: string[]; n_cells: number; n_shown: number; dims: number } | null;
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
  mixscape_defaults: Record<string, number | string>;
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
  /** Targets at least one cell has (the control left out) */
  targets: string[];
  /** Targets whose guides no cell carries at these settings (a vector or pilot guide) */
  targets_without_cells?: { target: string; guides: string[] }[];
  counts: AssignmentCounts;
  per_target: { target: string; n_cells: number; control: boolean; guides: string[] }[];
}

export interface MixscapeBody {
  labels?: string;
  control?: string;
  split_by?: string | null;
  lda?: boolean;
  /** all: every gene, Seurat RunMixscape's DE rule (default); variable: the variable genes (scView 0.77) */
  de_genes?: MixscapeDeGenes;
  /** The protein check's values; auto (default) by panel size */
  protein_layer?: "auto" | ProteinLayer;
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

/** The last Mixscape run's protein check on another layer (not saved: the run keeps its own). */
export function getMixscapeProtein(datasetId: string, layer: ProteinLayer | "auto"): Promise<ProteinCheck> {
  return apiFetch(`/datasets/${datasetId}/perturbation/mixscape/protein?layer=${encodeURIComponent(layer)}`);
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
