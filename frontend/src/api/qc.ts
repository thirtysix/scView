import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";
import { thresholdMode } from "@/lib/qcThresholds";
import type { ChatSource } from "./assistant";
import type { MitoMode, MitoTypesTable } from "./mitoFilter";
import type { DoubletAgreement, DoubletChoices, DoubletSource, DoubletSourceInfo } from "./doubletSource";

/** One cluster's QC numbers + the flags it earned against the other clusters. */
export interface ClusterQc {
  group: string;
  n_cells: number;
  fraction: number;
  median_genes: number | null;
  median_counts: number | null;
  median_pct_mt: number | null;
  median_pct_ribo: number | null;
  doublet_fraction: number | null;
  median_doublet_score: number | null;
  /** Median dissociation-stress score, when the dataset has one. */
  median_stress_score?: number | null;
  /** Stress-signature genes among the cluster's top markers. */
  stress_markers?: string[];
  /** Mean % of reads from haemoglobin genes (mean: most cells have none). */
  mean_pct_hb?: number | null;
  /** Red-cell lineage by its own markers (or a red-cell haemoglobin level). */
  erythroid?: boolean;
  /** Haemoglobin / erythroid genes among its top markers. */
  erythroid_markers?: string[];
  /** Share of its cells with a clear S or G2M score (outside the near-zero noise band). */
  cycling_fraction?: number | null;
  /** Median % of a cell's counts from MALAT1 (nuclear-retained); null when the gene is absent. */
  median_pct_malat1?: number | null;
  /** Median of the author's intronic / unspliced share column, in % (never computed here). */
  median_pct_intronic?: number | null;
  /** Top markers whose signal in the other clusters looks like ambient RNA (core/ambient.py). */
  ambient_markers?: string[];
  flags: ClusterFlag[];
  reasons: string[]; // one plain-language sentence per flag, with the numbers
  /** Not a problem, but worth saying (an erythroid cluster's haemoglobin is real). */
  notes?: string[];
}

export type ClusterFlag =
  | "low_quality"
  | "high_mito"
  | "low_genes"
  | "doublet_rich"
  | "stress_driven"
  | "hb_ambient"
  | "cycling"
  | "low_malat1"
  | "small";

/** A technical score that tracks one of the clustering's leading PCs (backend core/qc_drivers.py). */
export interface QcDriver {
  score: string;
  label: string;
  pc: number; // 1-based
  r: number;
  /** The same correlation inside clusters (cluster means removed). */
  r_within: number | null;
  pc_share: number;
  /** True: the score moves along the PC inside clusters too (an axis of the embedding). */
  inside_clusters: boolean;
  reason: string;
}

/** Per-cluster QC for one grouping column (deterministic, no LLM). */
export interface ClusterQcReport {
  column: string | null;
  n_cells: number;
  n_groups: number;
  metrics: string[];
  typical: Record<string, number>; // median across clusters, keyed like ClusterQc fields
  clusters: ClusterQc[];
  flag_help: Record<string, string>;
  computed_on_demand: boolean;
  note: string;
  /** Haemoglobin RNA spread across the non-erythroid clusters (ambient RNA); "" if none. */
  ambient_hb?: string;
  drivers?: QcDriver[];
  /** The technical scores looked at, and the space and number of PCs they were checked against. */
  drivers_checked?: string[];
  drivers_space?: string | null;
  drivers_n_pcs?: number;
  /** The median cell's MALAT1 share (%): low_malat1 compares a cluster's median with it. */
  dataset_median_pct_malat1?: number | null;
  /** The obs column the intronic / unspliced share came from, when obs has one. */
  intronic_column?: string | null;
  /** Whose doublet calls `doublet_fraction` counts: "cell hashing", "Scrublet", ... */
  doublet_source?: string | null;
  /** The ambient markers come from the soup share alone (no raw matrix attached). */
  ambient_markers_approximate?: boolean;
  /** What an ambient marker means; "" when no cluster has one. */
  ambient_markers_help?: string;
}

export async function getClusterQc(datasetId: string, groupby?: string): Promise<ClusterQcReport> {
  const q = groupby ? `?groupby=${encodeURIComponent(groupby)}` : "";
  return apiFetch<ClusterQcReport>(`/datasets/${datasetId}/cluster-qc${q}`);
}

/** QC distributions per group of one obs column (backend core/qc_groups.py). */
export interface QcByGroup {
  /** null: no column named and no sample / batch / clustering column found. */
  column: string | null;
  /** Why this column: picked as the donor/sample, batch or clustering column, or asked for. */
  reason: "donor" | "batch" | "clustering" | "requested" | null;
  n_cells: number;
  n_groups: number;
  max_groups: number;
  n_unlabelled?: number;
  /** False when the dataset has no MT- genes: % mito is then left out. */
  has_mt?: boolean;
  computed_on_demand?: boolean;
  n_quantiles?: number;
  groups: { name: string; n_cells: number }[];
  /** metric -> group -> the group's quantiles (all its values when it has fewer cells). */
  metrics: Record<string, Record<string, number[]>>;
  /** A stratified sample; `group` indexes `groups`. */
  scatter: {
    x: number[];
    y: number[];
    group: number[];
    x_label: string;
    y_label: string;
    n_shown: number;
  } | null;
  note?: string;
}

/** An API error as the server's own reason (FastAPI's `detail`), when it gave one. */
function withDetail(e: unknown): unknown {
  const body = e instanceof Error ? e.message.replace(/^API error \d+: /, "") : "";
  let detail = "";
  try {
    detail = String(JSON.parse(body).detail ?? "");
  } catch {
    /* not JSON: keep the original error */
  }
  return detail ? new Error(detail) : e;
}

/** QC by group; with no column the backend picks the donor, else batch, else clustering
 *  column. A refused column (too many groups) throws with the server's reason. */
export async function getQcByGroup(datasetId: string, column?: string): Promise<QcByGroup> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  try {
    return await apiFetch<QcByGroup>(`/datasets/${datasetId}/qc/by-group${q}`);
  } catch (e) {
    throw withDetail(e);
  }
}

/** `doublet_channel_key` for all cells as one channel (backend doublets.ALL_CELLS). */
export const ALL_CELLS_CHANNEL = "__all__";
/** Where the doublet step writes the channels it read from the barcodes. */
export const CAPTURE_CHANNEL_COLUMN = "capture_channel";

/** The capture channels the doublet step runs Scrublet in, each with its expected rate
 *  (backend core/doublets.py): a rule of thumb, ~0.8 % per 1,000 cells captured in one
 *  channel. */
export interface DoubletRateSuggestion {
  /** The share of all cells expected to be doublets (cell-weighted mean of the channels'). */
  rate: number;
  /** Scrublet runs once per channel (false: all cells as one channel). */
  per_channel?: boolean;
  /** The obs column of the channels (`capture_channel` when read from the barcodes);
   *  null = all cells taken as one channel. */
  column: string | null;
  /** requested | requested_all | channel_column | barcode_prefix | barcode_suffix |
   *  barcode_prefix_suffix | donor | batch | none */
  reason: string | null;
  reason_text?: string;
  /** Read from the barcodes: the step writes them to `capture_channel`. */
  derived?: boolean;
  /** A caveat (donors or batches taken as channels are a guess). */
  note?: string | null;
  /** Candidate columns passed over, and why. */
  skipped?: string[];
  n_cells: number;
  n_channels?: number;
  largest_channel?: number;
  /** Cells not scored: in channels under `min_channel_cells`, or with no channel. */
  n_unscored?: number;
  min_channel_cells?: number;
  channels: { name: string; n_cells: number; rate: number; scored: boolean }[];
  basis: string;
}

/** One capture channel on the doublet card. */
export interface DoubletChannelCalls {
  name: string;
  n_cells: number;
  n_scored: number;
  /** At the threshold shown; with none, the calls there are now (Scrublet's per channel). */
  n_called: number | null;
  fraction: number;
  /** The rate the run used in this channel (or would use). */
  expected_rate: number;
  /** The threshold Scrublet picked in this channel (null: none found, or not run). */
  scrublet_threshold: number | null;
  /** scored | too_small | failed | not_scored */
  status: string;
  /** Cells Scrublet's own threshold calls here now (null: no threshold of its own). Missing
   *  from an older backend. */
  scrublet_called?: number | null;
  /** That threshold called almost nothing in a channel of some size: it may have missed. */
  few_called?: boolean;
  /** For a few_called channel: its scores on the card's histogram bins. */
  hist?: number[];
}

/** The doublet-score histogram and the calls a threshold makes. Without scores
 *  (`available` false) only the suggested expected rate is filled in. */
export interface DoubletSummary {
  available: boolean;
  n_cells: number;
  suggestion: DoubletRateSuggestion;
  n_scored?: number;
  hist?: { bin_edges: number[]; counts: number[] };
  median?: number | null;
  /** The threshold Scrublet picked (null when it ran per channel or found none). */
  scrublet_threshold?: number | null;
  /** Cells with no score: their channel was too small for Scrublet (or failed), or
   *  Scrublet dropped them for under 3 genes. Never called doublets. */
  n_unscored?: number;
  n_unscored_small_channels?: number;
  min_channel_cells?: number;
  /** The last run scored each channel alone (more than one). */
  per_channel?: boolean;
  /** The last run's channels: their column (null: all cells as one) and why. */
  channel_column?: string | null;
  channel_reason?: string | null;
  channel_reason_text?: string | null;
  /** The threshold applied over Scrublet's call, if any. */
  applied?: { threshold: number; n_called: number } | null;
  /** Scrublet's own call is kept in `predicted_doublet_scrublet`. */
  scrublet_call_kept?: boolean;
  /** Cells `predicted_doublet` calls doublets now. */
  n_called_now?: number | null;
  /** The expected rate the last doublet step ran with. */
  expected_rate_used?: number | null;
  /** The threshold these counts are for (null: none known yet). */
  threshold?: number | null;
  n_called?: number | null;
  fraction_called?: number | null;
  /** Per capture channel (empty for one channel of every cell). */
  channels?: DoubletChannelCalls[];
  /** The column of those rows; whether they are the last run's channels. */
  channels_column?: string | null;
  channels_recorded?: boolean;
  /** Channels whose automatic threshold called almost nothing (under `few_called_share` of
   *  a channel of `few_called_min_cells` or more). */
  few_called?: string[];
  few_called_share?: number;
  few_called_min_cells?: number;
  /** What the doublet views use (cell hashing when run, else Scrublet). */
  source?: DoubletSourceInfo | null;
  /** Cell hashing x Scrublet at this threshold, when the dataset has both. */
  agreement?: DoubletAgreement | null;
}

/** `channel`: the doublet step's `doublet_channel_key` (an obs column, or
 *  `ALL_CELLS_CHANNEL`); empty: the last run's channels, else the step's own choice. */
export async function getDoublets(
  datasetId: string,
  threshold?: number | null,
  channel?: string,
): Promise<DoubletSummary> {
  const params = new URLSearchParams();
  if (threshold !== undefined && threshold !== null) params.set("threshold", String(threshold));
  if (channel) params.set("column", channel);
  const q = params.toString() ? `?${params}` : "";
  try {
    return await apiFetch<DoubletSummary>(`/datasets/${datasetId}/qc/doublets${q}`);
  } catch (e) {
    throw withDetail(e);
  }
}

/** Call doublets above `threshold` (null: back to Scrublet's own call) and save. */
export async function applyDoubletThreshold(datasetId: string, threshold: number | null): Promise<DoubletSummary> {
  try {
    return await apiFetch<DoubletSummary>(`/datasets/${datasetId}/qc/doublets/threshold`, {
      method: "POST",
      body: JSON.stringify({ threshold }),
    });
  } catch (e) {
    throw withDetail(e);
  }
}

export type QcThresholdMode = "mad" | "fixed";

/** The Cell/Gene Filtering step's cell thresholds (pipeline params); 0 turns
 *  min/max genes and min counts off. % mito is applied only when measurable.
 *  `qc_threshold_mode` "mad" judges each sample's outliers instead of the genes /
 *  counts cut-offs (max % mito is then a ceiling); a caller that sets no mode gets
 *  the fixed cut-offs, as before MAD mode existed. */
export interface FilterThresholds {
  min_genes: number;
  max_genes: number;
  min_counts: number;
  max_pct_mt: number;
  drop_doublets: boolean;
  qc_threshold_mode?: QcThresholdMode;
  /** MADs below the median for counts and genes (log scale); above too with `qc_mad_upper`. */
  qc_mad_k?: number;
  /** MADs above the median for % mito. */
  qc_mad_k_mt?: number;
  /** Also cut counts / genes above the median + k MADs (opt-in: trims large cell types). */
  qc_mad_upper?: boolean;
  /** The samples for MAD mode; "" = the one the backend detects. */
  qc_sample_column?: string;
  /** "per_type": each cell type's own % mito bound instead of the global rule. */
  qc_mito_mode?: MitoMode;
  /** The cell types for per-type bounds; "" = the backend's choice. */
  qc_mito_type_column?: string;
  /** Per type: never stricter than the global rule (a type's bound = max(own, global)). */
  qc_mito_type_floor?: boolean;
  /** Whose doublets "drop doublets" drops; "auto" = cell hashing when run, else Scrublet. */
  qc_doublet_source?: DoubletSource;
  /** Also drop cell hashing's negatives (no tag above background). */
  drop_hash_negatives?: boolean;
}

export type FilterCriterion =
  | "min_genes"
  | "max_genes"
  | "min_counts"
  | "max_counts"
  | "max_pct_mt"
  | "doublets"
  | "hash_negatives";

/** A sample's QC medians set it apart from the other samples (backend qc_groups.sample_summary). */
export type SampleFlag = "few_cells" | "low_genes" | "high_genes" | "low_counts" | "high_counts" | "high_mito";

export interface FilterGroupLoss {
  name: string;
  n_cells: number;
  n_removed: number;
  fraction: number;
  /** Cells each threshold removes from the group on its own. */
  by_criterion: Partial<Record<FilterCriterion, number>>;
  /** Samples only: the median per cell (no % mito key when it can't be measured). */
  median_genes?: number | null;
  median_counts?: number | null;
  median_pct_mt?: number | null;
  flags?: SampleFlag[];
  /** One sentence per flag, with the numbers. */
  reasons?: string[];
}

export interface FilterGrouping {
  column: string;
  reason: string | null;
  groups: FilterGroupLoss[];
  /** Set when the column has too many groups to list. */
  note?: string;
  /** Samples only: the median of the sample medians, keyed like the group fields. */
  typical?: Partial<Record<"median_genes" | "median_counts" | "median_pct_mt", number>>;
  flag_help?: Partial<Record<SampleFlag, string>>;
  /** Enough samples (3+) to say which stands out. */
  compared?: boolean;
}

/** One sample's MAD bounds, in counts, genes and % (null: none, its MAD was 0). */
export interface MadSampleBounds {
  name: string;
  n_cells: number;
  /** Too few cells for bounds of its own: judged against all cells pooled. */
  pooled: boolean;
  min_counts: number | null;
  max_counts: number | null;
  min_genes: number | null;
  max_genes: number | null;
  /** The lower of median + k_mt MADs and the ceiling; null when mito isn't measurable. */
  max_pct_mt: number | null;
  /** Metrics whose MAD was 0 in this sample ("total_counts" | "n_genes" | "pct_mt"). */
  skipped: string[];
  n_removed?: number;
}

/** MAD mode's per-sample thresholds (backend qc_filter.MadBounds.to_dict). */
export interface MadThresholds {
  /** null: no sample column, all cells are one pooled group. */
  column: string | null;
  reason: string | null;
  k: number;
  k_mt: number;
  /** The counts / genes upper bounds were on (null max_* bounds when off). */
  upper?: boolean;
  ceiling_pct_mt: number | null;
  min_sample_cells: number;
  samples: MadSampleBounds[];
  pooled: Omit<MadSampleBounds, "name" | "pooled">;
  n_unlabelled: number;
  /** The run whose recorded thresholds these are (a same-settings re-run), if any. */
  reused_from?: string | null;
}

/** Removed ~ condition in one group (backend qc_filter.rate_test): over the
 *  conditions with 10+ cells there, chi-square on the 2 x k table. */
export interface RemovalTest {
  compared: string[];
  high: string;
  high_rate: number;
  low: string;
  low_rate: number;
  /** Highest minus lowest removal rate, in percentage points. */
  max_diff_pp: number;
  /** null when nothing or everything was removed. */
  p: number | null;
  statistic: number | null;
  dof: number | null;
  /** Some expected count is under 5: the chi-square p is approximate. */
  low_expected: boolean;
}

/** One label group (or all cells) in the removal audit. */
export interface RemovalRow {
  /** null: the all-cells row. */
  name: string | null;
  n_cells: number;
  n_removed: number;
  fraction: number;
  by_criterion: Partial<Record<FilterCriterion, number>>;
  /** Per condition, in `condition.names` order (absent without a condition column). */
  conditions?: {
    n_cells: number[];
    n_removed: number[];
    by_criterion: Partial<Record<FilterCriterion, number[]>>;
  };
  /** null: fewer than two conditions with 10+ cells. */
  test?: RemovalTest | null;
  /** Recorded audit only: the removed cells' own medians. */
  removed_median?: { total_counts: number | null; n_genes: number | null; pct_mt?: number | null } | null;
}

/** A flag in the removal audit, shaped as the preview's warnings. */
export interface RemovalFlag {
  kind: "loss" | "condition";
  column: string | null;
  /** null: over all cells. */
  group: string | null;
  n_cells: number;
  n_removed: number;
  fraction: number;
  whole: boolean;
  criterion: FilterCriterion | null;
  message: string;
  condition?: string;
  p?: number;
  max_diff_pp?: number;
}

/** Who the thresholds remove, per label group x condition (backend qc_filter.audit). */
export interface RemovalAudit {
  label: { column: string | null; reason: string | null; n_groups: number };
  condition: { column: string; reason: string; names: string[] } | null;
  by_type: RemovalRow[];
  overall: RemovalRow;
  n_unlabelled: number;
  note: string | null;
  rule: string;
  flags: RemovalFlag[];
}

/** The same for what the last filtering step removed (backend qc_filter.recorded_audit). */
export interface RecordedRemovalAudit extends Partial<RemovalAudit> {
  timestamp: string;
  mode: string;
  n_before: number;
  n_removed: number;
  truncated: boolean;
  has_mt: boolean;
  /** Cells were added or removed since: only `note` is meaningful. */
  stale: boolean;
  /** The recorded columns still in the dataset (the selectors' choices). */
  columns: string[];
  note?: string | null;
}

/** Which columns the removal audit splits by ("" or absent: detected). */
export interface RemovalAuditColumns {
  label_column?: string;
  condition_column?: string;
}

/** What the thresholds would remove (backend core/qc_filter.py). */
export interface FilterPreview {
  n_cells: number;
  n_removed: number;
  n_kept: number;
  fraction_removed: number;
  thresholds: Omit<
    FilterThresholds,
    "max_pct_mt" | "qc_mad_k" | "qc_mad_k_mt" | "qc_mad_upper" | "qc_threshold_mode"
  > & {
    max_pct_mt: number | null;
    mode?: QcThresholdMode;
    mad_k?: number;
    mad_k_mt?: number;
    mad_upper?: boolean;
    mito_mode?: MitoMode;
    mito_type_column?: string;
    mito_type_floor?: boolean;
  };
  has_mt: boolean;
  /** The dataset has doublet calls, so "drop predicted doublets" has something to drop. */
  doublet_calls: boolean;
  by_criterion: Partial<Record<FilterCriterion, { label: string; n_removed: number }>>;
  /** MAD mode only (older backends omit it). */
  mad?: MadThresholds | null;
  samples: FilterGrouping | null;
  clusters: FilterGrouping | null;
  warnings: {
    kind: "sample" | "cluster";
    column: string;
    group: string;
    n_cells: number;
    n_removed: number;
    fraction: number;
    whole: boolean;
    criterion: FilterCriterion | null;
    message: string;
  }[];
  /** How scView last filtered this dataset, if it did. */
  previous?: { mode: QcThresholdMode; timestamp: string | null } | null;
  /** Who the thresholds remove per label group x condition (older backends omit it). */
  audit?: RemovalAudit;
  /** What the last filtering step removed; null for a dataset without the record. */
  recorded?: RecordedRemovalAudit | null;
  /** Per-type % mito bounds (per-type mode only): each type's rule and loss. */
  mito_types?: MitoTypesTable | null;
  /** Whose doublets, and hashing negatives (null without doublet calls; older backends omit it). */
  doublets?: DoubletChoices | null;
}

export async function getFilterPreview(
  datasetId: string,
  t: FilterThresholds,
  columns: RemovalAuditColumns = {},
): Promise<FilterPreview> {
  const q = new URLSearchParams({
    min_genes: String(t.min_genes),
    max_genes: String(t.max_genes),
    min_counts: String(t.min_counts),
    max_pct_mt: String(t.max_pct_mt),
    drop_doublets: String(t.drop_doublets),
    mode: thresholdMode(t),
  });
  if (t.qc_mad_k !== undefined) q.set("mad_k", String(t.qc_mad_k));
  if (t.qc_mad_k_mt !== undefined) q.set("mad_k_mt", String(t.qc_mad_k_mt));
  if (t.qc_mad_upper !== undefined) q.set("mad_upper", String(t.qc_mad_upper));
  if (t.qc_sample_column) q.set("sample_column", t.qc_sample_column);
  if (t.qc_mito_mode === "per_type") {
    q.set("mito_mode", "per_type");
    if (t.qc_mito_type_column) q.set("mito_type_column", t.qc_mito_type_column);
    if (t.qc_mito_type_floor) q.set("mito_type_floor", "true");
  }
  if (t.qc_doublet_source && t.qc_doublet_source !== "auto") q.set("doublet_source", t.qc_doublet_source);
  if (t.drop_hash_negatives) q.set("drop_hash_negatives", "true");
  if (columns.label_column) q.set("label_column", columns.label_column);
  if (columns.condition_column) q.set("condition_column", columns.condition_column);
  try {
    return await apiFetch<FilterPreview>(`/datasets/${datasetId}/qc/filter-preview?${q.toString()}`);
  } catch (e) {
    throw withDetail(e);
  }
}

/** The "Explain this data" QC report, written from the deterministic facts only. */
export interface QcReport {
  report: string; // Markdown in four sections; "" = not generated yet
  facts: ChatSource[];
  facts_hash: string;
  grounded: boolean; // an LLM wrote it (false = the deterministic facts summary)
  cached: boolean; // replayed from the per-dataset cache, no new LLM call
  model?: string | null;
  generated_at?: string | null;
  llm_available: boolean;
}

/** The cached report for the current facts, if any. Never calls the LLM. */
export async function getQcReport(datasetId: string): Promise<QcReport> {
  return apiFetch<QcReport>(`/datasets/${datasetId}/qc-report`);
}

export async function postQcReport(datasetId: string, regenerate = false): Promise<QcReport> {
  return apiFetch<QcReport>(`/datasets/${datasetId}/qc-report`, {
    method: "POST",
    body: JSON.stringify({ regenerate }),
  });
}

export interface QcReportHandlers {
  onFacts?: (ev: { facts: ChatSource[]; facts_hash: string; llm_available: boolean }) => void;
  onDelta?: (text: string) => void;
  onDone?: (ev: { grounded: boolean; cached: boolean; model?: string | null; generated_at?: string | null }) => void;
  onError?: (msg: string) => void;
}

/** Stream the report via SSE (same event shape as the co-pilot chat). Throws on
 *  a transport/HTTP error so the caller can fall back to ``postQcReport``. */
export async function streamQcReport(
  datasetId: string,
  regenerate: boolean,
  handlers: QcReportHandlers,
): Promise<void> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/qc-report-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ regenerate }),
  });
  if (!res.ok || !res.body) throw new Error(`API error ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 2);
      const line = raw.startsWith("data:") ? raw.slice(5).trim() : raw;
      if (!line) continue;
      let ev: { type: string; [k: string]: unknown };
      try {
        ev = JSON.parse(line);
      } catch {
        continue;
      }
      if (ev.type === "facts") handlers.onFacts?.(ev as never);
      else if (ev.type === "delta") handlers.onDelta?.(String(ev.text ?? ""));
      else if (ev.type === "done") handlers.onDone?.(ev as never);
      else if (ev.type === "error") handlers.onError?.(String(ev.error ?? "stream error"));
    }
  }
}
