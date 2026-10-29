import { apiFetch, readError } from "./client";
import { streamEvents } from "./clustering";
import { API_BASE } from "@/lib/constants";
import type { SessionSelection } from "@/lib/session";

/** Learned gates (backend core/learned_gate.py, api/v1/learned_gates.py). */

export interface GateCV {
  auroc: number;
  precision: number;
  recall: number;
  f1: number;
  n_pos: number;
  n_neg: number;
  folds: number;
  grouped: boolean;
  C: number;
  mid_share: number;
}

export interface GateGene {
  gene: string;
  /** The model's weight on the scaled gene (coefficient x spread). */
  weight: number;
  mean_in: number;
  mean_out: number;
  pct_in: number;
  pct_out: number;
}

export interface GateVerdict {
  level: "good" | "check" | "warn";
  text: string;
}

/** One library row. */
export interface LearnedGateSummary {
  id: string;
  name: string;
  created: string;
  species: string;
  gene_kind: string;
  n_genes: number;
  threshold: number;
  cv: GateCV;
  verdict: GateVerdict;
  top_for: string[];
  top_against: string[];
  source: {
    dataset_id: string;
    name: string;
    n_selected: number;
    sample_column: string | null;
    condition_column: string | null;
    negatives: string;
  };
  scview_version: string;
  /** An unreadable file in the library. */
  error?: string;
}

export interface LearnedGateFile {
  id: string;
  name: string;
  genes: string[];
  top_for: GateGene[];
  top_against: GateGene[];
  cv: GateCV;
  threshold: number;
  excluded_rules: string[];
  [key: string]: unknown;
}

export interface TrainReport {
  verdict: GateVerdict;
  n_candidates: number;
  n_genes: number;
  C: number;
  n_selected: number;
  n_pool: number;
  n_pos: number;
  n_neg: number;
  sample_column: string | null;
  condition_column: string | null;
  samples: string[];
  negatives: string;
  expression: string;
  warnings: string[];
  disagree: { n_missed: number; n_extra: number; n_training: number; cells: SessionSelection };
  seconds: number;
  score_note: string;
}

export interface ClusterOverlap {
  column: string;
  group: string;
  jaccard: number;
  is_cluster: boolean;
  note: string;
}

export interface TrainResult {
  gate: LearnedGateFile;
  summary: LearnedGateSummary;
  report: TrainReport;
  overlap: ClusterOverlap | null;
  saved: boolean;
}

export interface TrainStart {
  n_selected: number;
  n_pool: number;
  n_pos: number;
  n_neg: number;
  sample_column: string | null;
  condition_column: string | null;
  samples: string[];
  n_candidates: number;
  warnings: string[];
  n_c: number;
}

export interface TrainProgress {
  done: number;
  total: number;
  elapsed_s: number;
}

export interface TrainBody {
  name: string;
  cells: SessionSelection;
  /** null/undefined: picked by the server; "": none */
  sample_column?: string | null;
  exclude_cc_stress: boolean;
}

/** Train on the selection, streamed; resolves with the draft gate (not yet in the library). Aborting `signal` stops it. */
export async function trainLearnedGate(
  datasetId: string,
  body: TrainBody,
  handlers: {
    onStart?: (s: TrainStart) => void;
    onProgress?: (p: TrainProgress) => void;
    signal?: AbortSignal;
  } = {},
): Promise<TrainResult> {
  let result: TrainResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/learned-gates/train-stream`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as TrainStart);
      else if (event === "progress") handlers.onProgress?.(data as TrainProgress);
      else if (event === "result") result = data as TrainResult;
    },
    "Training the learned gate",
    handlers.signal,
  );
  if (!result) throw new Error("Training ended without a gate; nothing was kept.");
  return result;
}

export function saveLearnedGate(gateId: string): Promise<LearnedGateSummary> {
  return apiFetch(`/learned-gates/${gateId}/save`, { method: "POST" });
}

export function listLearnedGates(): Promise<{ gates: LearnedGateSummary[]; score_note: string; max_file_bytes: number }> {
  return apiFetch("/learned-gates");
}

/** The gate file's text (JSON), for "Export". */
export async function learnedGateFile(gateId: string): Promise<string> {
  const res = await fetch(`${API_BASE}/learned-gates/${gateId}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${await readError(res)}`);
  return res.text();
}

export function importLearnedGate(text: string): Promise<LearnedGateSummary> {
  return apiFetch("/learned-gates/import", { method: "POST", body: text });
}

export function deleteLearnedGate(gateId: string): Promise<{ deleted: string }> {
  return apiFetch(`/learned-gates/${gateId}`, { method: "DELETE" });
}

export interface GateCheck {
  ok: boolean;
  refusal: string | null;
  warnings: string[];
  species: string;
  gate_species: string;
  via: string | null;
  coverage: { n_genes: number; n_found: number; gene_share: number; weight_share: number; missing_top: string[] } | null;
  isg_weight_share?: number | null;
  columns?: { column: string; score_column: string };
  replaces?: boolean;
  sample_column?: string | null;
  /** A large dataset: applying reads its counts file once (backend lg.large_estimate) */
  large?: { seconds: number; peak_bytes: number; note: string } | null;
}

export function checkLearnedGate(datasetId: string, gateId: string): Promise<GateCheck> {
  return apiFetch(`/datasets/${datasetId}/learned-gates/${gateId}/check`, { method: "POST" });
}

export interface SampleCount {
  sample: string;
  n_cells: number;
  n_in: number;
  share: number;
}

export interface GateApplyResult {
  gate_id: string;
  name: string;
  column: string;
  score_column: string;
  replaced: boolean;
  n_cells: number;
  n_in: number;
  share: number;
  threshold: number;
  mid_share: number;
  training_mid_share: number;
  sample_column: string | null;
  per_sample: SampleCount[];
  coverage: NonNullable<GateCheck["coverage"]>;
  via: string;
  warnings: string[];
  verdict: GateVerdict;
  same_dataset: boolean;
  score_note: string;
}

export function applyLearnedGate(datasetId: string, gateId: string, sampleColumn?: string | null): Promise<GateApplyResult> {
  return apiFetch(`/datasets/${datasetId}/learned-gates/${gateId}/apply`, {
    method: "POST",
    body: JSON.stringify(sampleColumn ? { sample_column: sampleColumn } : {}),
  });
}
