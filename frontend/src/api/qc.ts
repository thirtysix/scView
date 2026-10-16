import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";
import type { ChatSource } from "./assistant";

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
  flags: ClusterFlag[];
  reasons: string[]; // one plain-language sentence per flag, with the numbers
}

export type ClusterFlag =
  | "low_quality"
  | "high_mito"
  | "low_genes"
  | "doublet_rich"
  | "stress_driven"
  | "small";

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

/** QC by group; with no column the backend picks the donor, else batch, else clustering
 *  column. A refused column (too many groups) throws with the server's reason. */
export async function getQcByGroup(datasetId: string, column?: string): Promise<QcByGroup> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  try {
    return await apiFetch<QcByGroup>(`/datasets/${datasetId}/qc/by-group${q}`);
  } catch (e) {
    const body = e instanceof Error ? e.message.replace(/^API error \d+: /, "") : "";
    let detail = "";
    try {
      detail = String(JSON.parse(body).detail ?? "");
    } catch {
      /* not JSON: keep the original error */
    }
    throw detail ? new Error(detail) : e;
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
