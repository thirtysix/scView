import { apiFetch, readError } from "./client";
import { API_BASE } from "@/lib/constants";

export interface IntegrationDatasetCheck {
  id: string;
  name: string;
  label: string;
  n_cells: number;
  n_genes: number;
  counts_source: string | null;
  columns: string[];
  error: string | null;
}

export interface IntegrationMergePlan {
  intersection: number;
  union: number;
  suspicious_low_overlap: boolean;
  reconciliation: { needed: boolean; feasible: boolean; message: string; overlap_after: number } | null;
  est_genes: number;
  warnings: string[];
}

export interface IntegrationPlan {
  datasets: IntegrationDatasetCheck[];
  merge: IntegrationMergePlan | null;
  shared_columns: string[];
  est_cells: number;
  blocking: string[];
  /** Design notes that don't block (e.g. each dataset holds one condition). */
  notes?: string[];
  ok: boolean;
}

export type IntegrationMethod = "harmony" | "scanorama" | "bbknn" | "none";

export interface IntegrateRequest {
  dataset_ids: string[];
  labels?: string[];
  name?: string;
  method?: IntegrationMethod;
  params?: Record<string, number>;
}

/** Batch mixing of one source dataset (values as in BatchMixing). */
export interface BatchMixingDataset {
  label: string;
  n_cells: number;
  ideal: number;
  uncorrected: number | null;
  corrected: number | null;
}

/**
 * Mean fraction of each cell's `k` nearest neighbours that come from another source
 * dataset, in the uncorrected and the corrected PCA; `ideal` is the value perfect
 * mixing gives for these dataset sizes.
 */
export interface BatchMixing {
  k: number;
  label_column: string;
  n_cells: number;
  n_scored: number;
  ideal: number;
  uncorrected: number | null;
  corrected: number | null;
  datasets: BatchMixingDataset[];
  /** Effective number of datasets among a cell's neighbours (1 = unmixed); older scores lack it. */
  ilisi?: { ideal: number; uncorrected: number | null; corrected: number | null };
  /** Bio-conservation: share of neighbours with the cell's own label, before and after. */
  conservation?: {
    label_column: string;
    kind: "shared" | "within_dataset";
    uncorrected: number | null;
    corrected: number | null;
    labels: { label: string; n_scored: number; uncorrected: number; corrected: number }[];
  } | null;
  /** Set when correction cost cell-type separation. */
  overcorrection?: string | null;
  /** What each space was scored on: an obsm key, or "graph" (a BBKNN neighbour graph). */
  spaces?: Record<string, string>;
  /** Neighbours per cell each space was scored with (a BBKNN graph can have fewer). */
  k_by_space?: Record<string, number>;
}

export interface JointSummary {
  dataset_id: string;
  joint: boolean;
  label_column?: string | null;
  cluster_column?: string | null;
  mixing?: BatchMixing | null;
}

/** A joint dataset's batch-mixing score; `joint` is false for any other dataset. */
export async function getJointSummary(datasetId: string): Promise<JointSummary> {
  return apiFetch<JointSummary>(`/datasets/${datasetId}/integration/summary`);
}

/** Read-only compatibility check (counts available, shared genes, annotations). */
export async function planIntegration(datasetIds: string[], labels?: string[]): Promise<IntegrationPlan> {
  return apiFetch<IntegrationPlan>("/integrate/plan", {
    method: "POST",
    body: JSON.stringify({ dataset_ids: datasetIds, labels }),
  });
}

export interface IntegrationEvent {
  event: string;
  data: Record<string, unknown>;
}

/**
 * Build + process a joint dataset, calling `onEvent` for each SSE event
 * (stage, created, step_start/step_done/step_error/substep, pipeline_complete,
 * complete). Resolves with the new dataset id; rejects on an `error` event.
 */
export async function runIntegration(
  body: IntegrateRequest,
  onEvent: (e: IntegrationEvent) => void,
): Promise<string> {
  const res = await fetch(`${API_BASE}/integrate/run-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = await readError(res);
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");

  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let datasetId: string | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ") && event) {
        const data = JSON.parse(line.slice(6)) as Record<string, unknown>;
        if (event === "error") throw new Error(String(data.error));
        if (event === "complete") datasetId = String(data.dataset_id);
        onEvent({ event, data });
        event = "";
      }
    }
  }
  if (!datasetId) throw new Error("Integration ended without completing.");
  return datasetId;
}
