import { apiFetch } from "./client";
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
  ok: boolean;
}

export type IntegrationMethod = "harmony" | "none";

export interface IntegrateRequest {
  dataset_ids: string[];
  labels?: string[];
  name?: string;
  method?: IntegrationMethod;
  params?: Record<string, number>;
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
    let detail = await res.text();
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
