import { apiFetch } from "./client";

export interface ProvSource {
  origin?: string;
  original_filename?: string;
  format?: string;
  ingested_at?: string;
  n_cells?: number;
  n_genes?: number;
  merged_from?: { sample: string; format?: string; n_files?: number }[];
  merge?: Record<string, unknown>;
}

export interface ProvStep {
  step: string;
  tool: string;
  params: Record<string, unknown>;
  timestamp: string;
  scview_version?: string;
  effect?: { n_cells: number; n_genes: number };
  note?: string;
}

export interface Provenance {
  recorded: {
    schema_version: number;
    source: ProvSource;
    history: ProvStep[];
    current: Record<string, unknown>;
  };
  has_history: boolean;
  reconcile_issues: string[];
}

export async function getProvenance(id: string): Promise<Provenance> {
  return apiFetch<Provenance>(`/datasets/${id}/provenance`);
}

export interface RerunPlan {
  edited_step: string;
  rerun_steps: string[];
  kept_steps: string[];
  requires_reprocess: boolean;
  message: string;
}

export async function getRerunPlan(id: string, step: string): Promise<RerunPlan> {
  return apiFetch<RerunPlan>(`/datasets/${id}/rerun-plan?step=${encodeURIComponent(step)}`);
}

export async function rerunStep(
  id: string,
  edited_step: string,
  params: Record<string, unknown>
): Promise<{ plan: RerunPlan; result: Record<string, unknown> }> {
  return apiFetch(`/datasets/${id}/rerun`, {
    method: "POST",
    body: JSON.stringify({ edited_step, params }),
  });
}
