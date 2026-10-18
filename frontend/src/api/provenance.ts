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

/** How the replay script reproduces a step (backend core/replay_script.py). */
export type ReplayStatus = "exact" | "approximate" | "recorded" | "not_replayable" | "unknown";

export interface RecipeStep {
  commit_id?: string | null;
  step: string;
  params: Record<string, unknown>;
  tool?: string | null;
  timestamp?: string | null;
  scview_version?: string | null;
  effect?: { n_cells: number; n_genes: number } | null;
  replay?: { status: ReplayStatus; note: string };
}

export interface Recipe {
  recipe: RecipeStep[];
  source: ProvSource;
}

export async function getRecipe(id: string): Promise<Recipe> {
  return apiFetch<Recipe>(`/datasets/${id}/recipe`);
}
