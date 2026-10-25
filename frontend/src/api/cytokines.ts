import { apiFetch } from "./client";

/** Cytokine response (CytoSig): backend core/cytokine_activity.py. */
export interface CytokineRequest {
  mode: "groups" | "conditions";
  /** groups: required; conditions: optional ("" = all cells as one group). */
  groupby?: string;
  reference?: "rest" | "all";
  condition?: string;
  a?: string;
  b?: string;
  n_permutations?: number;
}

export interface CytokineTop {
  group: string;
  /** The 5 highest z, strongest first. */
  cytokines: { cytokine: string; z: number; p: number }[];
}

export interface CytokineResult {
  mode: "groups" | "conditions";
  groupby: string | null;
  reference: "rest" | "all";
  condition: string | null;
  a: string | null;
  b: string | null;
  cytokines: string[];
  groups: string[];
  /** Cells per group: [n] (groups) or [n_a, n_b] (conditions). */
  group_cells: number[][];
  skipped: { group: string; reason: string }[];
  n_shared_genes: number;
  n_signature_genes: number;
  /** groups x cytokines */
  z: number[][];
  p: number[][];
  coef: number[][];
  top: CytokineTop[];
  lambda: number;
  n_permutations: number;
  species: string;
  mapped_by: string;
  matrix_source: string;
  citation: string;
  source_url: string;
  licence: string;
  elapsed_seconds: number;
  reused: boolean;
}

export async function getCytokineActivity(datasetId: string, body: CytokineRequest): Promise<CytokineResult> {
  return apiFetch<CytokineResult>(`/datasets/${datasetId}/cytokines/activity`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
