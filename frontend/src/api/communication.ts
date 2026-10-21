import { apiFetch } from "./client";
import { streamEvents } from "./clustering";
import { API_BASE } from "@/lib/constants";

/** LIANA's consensus ligand-receptor resource: backend core/lr_resource.py. */
export type LRResource = "consensus" | "mouseconsensus";

/** What the Communication tab can offer before a run (GET .../communication/options). */
export interface CommunicationOptions {
  /** Condition-like columns, to run on one condition's cells */
  conditions: { column: string; values: string[] }[];
  species: string;
  species_evidence: string;
  resource: LRResource;
  max_groups: number;
  max_perms: number;
  default_expr_prop: number;
  default_n_perms: number;
  min_cells: number;
  /** With a groupby: cells and groups the run would use, and its time estimate */
  n_cells?: number;
  n_groups?: number;
  estimate_seconds?: number;
}

export interface CommunicationRequest {
  groupby: string;
  subset_column?: string | null;
  subset_value?: string | null;
  resource?: "auto" | LRResource;
  expr_prop?: number;
  n_perms?: number;
}

export interface CommunicationStart {
  n_cells: number;
  n_groups: number;
  resource: LRResource;
  resource_label: string;
  species: string;
  species_evidence: string;
  n_perms: number;
  estimate_seconds: number;
}

/** Text columns of the result table (column-oriented, strongest magnitude first). */
export type RowText = "source" | "target" | "ligand_complex" | "receptor_complex" | "ligand" | "receptor";
/** Number columns; ranks and p-values: lower is stronger. null = not computed. */
export type RowNum =
  | "magnitude_rank"
  | "specificity_rank"
  | "lr_means"
  | "cellphone_pvals"
  | "expr_prod"
  | "scaled_weight"
  | "lr_logfc"
  | "spec_weight"
  | "lrscore"
  | "ligand_props"
  | "receptor_props";

export interface CommunicationResult {
  result_id: string;
  groupby: string;
  subset: { column: string; value: string } | null;
  resource: LRResource;
  resource_label: string;
  species: string;
  matrix_source: string;
  n_cells: number;
  groups: string[];
  group_sizes: number[];
  /** Groups left out for having fewer than the minimum cells: name -> cells */
  dropped_groups: Record<string, number>;
  expr_prop: number;
  n_perms: number;
  seed: number;
  n_resource: number;
  /** Source-target-interaction rows scored, and distinct ligand-receptor pairs among them */
  n_rows: number;
  n_interactions: number;
  /** Rows with specificity_rank <= specific_threshold (what the pair counts count) */
  n_specific: number;
  specific_threshold: number;
  rows_shown: number;
  columns: Record<RowText, string[]> & Record<RowNum, (number | null)[]>;
  /** groups x groups (source rows, target columns): specific interactions and their summed LR mean */
  pair_counts: number[][];
  pair_strength: (number | null)[][];
  roles: {
    outgoing: number[];
    incoming: number[];
    outgoing_strength: (number | null)[];
    incoming_strength: (number | null)[];
  };
  elapsed_seconds: number;
  citation: string;
}

export function getCommunicationOptions(
  datasetId: string,
  query: { groupby?: string; subset_column?: string; subset_value?: string; n_perms?: number } = {},
): Promise<CommunicationOptions> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) if (v !== undefined && v !== "") q.set(k, String(v));
  const s = q.toString();
  return apiFetch(`/datasets/${datasetId}/communication/options${s ? `?${s}` : ""}`);
}

/**
 * Score every sender x receiver x ligand-receptor pair, streamed: `onStart` once,
 * `onProgress` per batch of permutations, then the result. Rejects with the
 * server's reason; aborting `signal` stops reading and, on the server, the run.
 */
export async function runCommunication(
  datasetId: string,
  body: CommunicationRequest,
  handlers: {
    onStart?: (s: CommunicationStart) => void;
    onProgress?: (done: number, total: number) => void;
    signal?: AbortSignal;
  } = {},
): Promise<CommunicationResult> {
  let result: CommunicationResult | null = null;
  await streamEvents(
    `/datasets/${datasetId}/communication/run`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as CommunicationStart);
      else if (event === "progress") {
        const p = data as { done: number; total: number };
        handlers.onProgress?.(p.done, p.total);
      } else if (event === "result") result = data as CommunicationResult;
    },
    "The communication run",
    handlers.signal,
  );
  if (!result) throw new Error("The communication run stopped before it sent a result.");
  return result;
}

/** The full table of a result the server still keeps, as CSV. */
export function communicationCsvUrl(datasetId: string, resultId: string): string {
  return `${API_BASE}/datasets/${datasetId}/communication/results/${resultId}.csv`;
}
