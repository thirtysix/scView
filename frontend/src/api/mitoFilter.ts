import { apiFetch, readError } from "./client";
import { API_BASE } from "@/lib/constants";

/** How the filtering step judges % mito (backend PipelineParams.qc_mito_mode). */
export type MitoMode = "global" | "per_type";

/** One cell type's % mito rule in the filtering step (backend qc_filter.type_bounds). */
export interface MitoTypeRule {
  name: string;
  n_cells: number;
  median: number | null;
  mad: number | null;
  /** Its own bound, median + k MADs (capped at `cap`); null: the global rule (`why`). */
  bound: number | null;
  rule: "own" | "global";
  why: "too_few" | "no_spread" | "damaged" | "no_values" | null;
  capped: boolean;
  note: string;
  warning: string | null;
  /** Cells the global rule's % mito threshold removes, and the rule applied. */
  removed_global?: number;
  removed?: number;
  /** Removed per type with the floor off and on (both, whichever is applied). */
  removed_type?: number;
  removed_floor?: number;
  /** Which bound the floor uses for the type's cells; null: no bound of its own (the global rule). */
  floor_uses?: "type" | "global" | "mixed" | null;
  /** Its cells whose global bound is above its own (the floor lifts them). */
  n_floored?: number;
  /** [lowest, highest] bound the global rule applies to its cells (MAD mode: per sample); null: none. */
  global_bound?: [number, number] | null;
}

/** The per-type table in the filter preview, the removal record and provenance. */
export interface MitoTypesTable {
  column: string;
  reason: string | null;
  k: number;
  cap: number;
  min_cells: number;
  damaged_median: number;
  ceiling_pct_mt: number | null;
  types: MitoTypeRule[];
  n_own: number;
  n_global: number;
  n_unlabelled: number;
  reused_from: string | null;
  warnings: string[];
  applied?: MitoMode;
  /** The floor was applied: never stricter than the global rule. */
  floor?: boolean;
}

/** One type in the re-filter preview: its rule plus what changes. */
export interface RefilterType extends Partial<MitoTypeRule> {
  name: string;
  /** Cells of the type in the original upload. */
  n_all: number;
  /** Removed by the dataset's last filtering, and by the re-filter. */
  removed_before: number;
  removed_after: number;
  /** Removed before, kept now; kept before, removed now. */
  brought_back: number;
  newly_removed: number;
}

/** What re-filtering from the original upload would do (backend core/mito_refilter.preview). */
export interface RefilterPreview {
  column: string;
  /** The column the filter judges by: the label column itself (author types) or scview_mito_type. */
  type_column: string;
  mito_mode: MitoMode;
  /** Per type never stricter than the global rule. */
  mito_type_floor?: boolean;
  source_file: string;
  reset_to_counts: boolean;
  n_original: number;
  n_current: number;
  n_after: number;
  removed_before: number;
  removed_after: number;
  brought_back: number;
  newly_removed: number;
  removed_other: number;
  /** How each cell got its type: kept (by barcode), the original's column, by expression, none. */
  typed: { kept: number; column: number; expression: number; none: number };
  assignment: {
    method: string;
    metric: string;
    genes: number;
    n?: number;
    n_assigned?: number;
    median_r?: number | null;
    n_low?: number;
    note?: string;
  } | null;
  types: RefilterType[];
  rule: MitoTypesTable | null;
  steps: string[];
  skipped_steps: string[];
  /** Plain sentences: which cells it brings back, which it cannot. */
  can: string;
  cannot: string;
  notes: string[];
  warnings: string[];
}

function detailOf(e: unknown): Error {
  const msg = e instanceof Error ? e.message : String(e);
  const body = msg.replace(/^API error \d+: /, "");
  try {
    const detail = JSON.parse(body).detail;
    if (detail) return new Error(String(detail));
  } catch {
    /* not JSON */
  }
  return e instanceof Error ? e : new Error(msg);
}

export async function getMitoRefilter(datasetId: string, column?: string, floor = false): Promise<RefilterPreview> {
  const params = new URLSearchParams();
  if (column) params.set("column", column);
  if (floor) params.set("floor", "true");
  const q = params.toString() ? `?${params.toString()}` : "";
  try {
    return await apiFetch<RefilterPreview>(`/datasets/${datasetId}/qc/mito-refilter${q}`);
  } catch (e) {
    throw detailOf(e);
  }
}

export interface RefilterEvent {
  event: string;
  data: Record<string, unknown>;
}

/**
 * Re-filter from the original upload with per-type bounds (`floor`: never stricter
 * than the global rule), calling `onEvent` for
 * each SSE event (`refilter` stages, then the pipeline's step_start / step_done /
 * substep / step_error, and `complete`). Rejects on an `error` event, a refused
 * request, or a stream that ends without `complete`.
 */
export async function runMitoRefilter(
  datasetId: string,
  column: string | undefined,
  onEvent: (e: RefilterEvent) => void,
  floor = false,
): Promise<Record<string, unknown>> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/qc/mito-refilter-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ column: column ?? "", floor }),
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
  let result: Record<string, unknown> | null = null;
  for (;;) {
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
        if (event === "complete") result = data;
        onEvent({ event, data });
        event = "";
      }
    }
  }
  if (!result) throw new Error("The re-filter stopped before it finished; nothing was saved.");
  return result;
}
