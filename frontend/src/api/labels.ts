import { apiFetch } from "./client";

/** Cells that had `value` before a labelling (null: no value). */
export interface PreviousValues {
  value: string | null;
  indices: number[];
}

export interface LabelResult {
  column: string;
  label: string;
  n_cells: number;
  /** The labelling created the column */
  created: boolean;
  /** What an undo puts back */
  previous: PreviousValues[];
  /** The label's Cell Ontology term before this labelling, when it changed (label -> id or null) */
  previous_terms?: Record<string, string | null>;
}

export interface UndoLabelResult {
  column: string;
  n_cells: number;
  /** The column was removed: it held no labels any more */
  removed: boolean;
}

/** A failed request, with the server's reason as the message. */
export class ApiDetailError extends Error {
  constructor(message: string, readonly status: number | null) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  try {
    return await apiFetch<T>(path, init);
  } catch (e) {
    const raw = e instanceof Error ? e.message : String(e);
    const m = raw.match(/^API error (\d+): ([\s\S]*)$/);
    let detail = "";
    try {
      detail = String(JSON.parse(m?.[2] ?? "").detail ?? "");
    } catch {
      /* not JSON: keep the original message */
    }
    throw new ApiDetailError(detail || raw, m ? Number(m[1]) : null);
  }
}

/** The dataset's user annotation columns: where new labels can go. */
export function listLabelColumns(datasetId: string): Promise<{ columns: string[]; unassigned: string }> {
  return call(`/datasets/${datasetId}/labels/columns`);
}

/**
 * Give the cells at `indices` the label `label` in `column`: a new column when
 * `newColumn` (every other cell "Unassigned"), else an existing user annotation
 * column. Saved to the dataset before this resolves.
 */
export function labelCells(
  datasetId: string,
  body: { column: string; label: string; indices: number[]; newColumn: boolean; ontologyId?: string | null },
): Promise<LabelResult> {
  return call(`/datasets/${datasetId}/labels`, {
    method: "POST",
    body: JSON.stringify({
      column: body.column,
      label: body.label,
      indices: body.indices,
      new_column: body.newColumn,
      ontology_id: body.ontologyId ?? null,
    }),
  });
}

/** Undo one labelling: give its cells back their previous values. */
export function undoLabel(
  datasetId: string,
  body: {
    column: string;
    previous: PreviousValues[];
    created: boolean;
    /** The labelling's `previous_terms`, put back with the cells */
    previous_terms?: Record<string, string | null>;
  },
): Promise<UndoLabelResult> {
  return call(`/datasets/${datasetId}/labels/undo`, { method: "POST", body: JSON.stringify(body) });
}
