import { apiFetch, readError } from "./client";
import type { CellFilter } from "@/lib/cellFilter";
import { errorMessage, errorStatus } from "@/lib/errorMessage";
import { API_BASE } from "@/lib/constants";
import type { LabelImportReport } from "@/lib/labelImport";

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
    throw new ApiDetailError(errorMessage(e), errorStatus(e));
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
  body: {
    column: string;
    label: string;
    indices: number[];
    newColumn: boolean;
    ontologyId?: string | null;
    /** The filter the cells came from: recorded for the replay script when it reads metadata only */
    filter?: CellFilter | null;
  },
): Promise<LabelResult> {
  return call(`/datasets/${datasetId}/labels`, {
    method: "POST",
    body: JSON.stringify({
      column: body.column,
      label: body.label,
      indices: body.indices,
      new_column: body.newColumn,
      ontology_id: body.ontologyId ?? null,
      ...(body.filter ? { filter: body.filter } : {}),
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

/**
 * A per-cell label table (CSV/TSV, gzipped too) into a new user annotation column
 * (``POST /labels/import-file``): matched by cell name in `barcodeColumn`, or by order
 * when it is null. `dryRun` only reports the match. A refusal (too few rows match,
 * counts differ by order) comes back as an ApiDetailError with the server's reason.
 */
export async function importLabelFile(
  datasetId: string,
  file: File,
  opts: {
    column: string;
    barcodeColumn: string | null;
    labelColumn: string;
    stripSuffix: boolean;
    minMatch: number;
    dryRun: boolean;
    /** False: into an existing user label column (cells the table leaves out keep their label). */
    newColumn?: boolean;
    /** By order: the table's sample column to check row by row (default: one the server finds). */
    sampleColumn?: string | null;
  },
): Promise<LabelImportReport> {
  const form = new FormData();
  form.append("file", file);
  form.append("column", opts.column);
  form.append("label_column", opts.labelColumn);
  if (opts.barcodeColumn) form.append("barcode_column", opts.barcodeColumn);
  form.append("match", opts.barcodeColumn ? "name" : "order");
  if (opts.newColumn === false) form.append("new_column", "false");
  if (!opts.barcodeColumn && opts.sampleColumn) form.append("sample_column", opts.sampleColumn);
  form.append("strip_suffix", String(opts.stripSuffix));
  form.append("min_match", String(opts.minMatch));
  form.append("dry_run", String(opts.dryRun));
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/datasets/${datasetId}/labels/import-file`, { method: "POST", body: form });
  } catch (e) {
    throw new ApiDetailError(errorMessage(e), null);
  }
  if (!res.ok) {
    const body = await readError(res);
    throw new ApiDetailError(errorMessage(new Error(`API error ${res.status}: ${body}`)), res.status);
  }
  return res.json() as Promise<LabelImportReport>;
}
