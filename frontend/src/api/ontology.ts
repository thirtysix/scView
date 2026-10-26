import { API_BASE } from "@/lib/constants";
import { readError } from "./client";

/** Cell Ontology terms for labels (backend api/v1/ontology.py, core/cell_ontology.py). */

/** The term value for cells of no known type (CELLxGENE's "unknown"). */
export const UNKNOWN_TERM = "unknown";

export interface CellTypeTerm {
  id: string;
  name: string;
  /** The synonym the search matched, when not the name */
  matched: string | null;
  /** How it matched: id | name | synonym | words | prefix | tokens */
  match: string | null;
  definition: string | null;
  parents: { id: string; name: string }[];
  /** The term's page on the EBI Ontology Lookup Service */
  url: string;
}

export interface TermSuggestion extends CellTypeTerm {
  how: string;
  how_text: string;
}

export interface LabelTermRow {
  label: string;
  n_cells: number;
  /** A CL id, "unknown", or null (no term) */
  id: string | null;
  name: string | null;
  url: string | null;
}

export interface LabelTerms {
  column: string;
  labels: LabelTermRow[];
  n_labels: number;
  n_mapped: number;
  missing: string[];
  n_cells_unlabelled: number;
  /** Every label (and every cell) has a term */
  complete: boolean;
  version: string | null;
  /** Columns of the dataset with a mapping */
  columns: string[];
  /** After a change: each changed label's previous term */
  previous?: Record<string, string | null>;
}

export interface LabelSuggestions {
  column: string;
  labels: { label: string; n_cells: number; current: string | null; suggestions: TermSuggestion[] }[];
  n_labels: number;
  n_suggested: number;
  version: string | null;
}

/** The term's page on the EBI Ontology Lookup Service. */
export function olsUrl(id: string): string {
  return `https://www.ebi.ac.uk/ols4/ontologies/cl/classes?obo_id=${encodeURIComponent(id)}`;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await readError(res);
    let detail = text;
    try {
      detail = String(JSON.parse(text).detail ?? text);
    } catch {
      /* plain text */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

/** Cell Ontology terms for `q` (a name, synonym, part of one, or a CL id), best first. */
export function searchCellTypes(
  q: string,
  limit = 12,
  signal?: AbortSignal,
): Promise<{ query: string; terms: CellTypeTerm[]; version: string | null; n_terms: number }> {
  const params = new URLSearchParams({ q, limit: String(limit) });
  return call(`/ontology/cell-types?${params.toString()}`, { signal });
}

/** Each label of `column` with the term the user gave it. */
export function getLabelTerms(datasetId: string, column: string): Promise<LabelTerms> {
  return call(`/datasets/${datasetId}/ontology/terms?${new URLSearchParams({ column }).toString()}`);
}

/** Set (a CL id or "unknown") or clear (null) the terms of some labels; saved to the dataset. */
export function setLabelTerms(
  datasetId: string,
  column: string,
  terms: Record<string, string | null>,
): Promise<LabelTerms> {
  return call(`/datasets/${datasetId}/ontology/terms`, {
    method: "POST",
    body: JSON.stringify({ column, terms }),
  });
}

/** Clear every term of a column. */
export function clearLabelTerms(datasetId: string, column: string): Promise<LabelTerms> {
  return call(`/datasets/${datasetId}/ontology/terms`, {
    method: "POST",
    body: JSON.stringify({ column, clear: true }),
  });
}

/** Up to three terms per label from its name; nothing is saved. */
export function suggestLabelTerms(datasetId: string, column: string): Promise<LabelSuggestions> {
  return call(`/datasets/${datasetId}/ontology/terms/suggest`, {
    method: "POST",
    body: JSON.stringify({ column }),
  });
}
