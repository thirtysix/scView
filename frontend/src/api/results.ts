import { API_BASE } from "@/lib/constants";
import { apiFetch } from "./client";

/** A result kept in the per-dataset results store (backend core/results_store.py). */
export interface StoredResult {
  id: string;
  /** de | markers | enrichment | ccc | ccc_compare | composition | milo | cytokine | ... */
  kind: string;
  /** What it is about (a column, a column|group, a comparison label). */
  key: string;
  label: string;
  /** primary (the newest of its kind and key) | superseded | pinned */
  status: "primary" | "superseded" | "pinned";
  version: string;
  bytes: number;
  /** Computed on an earlier version of the data; null when the dataset is not open. */
  stale: boolean | null;
  /** ISO 8601, UTC */
  created_at: string;
}

export interface StoredResultsList {
  results: StoredResult[];
  keep_superseded: number;
  max_bytes: number;
  total_bytes: number;
}

/** Kinds whose earlier result can be made current again (they live in the h5ad). */
export const RESTORABLE_KINDS = new Set(["markers", "enrichment"]);

export function listResults(datasetId: string): Promise<StoredResultsList> {
  return apiFetch<StoredResultsList>(`/datasets/${datasetId}/results`);
}

export function deleteResult(datasetId: string, id: string): Promise<{ deleted: string }> {
  return apiFetch(`/datasets/${datasetId}/results/${id}`, { method: "DELETE" });
}

export function pinResult(datasetId: string, id: string, pinned: boolean): Promise<{ pinned: boolean }> {
  return apiFetch(`/datasets/${datasetId}/results/${id}/pin`, {
    method: "POST",
    body: JSON.stringify({ pinned }),
  });
}

export function restoreResult(datasetId: string, id: string): Promise<{ restored: string; kind: string }> {
  return apiFetch(`/datasets/${datasetId}/results/${id}/restore`, { method: "POST" });
}

export interface MarkersComparison {
  kind: "markers";
  n: number;
  groups: {
    group: string;
    /** Top-n genes in both; null when the group is not in the current markers. */
    shared: number | null;
    jaccard?: number;
    only_earlier: string[];
    only_current: string[];
  }[];
  earlier: Record<string, unknown>;
  current: Record<string, unknown>;
}

export interface EnrichmentComparison {
  kind: "enrichment";
  n: number;
  fdr: number;
  n_sig_earlier: number;
  n_sig_current: number;
  shared: number;
  jaccard: number | null;
  top_earlier: string[];
  top_current: string[];
  only_earlier: string[];
  only_current: string[];
}

export type ResultComparison = (MarkersComparison | EnrichmentComparison) & { id: string; key: string };

export function compareWithCurrent(datasetId: string, id: string, n = 50): Promise<ResultComparison> {
  return apiFetch(`/datasets/${datasetId}/results/${id}/compare?n=${n}`);
}

/** Plain links: the browser saves them (the zip can hold the whole h5ad). */
export function resultTableUrl(datasetId: string, id: string): string {
  return `${API_BASE}/datasets/${datasetId}/results/${id}/table.tsv.gz`;
}

export function resultsZipUrl(datasetId: string): string {
  return `${API_BASE}/datasets/${datasetId}/results.zip`;
}

export function h5adWithResultsUrl(datasetId: string): string {
  return `${API_BASE}/datasets/${datasetId}/export/h5ad?with_results=true`;
}

const KIND_LABELS: Record<string, string> = {
  de: "DE",
  markers: "Markers",
  enrichment: "Enrichment",
  ccc: "Ligand-receptor",
  ccc_compare: "L-R change",
  composition: "Composition",
  milo: "Milo",
  milode: "Neighbourhood DE",
  merge_check: "Merge check",
  count_split: "Count split",
  cytokine: "Cytokines",
  offtarget: "Off-target",
  ppi: "Protein interactions",
  planner: "Study planner",
};

export function kindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind.replace(/_/g, " ");
}
