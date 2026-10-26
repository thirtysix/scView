import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** A user-uploaded gene-set collection (backend core/custom_genesets.py). */
export interface CustomCollection {
  id: string;
  name: string;
  created_at: string;
  source_filename: string;
  format: string;
  n_sets: number;
  /** Distinct genes in the collection. */
  n_genes: number;
  category: "Custom";
  available: boolean;
  /** Present only when listed for a dataset: genes it measures, ignoring case. */
  n_genes_present?: number;
  n_sets_with_genes?: number;
}

export interface CustomUploadResult extends CustomCollection {
  warnings: string[];
}

/** A rejected upload: the reason, plus each problem the backend found in the file. */
export class GeneSetUploadError extends Error {
  problems: string[];
  constructor(message: string, problems: string[] = []) {
    super(message);
    this.name = "GeneSetUploadError";
    this.problems = problems;
  }
}

export const CUSTOM_COLLECTION_PREFIX = "custom-";

export function isCustomCollectionId(id: string): boolean {
  return id.startsWith(CUSTOM_COLLECTION_PREFIX);
}

/** How a result's collection is shown: a custom collection by name, MSigDB by id. */
export function collectionLabel(r: { collection?: string; collection_name?: string }): string {
  return r.collection_name || r.collection || "";
}

/** "3 sets · 41/52 genes in this dataset" (the gene part only when listed for a dataset). */
export function describeCoverage(c: CustomCollection): string {
  const sets = `${c.n_sets} set${c.n_sets === 1 ? "" : "s"}`;
  if (c.n_genes_present == null) return sets;
  return `${sets} · ${c.n_genes_present}/${c.n_genes} genes in this dataset`;
}

/** A collection as the gene-set browsers' dropdowns list it (MSigDB or custom). */
export interface CollectionChoice {
  id: string;
  name: string;
  category?: string;
  n_sets?: number;
  n_genes?: number;
  n_genes_present?: number;
  /** An MSigDB collection's species (whose gene symbols it lists). */
  species?: "human" | "mouse";
}

export function collectionOptionLabel(c: CollectionChoice): string {
  const sets = c.n_sets ? `${c.n_sets} sets` : "";
  const genes =
    c.category === "Custom" && c.n_genes_present != null && c.n_genes
      ? `${c.n_genes_present}/${c.n_genes} genes in data`
      : "";
  const extra = [sets, genes].filter(Boolean).join(", ");
  return extra ? `${c.name} (${extra})` : c.name;
}

function parseError(status: number, body: string): GeneSetUploadError {
  try {
    const detail = JSON.parse(body).detail;
    if (typeof detail === "string") return new GeneSetUploadError(detail);
    if (detail && typeof detail.message === "string") {
      return new GeneSetUploadError(
        detail.message,
        Array.isArray(detail.problems) ? detail.problems.map(String) : [],
      );
    }
  } catch {
    /* not JSON: fall through */
  }
  return new GeneSetUploadError(body ? `Error ${status}: ${body}` : `Error ${status}`);
}

async function send<T>(path: string, init: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw parseError(res.status, await readError(res));
  if (res.status === 204) return undefined as T;
  return res.json();
}

/** Every custom collection; with a dataset, each also says how much of it the dataset measures. */
export async function listCustomCollections(datasetId?: string | null): Promise<CustomCollection[]> {
  const path = datasetId ? `/datasets/${datasetId}/genesets/custom` : "/genesets/custom";
  const data = await apiFetch<{ collections: CustomCollection[] }>(path);
  return data.collections ?? [];
}

/** Upload a .gmt / .csv / .tsv as a named collection. A rejected file throws GeneSetUploadError. */
export async function uploadCustomCollection(file: File, name: string): Promise<CustomUploadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", name);
  return send<CustomUploadResult>("/genesets/custom/upload", { method: "POST", body: form });
}

/** Save named gene lists as one collection (one set per list). Rejected lists throw GeneSetUploadError. */
export async function customCollectionFromLists(
  name: string,
  sets: Record<string, string[]>,
): Promise<CustomUploadResult> {
  return send<CustomUploadResult>("/genesets/custom/from-lists", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, sets }),
  });
}

export async function renameCustomCollection(id: string, name: string): Promise<CustomCollection> {
  return send<CustomCollection>(`/genesets/custom/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function deleteCustomCollection(id: string): Promise<void> {
  await send<void>(`/genesets/custom/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/** The selection without custom ids that no longer exist (MSigDB ids are kept). */
export function pruneDeletedCustom(selected: Set<string>, existing: Iterable<string>): Set<string> {
  const keep = new Set(existing);
  const next = new Set<string>();
  for (const id of selected) {
    if (!isCustomCollectionId(id) || keep.has(id)) next.add(id);
  }
  return next;
}
