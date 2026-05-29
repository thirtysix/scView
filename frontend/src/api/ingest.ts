import { API_BASE } from "@/lib/constants";
import { apiFetch } from "./client";

// --- Types (mirror backend scview.core.ingestion models) ------------------

export type FileRole = "matrix" | "barcodes" | "features" | "data" | "unknown";
export type UnitFormat =
  | "tenx_mex"
  | "tenx_h5"
  | "anndata"
  | "loom"
  | "zarr"
  | "dense_table"
  | "seurat"
  | "unknown";

export interface BundleFile {
  path: string;
  name: string;
  kind: string;
  role: FileRole;
  confidence: number;
}

export interface IngestUnit {
  label: string;
  format: UnitFormat;
  files: BundleFile[];
  complete: boolean;
  missing_roles: FileRole[];
  issues: string[];
}

export interface Bundle {
  units: IngestUnit[];
  is_merge: boolean;
  complete: boolean;
  issues: string[];
}

export type IssueSeverity = "info" | "warn" | "error";

export interface IngestIssue {
  severity: IssueSeverity;
  code: string;
  message: string;
  suggestion: string;
  unit_label: string;
}

export interface ValidationReport {
  issues: IngestIssue[];
  ok: boolean;
}

export type MergeJoin = "inner" | "outer";

export interface IngestOptions {
  name: string | null;
  join: MergeJoin;
  sample_label: string;
  apply_reconciliation: boolean;
  genes_in_rows: boolean;
}

export interface IngestState {
  session_id: string;
  bundle: Bundle;
  validation: ValidationReport;
  options: IngestOptions;
}

export interface VarReset {
  sample: string;
  via_column: string;
  from_basis: string;
}

export interface Reconciliation {
  needed: boolean;
  feasible: boolean;
  target_basis: string;
  resets: VarReset[];
  overlap_before: number;
  overlap_after: number;
  message: string;
}

export interface MergePlan {
  is_merge?: boolean; // only present (false) when there's nothing to merge
  samples: string[];
  per_sample_genes: Record<string, number>;
  bases: Record<string, string>;
  intersection: number;
  union: number;
  suspicious_low_overlap: boolean;
  reconciliation: Reconciliation | null;
  recommended_join: MergeJoin;
  est_cells: number;
  est_genes: number;
  warnings: string[];
}

// --- API calls -------------------------------------------------------------

export async function createIngestSession(): Promise<{ session_id: string }> {
  return apiFetch<{ session_id: string }>("/ingest/session", { method: "POST" });
}

export async function uploadIngestFiles(sid: string, files: File[]): Promise<IngestState> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const res = await fetch(`${API_BASE}/ingest/session/${sid}/files`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload error ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function getIngestState(sid: string): Promise<IngestState> {
  return apiFetch<IngestState>(`/ingest/session/${sid}`);
}

export async function getMergePlan(sid: string): Promise<MergePlan> {
  return apiFetch<MergePlan>(`/ingest/session/${sid}/merge-plan`);
}

export async function setIngestOptions(
  sid: string,
  options: IngestOptions
): Promise<IngestState> {
  return apiFetch<IngestState>(`/ingest/session/${sid}/options`, {
    method: "POST",
    body: JSON.stringify(options),
  });
}

export async function commitIngestSession(sid: string): Promise<{ dataset_id: string }> {
  return apiFetch<{ dataset_id: string }>(`/ingest/session/${sid}/commit`, {
    method: "POST",
  });
}

export async function discardIngestSession(sid: string): Promise<void> {
  await apiFetch(`/ingest/session/${sid}`, { method: "DELETE" });
}
