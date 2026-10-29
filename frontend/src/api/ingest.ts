import { API_BASE } from "@/lib/constants";
import type { UploadItem } from "@/lib/ingestUpload";
import type { ConversionProgress } from "./types";
import { apiFetch, readError } from "./client";

// --- Types (mirror backend scview.core.ingestion models) ------------------

export type FileRole = "matrix" | "barcodes" | "features" | "data" | "side" | "unknown";
/** What a side file is: Cell Ranger metrics, the raw matrix (sequencing QC), a TCR/BCR contig
 *  table, Cell Ranger's CRISPR guide calls, or a Space Ranger spatial file. */
export type SideKind = "metrics" | "raw" | "vdj" | "crispr" | "spatial";
/** Which Space Ranger spatial file (core/spatial.py PARTS). */
export type SpatialPart = "positions" | "scalefactors" | "hires" | "lowres" | "archive";
export type UnitFormat =
  | "tenx_mex"
  | "tenx_h5"
  | "anndata"
  | "loom"
  | "zarr"
  | "dense_table"
  | "seurat"
  | "h5seurat"
  | "unknown";

export interface BundleFile {
  path: string;
  name: string;
  kind: string;
  role: FileRole;
  confidence: number;
  /** Side files only (absent from older backends). */
  side_kind?: SideKind | null;
  /** Space Ranger spatial files only. */
  spatial_part?: SpatialPart | null;
  /** Contig tables: "T", "B" or "T+B", from their chains (else the file name). */
  receptor?: string | null;
  /** The path inside an uploaded folder; the name for a plain upload (absent from older backends). */
  rel_path?: string;
}

/** What to call a file in the wizard: its path inside the uploaded folder, so the raw and
 * filtered barcodes.tsv.gz are told apart; its name for a plain upload. */
export function filePath(f: Pick<BundleFile, "name" | "rel_path">): string {
  return f.rel_path || f.name;
}

export interface IngestUnit {
  label: string;
  format: UnitFormat;
  files: BundleFile[];
  complete: boolean;
  missing_roles: FileRole[];
  issues: string[];
  /** Never loaded as cells: Cell Ranger metrics / raw matrix (sequencing QC) and contig tables (absent from older backends). */
  side_files?: BundleFile[];
}

/** A unit's side files for sequencing QC (metrics, raw matrix), without its contig tables. */
export function seqQcSideFiles(unit: IngestUnit): BundleFile[] {
  return (unit.side_files ?? []).filter(
    (f) => f.side_kind !== "vdj" && f.kind !== "vdj_contigs" && f.side_kind !== "crispr",
  );
}

/** A unit's TCR/BCR contig tables. */
export function vdjSideFiles(unit: IngestUnit): BundleFile[] {
  return (unit.side_files ?? []).filter((f) => f.side_kind === "vdj" || f.kind === "vdj_contigs");
}

export interface Bundle {
  units: IngestUnit[];
  is_merge: boolean;
  complete: boolean;
  issues: string[];
  /** Files scView does not read, also summed up in one of the issues (absent from older backends). */
  skipped?: { path: string; reason: string; bytes?: number | null }[];
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
  /** The R object a commit is converting now (absent from older backends). */
  converting?: IngestConverting | null;
}

/** An .rds of the session being converted at commit: which file, which of how many. */
export interface IngestConverting {
  label: string;
  file: string;
  index: number;
  of: number;
  progress: ConversionProgress | null;
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

// The server's multipart parser takes at most 1000 files and 1000 fields per request.
export const UPLOAD_BATCH = 400;

/** Stage files, each with its path inside the uploaded folder (`paths`, one per file in
 * order): same-named files in different folders stay apart. A large folder goes in
 * batches of `UPLOAD_BATCH` files; the state after the last one is returned. */
export async function uploadIngestFiles(sid: string, items: UploadItem[]): Promise<IngestState> {
  let state: IngestState | null = null;
  for (let i = 0; i < items.length; i += UPLOAD_BATCH) {
    const form = new FormData();
    for (const { file, path } of items.slice(i, i + UPLOAD_BATCH)) {
      form.append("files", file);
      form.append("paths", path);
    }
    const res = await fetch(`${API_BASE}/ingest/session/${sid}/files`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(detailOf(await readError(res)) || `Upload error ${res.status}.`);
    state = (await res.json()) as IngestState;
  }
  return state ?? getIngestState(sid);
}

/** The server's sentence (FastAPI's `detail`) from an error body, else the body itself. */
function detailOf(text: string): string {
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON: the text as it is */
  }
  return text;
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
