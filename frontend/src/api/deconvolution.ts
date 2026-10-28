import { apiFetch, readError } from "./client";
import { streamEvents } from "./clustering";
import { API_BASE } from "@/lib/constants";

// --- Bulk deconvolution with the open dataset as the reference (backend core/deconvolution.py,
// api/v1/deconvolution.py) ---

export type DeconvMethod = "auto" | "weighted" | "nnls";

export interface DeconvOptions {
  celltype_column?: string | null;
  /** null: detect the donors; "": no donor column */
  donor_column?: string | null;
  method?: DeconvMethod;
  /** Only cells with cells_column = cells_value (one condition) form the reference */
  cells_column?: string | null;
  cells_value?: string | null;
}

export interface ReferenceCells {
  column: string;
  value: string;
  n_cells: number;
}

export interface ReferenceColumn {
  column: string;
  values: { value: string; n_cells: number }[];
}

/** A type estimated far above its share of the reference cells in one sample */
export interface DeconvFlag {
  sample: string;
  cell_type: string;
  fraction: number | null;
  reference_share: number | null;
}

export interface DeconvDefaults {
  celltype_column: string | null;
  donor_column: string | null;
  donor_reason: string | null;
  n_donors: number;
  min_donors_weighted: number;
  min_donors_check: number;
  /** Self-check time per held-out donor (2 cores) */
  check_seconds_per_donor: number;
  max_bulk_mb: number;
  max_samples: number;
  /** Condition-like columns to restrict the reference cells by */
  reference_columns: ReferenceColumn[];
}

export interface DeconvSample {
  sample: string;
  cell_fractions: Record<string, number | null>;
  rna_fractions: Record<string, number | null>;
  r2: number | null;
  residual: number | null;
  genes: number;
  iterations: number;
  converged: boolean;
  /** Types flagged in this sample (see DeconvResult.flags) */
  flagged: string[];
}

export interface DeconvResult {
  result_id: string;
  method: "weighted" | "nnls";
  method_name: string;
  method_requested: DeconvMethod;
  method_reason: string;
  celltype_column: string;
  donor_column: string | null;
  donor_reason: string | null;
  types: string[];
  /** Mean counts per cell of each type */
  cell_size: (number | null)[];
  /** Donors with cells of each type */
  n_donors: number[];
  n_reference_cells: number;
  n_reference_donors: number;
  counts_source: string;
  filename: string;
  n_samples: number;
  samples: DeconvSample[];
  n_genes_file: number;
  n_genes_matched: number;
  /** Share of the reference's genes found in the bulk file */
  gene_share: number | null;
  n_genes_used: number;
  marker_source: string | null;
  id_kind: "ensembl" | "symbol";
  input_kind: "counts" | "tpm" | "non-integer";
  warnings: string[];
  notes: string[];
  elapsed_seconds: number;
  created_at: number;
  reference_cells: ReferenceCells | null;
  /** Each type's share of the reference cells */
  type_share: (number | null)[];
  flags: DeconvFlag[];
}

export interface CheckMetric {
  r: number | null;
  rmse: number | null;
  bias: number | null;
  n: number;
}

export interface DeconvCheck {
  result_id: string;
  celltype_column: string;
  donor_column: string;
  methods: ("weighted" | "nnls")[];
  method_names: Record<string, string>;
  n_mixtures: number;
  seed: number;
  donors: string[];
  types: string[];
  overall: (CheckMetric & { method: string })[];
  per_type: (CheckMetric & { method: string; cell_type: string })[];
  /** Per method, columnar: type index, truth, estimate, the donor's own pseudobulk or a mixture */
  points: Record<string, { type: number[]; truth: (number | null)[]; estimate: (number | null)[]; pseudobulk: boolean[] }>;
  points_capped: boolean;
  /** Points per method before the cap */
  n_points: Record<string, number>;
  notes: string[];
  elapsed_seconds: number;
  created_at: number;
  reference_cells: ReferenceCells | null;
}

export interface CheckProgress {
  done: number;
  total: number;
  donor: string | null;
  estimate_seconds?: number;
}

async function orNull<T>(p: Promise<T>): Promise<T | null> {
  try {
    return await p;
  } catch (e) {
    if (e instanceof Error && /API error 404/.test(e.message)) return null;
    throw e;
  }
}

/** The last run on the dataset, or null when none has run. */
export function getDeconvolution(datasetId: string): Promise<DeconvResult | null> {
  return orNull(apiFetch<DeconvResult>(`/datasets/${datasetId}/deconvolution`));
}

export function getDeconvolutionDefaults(datasetId: string): Promise<DeconvDefaults> {
  return apiFetch(`/datasets/${datasetId}/deconvolution/defaults`);
}

/** POST /deconvolution/upload (multipart). Without a file the server uses the last run's file again. */
export async function runDeconvolution(datasetId: string, file: File | null, options: DeconvOptions): Promise<DeconvResult> {
  const form = new FormData();
  if (file) form.append("file", file);
  form.append("options", JSON.stringify(options));
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/deconvolution/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await readError(res)}`);
  return res.json();
}

export function getDeconvolutionCheck(datasetId: string): Promise<DeconvCheck | null> {
  return orNull(apiFetch<DeconvCheck>(`/datasets/${datasetId}/deconvolution/check`));
}

/** The leave-one-donor-out self-check, streamed; resolves with its result. */
export async function runDeconvolutionCheck(
  datasetId: string,
  body: DeconvOptions & { n_mixtures?: number; seed?: number },
  handlers: { onProgress?: (p: CheckProgress) => void; signal?: AbortSignal } = {},
): Promise<DeconvCheck> {
  let result: DeconvCheck | null = null;
  await streamEvents(
    `/datasets/${datasetId}/deconvolution/validate`,
    body,
    (event, data) => {
      if (event === "progress") handlers.onProgress?.(data as CheckProgress);
      else if (event === "result") result = data as DeconvCheck;
    },
    "The self-check",
    handlers.signal,
  );
  if (!result) throw new Error("The self-check ended without a result.");
  return result;
}

export function deconvolutionCsvUrl(datasetId: string, resultId: string): string {
  return `${API_BASE}/datasets/${datasetId}/deconvolution/${resultId}.csv`;
}

export function referenceZipUrl(datasetId: string, options: DeconvOptions = {}): string {
  const q = new URLSearchParams();
  if (options.celltype_column) q.set("celltype_column", options.celltype_column);
  if (options.donor_column != null) q.set("donor_column", options.donor_column);
  if (options.cells_column && options.cells_value != null) {
    q.set("cells_column", options.cells_column);
    q.set("cells_value", options.cells_value);
  }
  const s = q.toString();
  return `${API_BASE}/datasets/${datasetId}/deconvolution/reference.zip${s ? `?${s}` : ""}`;
}
