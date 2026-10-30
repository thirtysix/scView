import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** Spot composition: Visium spots split into a reference's cell types (backend
 * core/spot_deconvolution.py, api/v1/spot_deconvolution.py). */
export type SpotMethod = "rctd" | "nnls";
export type ShareKind = "rna" | "cell";

export interface SpotTypeStats {
  type: string;
  n_reference_cells: number;
  cell_size: number;
  rna_mean: number | null;
  rna_max: number | null;
  cell_mean: number | null;
  cell_max: number | null;
  /** Spots where this type has the largest RNA share */
  n_top: number;
}

export interface SpotLibrary {
  id: string;
  n_spots: number;
  n_fitted: number;
  /** Spread of the per-gene platform effect (log scale); null when it could not be fitted */
  s_g: number | null;
  /** Section-wide RNA shares, in the order of `types` */
  bulk: number[] | null;
}

export interface SpotSummary {
  result_id: string;
  method: SpotMethod;
  method_name: string;
  types: string[];
  type_stats: SpotTypeStats[];
  small_types: string[];
  reference: { id: string; name: string; celltype_column: string; n_reference_cells: number };
  celltype_column: string;
  sources: { reference: string; spots: string; renamed: string | null };
  s_e: number | null;
  libraries: SpotLibrary[];
  n_shared_genes: number;
  n_genes: number;
  n_platform_genes: number;
  n_spots: number;
  n_fitted: number;
  n_skipped: number;
  n_unconverged: number;
  n_clipped: number;
  cluster_column: string | null;
  /** Mean share per cluster, in the order of `types` */
  clusters: { rna: Record<string, number[]>; cells: Record<string, number[]> };
  columns: { rna: string; cells: string; top: string; top_share: string };
  seconds: number | null;
  notes: string[];
  note: string;
}

export interface SpotInfo {
  methods: SpotMethod[];
  method: SpotMethod;
  method_names: Record<SpotMethod, string>;
  min_type_cells: number;
  max_types: number;
  min_spot_umi: number;
  min_shared_genes: number;
  available: boolean;
  reason: string | null;
  saved: SpotSummary | null;
  n_spots: number;
  running: boolean;
  note: string;
}

export interface SpotReferenceCheck {
  ok: boolean;
  reason: string | null;
  n_cells?: number;
  n_genes?: number;
  label_columns?: { column: string; n_labels: number }[];
  celltype_column?: string | null;
  types?: { type: string; n_cells: number; small: boolean }[];
  n_small?: number;
  species?: string;
  dataset_species?: string;
  shared_genes_by_name?: number;
  min_shared_genes?: number;
  estimate?: {
    peak_bytes: number;
    seconds: number;
    seconds_range: [number, number];
    note: string;
    // A large dataset (matrices on disk): its counts are read from disk in one pass.
    large_reference?: boolean;
  };
}

export interface SpotCheckRun {
  method: SpotMethod;
  depth: number;
  s_e: number | null;
  median_umi: number;
  n_genes: number;
  rmse: number | null;
  jsd: number | null;
  aupr: number | null;
  rmse_cells?: number | null;
  n_spots: number;
  per_type: { type: string; r: number | null; bias: number | null; rmse: number | null }[];
}

export interface SpotCheck {
  id: string;
  types: string[];
  runs: SpotCheckRun[];
  n_spots: number;
  cells_per_spot: [number, number];
  depths: number[];
  reference: { id: string; name: string; celltype_column: string };
  seconds: number;
  note: string;
}

const base = (datasetId: string) => `/datasets/${datasetId}/spot-deconvolution`;

export function getSpotDeconvolution(datasetId: string): Promise<SpotInfo> {
  return apiFetch(base(datasetId));
}

export function checkSpotReference(
  datasetId: string,
  referenceId: string,
  celltypeColumn: string | null,
  method: SpotMethod,
): Promise<SpotReferenceCheck> {
  const q = new URLSearchParams({ method });
  if (celltypeColumn) q.set("celltype_column", celltypeColumn);
  return apiFetch(`${base(datasetId)}/reference/${encodeURIComponent(referenceId)}?${q}`);
}

export function getSpotShare(datasetId: string, type: string, kind: ShareKind): Promise<{ values: (number | null)[] }> {
  const q = new URLSearchParams({ type, kind });
  return apiFetch(`${base(datasetId)}/share?${q}`);
}

/** The last self-check, or null when none has run (allow_empty: no 404 in the console). */
export function getSpotCheck(datasetId: string): Promise<SpotCheck | null> {
  return apiFetch(`${base(datasetId)}/check?allow_empty=true`);
}

interface StreamHandlers {
  onProgress?: (message: string, done?: number, total?: number) => void;
  signal?: AbortSignal;
}

async function stream<T>(path: string, body: unknown, what: string, handlers: StreamHandlers): Promise<T> {
  let result: T | null = null;
  await streamEvents(
    path,
    body,
    (event, data) => {
      const d = data as { message?: string; done?: number; total?: number };
      if (event === "progress" && d.message) handlers.onProgress?.(d.message, d.done, d.total);
      else if (event === "result") result = data as T;
    },
    what,
    handlers.signal,
  );
  if (!result) throw new Error(`${what} ended without a result; nothing was saved.`);
  return result;
}

/** Fit the spots (SSE); resolves with the saved summary. */
export function runSpotDeconvolution(
  datasetId: string,
  body: { reference_id: string; celltype_column: string; method: SpotMethod },
  handlers: StreamHandlers = {},
): Promise<SpotSummary> {
  return stream(`${base(datasetId)}/run-stream`, body, "The spot composition run", handlers);
}

/** The self-check on synthetic spots from held-out reference cells (SSE). */
export function runSpotCheck(
  datasetId: string,
  body: { reference_id: string; celltype_column: string },
  handlers: StreamHandlers = {},
): Promise<SpotCheck> {
  return stream(`${base(datasetId)}/check-stream`, body, "The spot composition self-check", handlers);
}
