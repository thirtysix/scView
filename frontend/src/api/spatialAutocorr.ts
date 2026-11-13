import { apiFetch } from "./client";
import { API_BASE } from "@/lib/constants";
import { streamEvents } from "./clustering";

// --- Spatially variable genes: Moran's I per gene (backend core/spatial_autocorr.py,
// api/v1/spatial_autocorr.py) ---

export interface MoranRow {
  gene: string;
  I: number | null;
  expected_I: number | null;
  var_I: number | null;
  z: number | null;
  pval: number | null;
  pval_adj: number | null;
  /** Several sections: each section's own values (a section where the gene is constant is absent) */
  by_library?: Record<string, MoranSectionValues>;
}

export interface MoranSectionValues {
  I: number | null;
  z: number | null;
  pval: number | null;
  pval_adj: number | null;
}

export interface MoranResult {
  result_id: string;
  method: "moran";
  n_obs: number;
  n_genes: number;
  neighbors: { kind: "grid" | "knn"; k: number };
  libraries: string[];
  /** The genes actually tested: "hvg" falls back to "all" without variable genes */
  genes: "hvg" | "all" | "list";
  matrix_source: string;
  unit: "spot" | "bin" | "cell";
  n_constant: number;
  /** The section the run was on, or "all" */
  library_id: string;
  /** "section": several sections pooled, each centred on its own mean; "global": one section */
  centring: "section" | "global";
  /** Several sections: each one's size */
  sections: { library_id: string; n_obs: number }[];
  notes: string[];
  /** Rows sent (highest I first); the CSV has every gene */
  rows_shown: number;
  /** I > 0 with adjusted p < 0.05 */
  n_significant: number;
  /** I < 0 with adjusted p < 0.05 */
  n_dispersed: number;
  elapsed_seconds: number;
  created_at: number;
  rows: MoranRow[];
}

export interface MoranRunBody {
  genes: "hvg" | "all" | string[];
  library_id?: string | null;
  layer?: string | null;
}

/**
 * The last run on the dataset, or null when none has run (allow_empty, so no 404). With
 * `libraryId`: the last run on that section ("all": over every section).
 */
export async function getSpatialAutocorr(datasetId: string, libraryId?: string): Promise<MoranResult | null> {
  const section = libraryId ? `&library_id=${encodeURIComponent(libraryId)}` : "";
  try {
    return await apiFetch<MoranResult | null>(`/datasets/${datasetId}/spatial/autocorr?allow_empty=true${section}`);
  } catch (e) {
    if (e instanceof Error && /API error 404/.test(e.message)) return null;
    throw e;
  }
}

export interface MoranProgress {
  done: number;
  total: number;
}

/**
 * Run it as a stream (POST .../spatial/autocorr/run-stream): all genes on a large dataset take
 * many minutes, past what a proxy waits for a silent response. Resolves with the saved result.
 */
export async function runSpatialAutocorr(
  datasetId: string,
  body: MoranRunBody,
  onProgress?: (p: MoranProgress) => void,
): Promise<MoranResult> {
  const out: { result?: MoranResult } = {};
  await streamEvents(
    `/datasets/${datasetId}/spatial/autocorr/run-stream`,
    body,
    (event, data) => {
      if (event === "progress") onProgress?.(data as MoranProgress);
      else if (event === "result") out.result = data as MoranResult;
    },
    "Spatially variable genes",
  );
  if (!out.result) throw new Error("Spatially variable genes: the run ended without a result.");
  return out.result;
}

export function spatialAutocorrCsvUrl(datasetId: string, resultId: string): string {
  return `${API_BASE}/datasets/${datasetId}/spatial/autocorr/${resultId}.csv`;
}

/** Seconds a run takes on 2 cores (backend spatial_autocorr.SECONDS_PER_VALUE, LARGE_SECONDS_PER_VALUE). */
export function estimateSeconds(nObs: number, nGenes: number, large = false): number {
  return Math.max(0.2, (large ? 2.3e-9 : 1.3e-8) * nObs * nGenes);
}
