import { apiFetch } from "./client";

/** How a load or a run compares with the memory scView has free (core/preflight.py). */
export type PreflightLevel = "fits" | "tight" | "wont_fit" | "unknown";

export interface PreflightVerdict {
  level: PreflightLevel;
  /** The one plain sentence: peak, where it is, rough time, memory free. */
  sentence: string;
  /** Memory needed at the peak over what scView uses now */
  need_bytes: number;
  /** All the memory this dataset takes at the peak (some of it may be in use already) */
  peak_bytes: number;
  /** Where the peak is: a step label, or "reading the file" */
  peak_stage: string;
  /** limit - use now, file cache the system can drop not counted as use */
  available_bytes: number | null;
  limit_bytes: number | null;
  used_bytes: number | null;
  limited: boolean;
  source: "container limit" | "machine" | "unknown";
  /** Rough time as [low, high] seconds */
  seconds: [number, number];
  /** Other datasets scView holds in memory, and whether closing them would free enough */
  cached: { count: number; bytes: number; frees_enough: boolean };
  /** Concrete steps for a load or run that may not fit, most useful first */
  suggestions: string[];
  /** Measured error of the estimate on the calibration datasets, e.g. 0.3 = within 30% */
  error_band: number;
  /** For the tooltip: what the estimate adds up */
  breakdown: { label: string; bytes: number }[];
  /** What the numbers rest on (estimated rows, a guess, steps not measured, time caveat) */
  notes: string[];
}

export interface PreflightStep {
  step: string;
  label: string;
  /** Memory over what the dataset holds when the step starts, at the step's peak */
  transient_bytes: number;
  /** What the dataset holds when the step starts */
  resident_bytes: number;
  peak_bytes: number;
  seconds: [number, number];
  /** False for steps whose estimate was not calibrated by a measured run */
  measured: boolean;
}

/** A run that clusters with Leiden: its time both ways, and whether to offer fast Leiden. */
export interface LeidenChoice {
  /** Cells clustered (after filtering's share, when the run filters) */
  n_cells: number;
  /** Point estimates; Leiden uses one CPU */
  full_seconds: number;
  fast_seconds: number;
  /** As ranges in words: "roughly 20 to 80 min" */
  full_text: string;
  fast_text: string;
  /** The full optimisation's estimate over this offers fast Leiden (600 s by default) */
  offer_seconds: number;
  offer_fast: boolean;
  /** The run's setting now: -1 full optimisation, 2 fast Leiden */
  iterations: number;
}

/** A run with the UMAP step: whether it makes the 3D UMAP, and the time that adds. */
export interface Umap3dChoice {
  /** Cells laid out (after filtering's share, when the run filters) */
  n_cells: number;
  /** The run's setting: auto (off in large-dataset mode), on or off */
  setting: "auto" | "on" | "off";
  /** Whether this run makes it */
  on: boolean;
  /** What auto means for this run (false in large-dataset mode) */
  default_on: boolean;
  extra_seconds: number;
  /** As a range in words: "roughly 10 to 40 min" */
  extra_text: string;
  /** Show the note: large-dataset mode, or the 3D UMAP alone over 10 minutes */
  offer: boolean;
}

export interface RunPreflight {
  dataset_id: string;
  steps: PreflightStep[];
  verdict: PreflightVerdict;
  /** Null when the run does not cluster with Leiden (missing from an older backend) */
  leiden?: LeidenChoice | null;
  /** Null when the run has no UMAP step (missing from an older backend) */
  umap_3d?: Umap3dChoice | null;
}

export interface UnitPreflight {
  label: string;
  format: string;
  verdict: PreflightVerdict;
}

export interface IngestPreflight {
  session_id: string;
  units: UnitPreflight[];
  /** The commit as a whole when several samples are merged, else null */
  merge: PreflightVerdict | null;
}

/** Pipeline parameters the run estimate depends on; others are ignored. */
const RUN_PARAMS = [
  "n_top_genes", "n_comps", "n_neighbors", "normalization_method", "annotation_method",
  "doublet_channel_key", // doublet detection's peak is its largest channel's
  "clustering_method", "leiden_iterations", // clustering's time, and the fast-Leiden offer
  "umap_3d", // UMAP's time, and the 3D UMAP note
  "regress_out", // scaling's fit (a list, sent comma-separated)
  "ambient_rho", // set: the ambient RNA correction estimates nothing (no clustering of its own)
] as const;

export function getRunPreflight(
  datasetId: string,
  steps: string[],
  params: Record<string, unknown> | null = null,
  signal?: AbortSignal,
): Promise<RunPreflight> {
  const q = new URLSearchParams({ steps: steps.join(",") });
  for (const key of RUN_PARAMS) {
    const v = params?.[key];
    if (typeof v === "number" || (typeof v === "string" && v)) q.set(key, String(v));
    else if (Array.isArray(v) && v.length) q.set(key, v.map(String).join(","));
  }
  return apiFetch<RunPreflight>(`/datasets/${datasetId}/preflight?${q}`, { signal });
}

export function getIngestPreflight(sessionId: string, signal?: AbortSignal): Promise<IngestPreflight> {
  return apiFetch<IngestPreflight>(`/ingest/session/${sessionId}/preflight`, { signal });
}
