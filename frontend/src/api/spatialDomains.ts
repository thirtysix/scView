import { apiFetch } from "./client";
import { streamEvents } from "./clustering";
import type { WatchdogEvent } from "./pipelineRun";

// --- Spatial domains and niches (backend core/spatial_domains.py, api/v1/spatial_domains.py) ---

/** What a domains run would use and need (backend DomainPlan). */
export interface DomainPlan {
  n_obs: number;
  technology: string;
  unit: string;
  libraries: string[];
  /** X_pca, or X_pca_harmony with more than one section */
  rep: string;
  n_pcs: number;
  n_pcs_available: number;
  n_hops: number;
  dims: number;
  k: number | null;
  k_min: number;
  k_max: number;
  repeats: number;
  seed: number;
  n_fit: number;
  graph: string;
  notes: string[];
  estimate_seconds: number;
  estimate_scan_seconds: number;
  estimate_bytes: number;
}

export interface StabilityCurve {
  ks: number[];
  mean: number[];
  sd: number[];
  peaks: number[];
  suggested: number;
  has_peak: boolean;
  repeats: number;
  n_fit: number;
}

export interface DomainsResult {
  available: boolean;
  columns: string[];
  scan?: StabilityCurve | null;
  column?: string;
  k?: number;
  suggested?: boolean;
  unit?: string;
  rep?: string;
  n_pcs?: number;
  n_hops?: number;
  n_obs?: number;
  n_fit?: number;
  graph?: string;
  n_links?: number;
  n_isolated?: number;
  libraries?: string[];
  notes?: string[];
  sizes?: number[];
  per_library?: Record<string, number[]>;
  coherence?: { within: number; shuffled: number };
  seconds?: number;
  method?: string;
  stale?: boolean;
  stale_reason?: string | null;
  run_id?: string;
}

export interface EnrichmentResult {
  available: boolean;
  columns: string[];
  column?: string;
  labels?: string[];
  sizes?: number[];
  zscore?: (number | null)[][];
  count?: number[][];
  n_perms?: number;
  n_obs?: number;
  n_labelled?: number;
  unit?: string;
  graph?: string;
  notes?: string[];
  seconds?: number;
  stale?: boolean;
  stale_reason?: string | null;
}

export interface NichesResult {
  available: boolean;
  columns: string[];
  column?: string;
  label_column?: string;
  labels?: string[];
  niches?: string[];
  fractions?: (number | null)[][];
  sizes?: number[];
  k_window?: number;
  n_niches?: number;
  n_labelled?: number;
  unit?: string;
  notes?: string[];
  seconds?: number;
  stale?: boolean;
  stale_reason?: string | null;
}

export interface DomainsInfo {
  runnable: { ok: boolean; reason: string | null; plan: DomainPlan | null };
  defaults: {
    k: number;
    n_hops: number;
    n_pcs: number;
    k_min: number;
    k_max: number;
    repeats: number;
    max_fit_obs: number;
    n_perms: number;
    k_window: number;
    n_niches: number;
    domain_column: string;
    niche_column: string;
  };
  n_obs: number;
  /** A name no column holds yet, for the k asked about */
  free_column: string;
  free_niche_column: string;
  niches_allowed: { ok: boolean; reason: string | null; note: string | null };
  estimates: { enrichment_seconds: number; enrichment_bytes: number };
  domains: DomainsResult;
  niches: NichesResult;
  enrichment: EnrichmentResult;
  running: boolean;
}

export function getSpatialDomains(
  datasetId: string,
  params: { k?: number; n_hops?: number; n_pcs?: number; enrichment_column?: string } = {},
): Promise<DomainsInfo> {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") q.set(key, String(value));
  }
  const text = q.toString();
  return apiFetch(`/datasets/${datasetId}/spatial/domains${text ? `?${text}` : ""}`);
}

export interface RunStage {
  stage: string;
  label: string;
  index: number;
  total: number;
  elapsed_s: number;
}

/** One finished fit of a k scan, or a block of permutations. */
export interface RunProgress {
  done: number;
  total: number;
  elapsed_s: number;
  k?: number;
  run?: number;
}

export interface RunHandlers {
  onStart?: (s: { estimate_seconds?: number; estimate_scan_seconds?: number; k?: number | null }) => void;
  onStage?: (s: RunStage) => void;
  onProgress?: (p: RunProgress) => void;
  onCurve?: (c: StabilityCurve) => void;
  onWatchdog?: (w: WatchdogEvent) => void;
  signal?: AbortSignal;
}

async function run<T>(path: string, body: unknown, what: string, handlers: RunHandlers): Promise<T> {
  let result: T | null = null;
  await streamEvents(
    path,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as never);
      else if (event === "stage") handlers.onStage?.(data as RunStage);
      else if (event === "scan" || event === "progress") handlers.onProgress?.(data as RunProgress);
      else if (event === "curve") handlers.onCurve?.((data as { stability: StabilityCurve }).stability);
      else if (event === "watchdog") handlers.onWatchdog?.(data as WatchdogEvent);
      else if (event === "result") result = data as T;
    },
    what,
    handlers.signal,
  );
  if (!result) throw new Error(`${what} ended without a result; nothing was saved.`);
  return result;
}

export interface DomainsRunBody {
  /** null: scan the stability of each k and use its best peak */
  k: number | null;
  n_hops?: number;
  n_pcs?: number;
  column: string;
  overwrite?: boolean;
}

export function runDomains(datasetId: string, body: DomainsRunBody, handlers: RunHandlers = {}): Promise<DomainsResult> {
  return run(`/datasets/${datasetId}/spatial/domains/run-stream`, body, "The domains run", handlers);
}

export function runEnrichment(
  datasetId: string,
  body: { column: string; n_perms?: number },
  handlers: RunHandlers = {},
): Promise<EnrichmentResult> {
  return run(`/datasets/${datasetId}/spatial/nhood-enrichment/run-stream`, body, "Neighbourhood enrichment", handlers);
}

export function runNiches(
  datasetId: string,
  body: { label_column: string; k_window?: number; n_niches?: number; column: string; overwrite?: boolean },
  handlers: RunHandlers = {},
): Promise<NichesResult> {
  return run(`/datasets/${datasetId}/spatial/niches/run-stream`, body, "Composition niches", handlers);
}
