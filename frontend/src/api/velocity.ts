import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";
import { streamEvents } from "./clustering";

/** RNA velocity (backend core/velocity_input.py, core/velocity.py, api/v1/velocity.py). */
export type VelocityMethod = "stochastic" | "steady_state";

export interface VelocitySource {
  kind?: string;
  file?: string | null;
  files?: string[];
  matrices?: string[];
  cells_in_file?: number;
  cells_matched?: number;
  cells_in_dataset?: number;
  cells_without?: number | null;
  cells_ambiguous?: number;
  sample_used?: boolean;
  genes_in_file?: number;
  genes_by_id?: number;
  genes_by_name?: number;
  /** Cells of the dataset as it is now, with and without velocity counts */
  cells_now?: number;
  cells_without_now?: number;
}

export interface VelocityGroup {
  group: string;
  n_cells: number;
  /** Mean velocity confidence of the group's cells */
  coherence: number;
  length: number;
  unspliced_fraction: number;
  n_velocity_genes: number;
  top_genes: string[];
}

export interface VelocityRun {
  result_id: string;
  method: VelocityMethod;
  method_name: string;
  params: { n_top_genes: number; n_neighbors: number; n_pcs: number; min_shared_counts: number };
  n_cells: number;
  n_dataset_cells: number;
  cells_without: number;
  n_genes_store: number;
  n_genes_kept: number;
  n_genes_fit: number;
  n_velocity_genes: number;
  median_confidence: number;
  median_length: number;
  unspliced_fraction: number;
  neighbours_note: string;
  pca_note: string;
  group_column: string | null;
  clusters: VelocityGroup[];
  warnings: string[];
  note: string;
  embeddings: string[];
}

export interface VelocityEstimate {
  peak_bytes: number;
  seconds: number;
  seconds_range: [number, number];
  note: string;
  n_cells: number;
}

export interface VelocityStatus {
  attached: boolean;
  source: VelocitySource | null;
  run: VelocityRun | null;
  /** Set when the cells changed since the run: it says what to do */
  stale: string | null;
  methods: VelocityMethod[];
  method_names: Record<VelocityMethod, string>;
  defaults: { method: VelocityMethod; n_top_genes: number; n_neighbors: number; n_pcs: number };
  note: string;
  n_cells: number;
  estimate: VelocityEstimate | null;
  running: boolean;
}

export interface VelocityGrid {
  embedding: string;
  density: number;
  n: number;
  x: number[];
  y: number[];
  u: number[];
  v: number[];
  /** The grid's spacing in embedding units; arrows are scaled against it */
  step: number;
  length_p95: number;
  /** Projected now: the embedding is not the one the run saw */
  reprojected: boolean;
}

export type VelocityColor =
  | { column: string; kind: "category"; categories: string[]; codes: number[] }
  | { column: string; kind: "numeric"; values: (number | null)[] };

export interface VelocityPhase {
  gene: string;
  gamma: number;
  r2: number;
  velocity_gene: boolean;
  spliced: number[];
  unspliced: number[];
  n_points: number;
  n_cells: number;
  capped: boolean;
  max_points: number;
  color: VelocityColor | null;
  method: VelocityMethod;
}

export interface VelocityGene {
  gene: string;
  gamma: number;
  r2: number;
  velocity_gene: boolean;
}

export function getVelocity(datasetId: string): Promise<VelocityStatus> {
  return apiFetch(`/datasets/${datasetId}/velocity`);
}

export async function attachVelocity(datasetId: string, files: File[]): Promise<{ attached: boolean; source: VelocitySource }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/velocity/attach`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await readError(res)}`);
  return res.json();
}

export interface VelocityRunBody {
  method: VelocityMethod;
  n_top_genes: number;
  n_neighbors: number;
  group_by?: string | null;
}

export async function runVelocity(
  datasetId: string,
  body: VelocityRunBody,
  handlers: { signal?: AbortSignal; onProgress?: (message: string) => void; onWatchdog?: (message: string) => void } = {},
): Promise<VelocityRun> {
  let run: VelocityRun | null = null;
  await streamEvents(
    `/datasets/${datasetId}/velocity/run-stream`,
    body,
    (event, data) => {
      const d = data as { message?: string; run?: VelocityRun };
      if (event === "progress" && d.message) handlers.onProgress?.(d.message);
      else if (event === "watchdog" && d.message) handlers.onWatchdog?.(d.message);
      else if (event === "result" && d.run) run = d.run;
    },
    "RNA velocity",
    handlers.signal,
  );
  if (!run) throw new Error("RNA velocity ended without a result.");
  return run;
}

export function getVelocityGrid(datasetId: string, embedding: string, density: number, signal?: AbortSignal): Promise<VelocityGrid> {
  const q = new URLSearchParams({ embedding, density: String(density) });
  return apiFetch(`/datasets/${datasetId}/velocity/grid?${q}`, { signal });
}

export function getVelocityPhase(datasetId: string, gene: string, colorBy?: string | null): Promise<VelocityPhase> {
  const q = new URLSearchParams({ gene });
  if (colorBy) q.set("color_by", colorBy);
  return apiFetch(`/datasets/${datasetId}/velocity/phase?${q}`);
}

export function getVelocityGenes(datasetId: string): Promise<{ genes: VelocityGene[]; n_genes: number; n_velocity_genes: number; sort: string }> {
  return apiFetch(`/datasets/${datasetId}/velocity/genes`);
}
