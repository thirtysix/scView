import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** Whether a resolution sweep can run on a dataset, and its limits. */
export interface SweepInfo {
  available: boolean;
  /** Why not, when it can't (e.g. no neighbour graph) */
  reason: string | null;
  method: "leiden" | "louvain";
  n_cells: number;
  default_resolutions: number[];
  max_resolutions: number;
  min_resolution: number;
  max_resolution: number;
  max_cells: number;
  silhouette_cells: number;
}

export interface SweepStart {
  method: "leiden" | "louvain";
  n_cells: number;
  resolutions: number[];
  /** obsm key the silhouette is measured in, or null when there is none */
  representation: string | null;
  silhouette_cells: number | null;
}

export interface SweepRow {
  resolution: number;
  n_clusters: number;
  /** Cluster sizes, largest first */
  sizes: number[];
  silhouette: number | null;
  /** Adjusted Rand index to the previous (lower) resolution; null for the first */
  ari_previous: number | null;
  /** The column "Keep" saves it as */
  column: string;
  /** That column exists already */
  exists: boolean;
}

export function getSweepInfo(datasetId: string): Promise<SweepInfo> {
  return apiFetch(`/datasets/${datasetId}/clustering/sweep`);
}

/**
 * Cluster at each resolution without changing the dataset, calling `onStart`
 * once and `onResult` per resolution as each one finishes. Rejects with the
 * server's reason; aborting `signal` stops reading.
 */
export async function runSweep(
  datasetId: string,
  body: { resolutions: number[]; method?: string },
  handlers: { onStart?: (s: SweepStart) => void; onResult: (r: SweepRow) => void; signal?: AbortSignal },
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/clustering/sweep`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as SweepStart);
      else if (event === "result") handlers.onResult(data as SweepRow);
    },
    "The sweep",
    handlers.signal,
  );
}

/**
 * POST `body` to a server-sent-events endpoint and hand each event to `onEvent`
 * as it arrives. Rejects with the server's reason, on an `error` event, or when
 * the stream ends before `complete`; aborting `signal` stops reading.
 */
export async function streamEvents(
  path: string,
  body: unknown,
  onEvent: (event: string, data: unknown) => void,
  what: string,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let detail = await readError(res);
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");

  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let complete = false;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ") && event) {
        const data = JSON.parse(line.slice(6));
        if (event === "error") throw new Error(String(data.error));
        if (event === "complete") complete = true;
        else onEvent(event, data);
        event = "";
      }
    }
  }
  if (!complete) throw new Error(`${what} stopped before it finished.`);
}

/** Save one resolution's clusters as its scview_<method>_r<res> column (once). */
export function keepResolution(
  datasetId: string,
  resolution: number,
  method?: string,
): Promise<{ column: string; created: boolean; n_clusters: number }> {
  return apiFetch(`/datasets/${datasetId}/clustering/keep`, {
    method: "POST",
    body: JSON.stringify({ resolution, method }),
  });
}

export interface SubclusterResult {
  column: string;
  group: string;
  n_cells: number;
  n_subclusters: number;
  groups: string[];
  sizes: number[];
  representation: string;
  batch_key: string | null;
}

/** Split one group of a column on its own graph, saved as `<column>_sub`. */
export function subclusterGroup(
  datasetId: string,
  column: string,
  group: string,
  resolution = 0.5,
): Promise<SubclusterResult> {
  return apiFetch(`/datasets/${datasetId}/clustering/subcluster`, {
    method: "POST",
    body: JSON.stringify({ column, group, resolution }),
  });
}

/** Distances between a column's groups in the PCs the neighbour graph used. */
export interface GroupDistances {
  column: string;
  /** Groups with cells, in the column's own order */
  groups: string[];
  sizes: number[];
  /** distances[i][j]: Euclidean distance between the centroids of groups i and j */
  distances: number[][];
  /** Group indices in complete-linkage leaf order (similar groups adjacent) */
  order: number[];
  /** obsm key of the PCs (X_pca_harmony when the graph was built on them) */
  representation: string;
  n_pcs: number;
  /** graph: the graph's own space; missing: the graph's space is not in the
   *  file; no_graph: no neighbour graph recorded */
  rep_source: "graph" | "missing" | "no_graph";
  linkage: string;
  metric: string;
  /** Cells without a value (or with non-finite PCs), left out */
  n_left_out: number;
  /** The clustering behind `order`: scipy linkage rows [a, b, height, size] */
  tree?: [number, number, number, number][];
  /** correlation[i][j]: Pearson correlation of the centroids of groups i and j */
  correlation?: number[][];
  correlation_method?: string;
}

export function getGroupDistances(datasetId: string, column: string): Promise<GroupDistances> {
  return apiFetch(`/datasets/${datasetId}/clustering/distances?column=${encodeURIComponent(column)}`);
}

// --- Variable-gene sensitivity check (backend core/hvg_sensitivity.py) ---

export type HvgFlavor = "seurat" | "cell_ranger" | "seurat_v3" | "pearson_residuals";

/** One variable-gene setting: a flavour and how many genes. */
export interface HvgSetting {
  flavor: HvgFlavor;
  n_top_genes: number;
  /** `${flavor}_${n_top_genes}` */
  key: string;
  /** "Seurat, 2,000 genes" */
  label: string;
}

/** How the dataset's clustering was made, as provenance recorded it. */
export interface HvgRecorded {
  /** The obs column compared with */
  reference: string;
  method: "leiden" | "louvain";
  resolution: number;
  flavor: HvgFlavor;
  n_top_genes: number;
  hvg_batch_key: string | null;
  target_sum: number;
  n_comps: number;
  n_neighbors: number;
  /** Harmony's key, when the neighbour graph was batch-corrected */
  batch_key: string | null;
  /** What had to be assumed */
  notes: string[];
  /** The reference came with the file, not from scView's clustering */
  reference_from_file: boolean;
  /** scView version and date that made the graph's PCs (null: not recorded) */
  pcs_made_by: string | null;
  pcs_made_on: string | null;
  /** Made by an older scView: the re-runs build the PCs with this version's pipeline */
  pcs_older: boolean;
}

export interface HvgSensitivityInfo {
  available: boolean;
  reason: string | null;
  n_cells: number;
  n_genes: number;
  max_settings: number;
  min_genes: number;
  max_genes: number;
  max_cells: number;
  /** Extra seconds the first run after a server start takes (compiling) */
  first_run_seconds: number;
  // Present when available:
  recorded?: HvgRecorded;
  current?: HvgSetting;
  default_settings?: HvgSetting[];
  /** Each flavour, with why it can't run here (null when it can) */
  flavors?: { flavor: HvgFlavor; label: string; unavailable: string | null }[];
  estimate_seconds?: number;
  seconds_per_setting?: number;
}

export interface HvgStart {
  n_cells: number;
  reference: string;
  recorded: HvgRecorded;
  current: HvgSetting;
  settings: HvgSetting[];
  n_current_hvgs: number | null;
  estimate_seconds: number;
}

export type HvgClusterStatus = "stable" | "splits" | "merges" | "dissolves";

/** What became of one current cluster under a setting. */
export interface HvgClusterMatch {
  cluster: string;
  n_cells: number;
  /** Best-match Jaccard index in the new clustering */
  jaccard: number;
  /** The new cluster it matches best */
  match: string;
  /** Share of its cells in the new cluster holding most of them */
  contained: number;
  /** Share of that new cluster made of its cells */
  purity: number;
  /** New clusters holding at least 10% of its cells */
  parts: number;
  status: HvgClusterStatus;
}

export interface HvgRow extends HvgSetting {
  /** The current setting, re-run as a control */
  is_current: boolean;
  n_hvgs: number;
  /** Variable genes shared with the current ones (null when the dataset has none marked) */
  shared_hvgs: number | null;
  ari: number;
  ami: number;
  n_clusters: number;
  sizes: number[];
  clusters: HvgClusterMatch[];
  seconds: number;
  /** The column "Keep" saves it as */
  column: string;
  exists: boolean;
}

export function getHvgSensitivityInfo(datasetId: string): Promise<HvgSensitivityInfo> {
  return apiFetch(`/datasets/${datasetId}/clustering/hvg-sensitivity`);
}

/**
 * Re-cluster under each variable-gene setting without changing the dataset,
 * calling `onStart` once and `onResult` per setting. Rejects with the server's
 * reason; aborting `signal` stops reading and, on the server, the run.
 */
export async function runHvgSensitivity(
  datasetId: string,
  body: { settings: { flavor: HvgFlavor; n_top_genes: number }[] },
  handlers: { onStart?: (s: HvgStart) => void; onResult: (r: HvgRow) => void; signal?: AbortSignal },
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/clustering/hvg-sensitivity`,
    body,
    (event, data) => {
      if (event === "start") handlers.onStart?.(data as HvgStart);
      else if (event === "result") handlers.onResult(data as HvgRow);
    },
    "The check",
    handlers.signal,
  );
}

/** Save one setting's clusters as a column (once). */
export function keepHvgSetting(
  datasetId: string,
  setting: { flavor: HvgFlavor; n_top_genes: number },
): Promise<{ column: string; created: boolean; n_clusters: number }> {
  return apiFetch(`/datasets/${datasetId}/clustering/hvg-sensitivity/keep`, {
    method: "POST",
    body: JSON.stringify({ flavor: setting.flavor, n_top_genes: setting.n_top_genes }),
  });
}
