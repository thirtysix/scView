import { API_BASE } from "@/lib/constants";
import { apiFetch } from "./client";

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
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/clustering/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: handlers.signal,
  });
  if (!res.ok) {
    let detail = await res.text();
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
        if (event === "start") handlers.onStart?.(data as SweepStart);
        else if (event === "result") handlers.onResult(data as SweepRow);
        else if (event === "complete") complete = true;
        event = "";
      }
    }
  }
  if (!complete) throw new Error("The sweep stopped before it finished.");
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
