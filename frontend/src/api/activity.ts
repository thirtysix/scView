import { apiFetch } from "./client";

/** CollecTRI (TFs) or PROGENy (pathways): backend core/activity_nets.py. */
export type ActivityNetwork = "collectri" | "progeny";

/** One TF or pathway of a network: what an activity colouring shows. */
export interface ActivityRef {
  network: ActivityNetwork;
  source: string;
}

/** Sources scored on a dataset, and each group's mean activity (ULM t-value). */
export interface ActivitySummary {
  network: ActivityNetwork;
  label: string;
  kind: "tf" | "pathway";
  citation: string;
  species: string;
  /** Which matrix was scored, e.g. "log-normalised .raw". */
  matrix_source: string;
  n_cells: number;
  n_genes_measured: number;
  sources: string[];
  n_targets: number[];
  groupby: string | null;
  /** With a groupby: groups in the column's order, and groups x sources means. */
  groups?: string[];
  group_sizes?: number[];
  mean?: number[][];
  elapsed_seconds: number;
}

export interface ActivityCells {
  network: ActivityNetwork;
  source: string;
  n_targets: number;
  scores: number[];
}

export interface DEActivityResult {
  source: string;
  /** ULM t-value on the DE statistic: > 0 = the source's targets move up in A. */
  score: number;
  pvalue: number;
  padj: number;
  n_targets: number;
}

export interface DEActivityResponse {
  network: ActivityNetwork;
  label: string;
  kind: "tf" | "pathway";
  citation: string;
  n_genes: number;
  results: DEActivityResult[];
}

export const NETWORK_LABELS: Record<ActivityNetwork, { short: string; long: string }> = {
  collectri: { short: "TFs", long: "TFs (CollecTRI)" },
  progeny: { short: "Pathways", long: "Pathways (PROGENy)" },
};

export async function getActivitySummary(
  datasetId: string,
  network: ActivityNetwork,
  groupby?: string,
): Promise<ActivitySummary> {
  const q = groupby ? `?groupby=${encodeURIComponent(groupby)}` : "";
  return apiFetch<ActivitySummary>(`/datasets/${datasetId}/activity/${network}${q}`);
}

export async function getActivityCells(
  datasetId: string,
  network: ActivityNetwork,
  source: string,
): Promise<ActivityCells> {
  return apiFetch<ActivityCells>(
    `/datasets/${datasetId}/activity/${network}/cells?source=${encodeURIComponent(source)}`,
  );
}

export async function getDEActivity(
  datasetId: string,
  network: ActivityNetwork,
  genes: string[],
  stats: number[],
  resultId?: string | null,
): Promise<DEActivityResponse> {
  // By the id the DE response carried when there is one: a whole transcriptome
  // sent back can pass the backend's 512 KB request cap.
  return apiFetch<DEActivityResponse>(`/datasets/${datasetId}/activity/${network}/de`, {
    method: "POST",
    body: JSON.stringify(resultId ? { result_id: resultId } : { genes, stats }),
  });
}
