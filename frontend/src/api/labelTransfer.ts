import { apiFetch } from "./client";
import { streamEvents } from "./clustering";

/** Reference mapping / label transfer (backend core/label_transfer.py). */
export type LabelTransferMethod = "projection" | "integration" | "celltypist";

export interface LabelTransferCluster {
  cluster: string;
  n_cells: number;
  unmapped_share: number;
  top_label: string;
  top_label_share: number;
  /** Most of its cells are "not in reference" */
  candidate_absent: boolean;
}

export interface LabelTransferSummary {
  prefix: string;
  columns: { label: string; score: string; unmapped: string; embedding?: string };
  method: LabelTransferMethod;
  method_text: string;
  k: number;
  n_cells: number;
  n_shared_genes: number;
  n_hvg: number;
  threshold: number;
  /** Query median novelty ratio over the reference's (>= 1); the cut is scaled by it. Absent on old results. */
  shift?: number;
  percentile: number;
  n_unmapped: number;
  unmapped_share: number;
  median_score: number;
  labels: { label: string; n_cells: number; mean_score: number; unmapped_share: number }[];
  reference_labels: { label: string; n_cells: number; n_query: number }[];
  score_hist: { counts: number[]; edges: number[] };
  cluster_column: string | null;
  clusters: LabelTransferCluster[];
  reference: { id: string; name: string; label_column: string; n_reference_cells: number };
  sources: { reference: string; query: string; renamed: string | null };
  seconds: number | null;
  note: string;
}

export interface LabelTransferInfo {
  methods: LabelTransferMethod[];
  method: LabelTransferMethod;
  k: number;
  min_k: number;
  max_k: number;
  min_shared_genes: number;
  note: string;
  saved: LabelTransferSummary[];
  available: boolean;
  reason: string | null;
  n_cells: number;
  running: boolean;
}

export interface ReferenceCheck {
  ok: boolean;
  reason: string | null;
  n_cells?: number;
  has_umap?: boolean;
  label_columns?: { column: string; n_labels: number }[];
  species?: string;
  dataset_species?: string;
  shared_genes_by_name?: number;
  min_shared_genes?: number;
  estimate?: { peak_bytes: number; seconds: number; seconds_range: [number, number] };
}

export function getLabelTransfer(datasetId: string): Promise<LabelTransferInfo> {
  return apiFetch(`/datasets/${datasetId}/label-transfer`);
}

export function checkReference(datasetId: string, referenceId: string, method: LabelTransferMethod): Promise<ReferenceCheck> {
  return apiFetch(`/datasets/${datasetId}/label-transfer/reference/${referenceId}?method=${method}`);
}

export interface LabelTransferRunBody {
  reference_id: string;
  label_column: string;
  method: LabelTransferMethod;
  k: number;
}

/** Map the dataset onto the reference (SSE); resolves with the saved summary. */
export async function runLabelTransfer(
  datasetId: string,
  body: LabelTransferRunBody,
  handlers: { onProgress?: (message: string) => void; signal?: AbortSignal } = {},
): Promise<LabelTransferSummary> {
  let result: LabelTransferSummary | null = null;
  await streamEvents(
    `/datasets/${datasetId}/label-transfer/run-stream`,
    body,
    (event, data) => {
      const d = data as { message?: string };
      if (event === "progress" && d.message) handlers.onProgress?.(d.message);
      else if (event === "result") result = data as LabelTransferSummary;
    },
    "The reference mapping",
    handlers.signal,
  );
  if (!result) throw new Error("The reference mapping ended without a result; nothing was saved.");
  return result;
}
