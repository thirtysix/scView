import { streamEvents } from "./clustering";
import { apiFetch } from "./client";

// --- Compare batch-correction methods on the current PCA (backend core/batch_compare.py,
// api/v1/batch_compare.py) ---

export type BatchMethod = "harmony" | "scanorama" | "bbknn";
export type CompareMethod = "none" | BatchMethod;

export const BATCH_METHOD_LABELS: Record<CompareMethod, string> = {
  none: "None",
  harmony: "Harmony",
  scanorama: "Scanorama",
  bbknn: "BBKNN",
};

export interface CompareInfo {
  available: boolean;
  reason: string | null;
  methods: CompareMethod[];
  background: string;
  background_tip: string;
  n_cells: number;
  running: boolean;
  /** The label conservation is scored with (null: none found) */
  label_column?: string | null;
  label_kind?: "shared" | "within_dataset" | null;
  estimate_seconds?: number;
  /** The estimate counts BBKNN's one-off compilation (no comparison has run in this backend yet) */
  first_run?: boolean;
}

export interface CompareRow {
  method: CompareMethod;
  label: string;
  seconds: number;
  error?: string;
  /** Share of a cell's neighbours from another batch */
  mixing?: number;
  /** Effective number of batches around a cell */
  ilisi?: number;
  /** (iLISI - 1) / (ideal - 1), 0..1: what "mixes acceptably" is judged on */
  mixing_ratio?: number | null;
  /** Share of a cell's neighbours with its label; null without a label column */
  conservation?: number | null;
  conservation_change?: number | null;
  overcorrection?: boolean;
}

export interface CompareStart {
  methods: CompareMethod[];
  n_cells: number;
  n_scored: number;
  k: number;
  batch_key: string;
  n_batches: number;
  label_column: string | null;
  label_kind: "shared" | "within_dataset" | null;
  ideal: number;
  ilisi_ideal: number;
}

export interface CompareResult {
  rows: CompareRow[];
  recommended: CompareMethod | null;
  reason: string;
  warnings: string[];
  notes: string[];
  label_column: string | null;
  label_kind: "shared" | "within_dataset" | null;
  k: number;
  ideal: number;
  ilisi_ideal: number;
  background: string;
  background_tip: string;
}

export function getCompareInfo(datasetId: string, batchKey: string, labelColumn?: string | null) {
  const q = new URLSearchParams({ batch_key: batchKey });
  if (labelColumn) q.set("label_column", labelColumn);
  return apiFetch<CompareInfo>(`/datasets/${datasetId}/batch/compare?${q}`);
}

export interface CompareHandlers {
  onStart?: (s: CompareStart) => void;
  onMethodStart?: (method: CompareMethod, index: number, total: number) => void;
  onMethod?: (row: CompareRow) => void;
  onResult?: (r: CompareResult) => void;
}

/** Run the comparison; resolves when it completes (rejects on error or a stopped stream). */
export async function runCompare(
  datasetId: string,
  body: { batch_key: string; label_column?: string | null; methods?: CompareMethod[] },
  handlers: CompareHandlers,
  signal?: AbortSignal,
): Promise<void> {
  await streamEvents(
    `/datasets/${datasetId}/batch/compare-stream`,
    body,
    (event, data) => {
      const d = data as Record<string, unknown>;
      if (event === "start") handlers.onStart?.(d as unknown as CompareStart);
      else if (event === "method_start")
        handlers.onMethodStart?.(d.method as CompareMethod, Number(d.index), Number(d.total));
      else if (event === "method") handlers.onMethod?.(d as unknown as CompareRow);
      else if (event === "result") handlers.onResult?.(d as unknown as CompareResult);
    },
    "The comparison",
    signal,
  );
}
