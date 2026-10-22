import { API_BASE } from "@/lib/constants";
import { readError } from "./client";

/** Typed Cell Ranger metrics of one sample (core/seq_metrics.py); fractions are 0..1. */
export interface SeqMetricsValues {
  estimated_cells?: number;
  mean_reads_per_cell?: number;
  median_genes_per_cell?: number;
  median_umi_per_cell?: number;
  total_genes_detected?: number;
  n_reads?: number;
  valid_barcodes?: number;
  valid_umis?: number;
  sequencing_saturation?: number;
  q30_barcode?: number;
  q30_rna_read?: number;
  q30_umi?: number;
  mapped_genome?: number;
  conf_genome?: number;
  conf_exonic?: number;
  conf_intronic?: number;
  conf_intergenic?: number;
  conf_transcriptome?: number;
  conf_probe_set?: number;
  antisense?: number;
  frac_reads_in_cells?: number;
}

export type SeqMetricKey = keyof SeqMetricsValues;

export interface ParsedMetrics {
  /** "count": one row of metric columns; "multi": one metric per row. */
  format: "count" | "multi";
  values: SeqMetricsValues;
  /** Where each value came from: "count", "cells" (the sample) or "library". */
  scope: Partial<Record<SeqMetricKey, "count" | "cells" | "library">>;
  /** Every row of the file as [name, value], in file order. */
  raw: [string, string][];
  notes: string[];
  file?: string;
}

export interface SeqFlag {
  metric: SeqMetricKey;
  label: string;
  /** error / warn: a Cell Ranger web-summary alert; info: a 10x recommendation. */
  level: "error" | "warn" | "info";
  title: string;
  value: number;
  /** The line crossed (warn or error). */
  threshold: number;
  warn: number;
  error: number | null;
  direction: "below" | "above";
  ideal: string;
  message: string;
  source: string;
  url: string;
}

export interface RankCurve {
  rank: number[];
  umi: number[];
  /** 1: the barcode is one of the dataset's cells. */
  kept: number[];
}

export interface KneeSummary {
  n_barcodes: number;
  n_barcodes_in_file?: number;
  total_umis: number;
  n_cells: number;
  n_cells_matched: number;
  match: "barcode" | "sequence";
  knee: number | null;
  inflection: number | null;
  n_above_knee: number | null;
  n_above_inflection: number | null;
  knee_note?: string;
  cells_umi_min?: number;
  cells_umi_median?: number;
  frac_umis_in_cells?: number;
  n_cells_below_knee?: number;
  n_above_knee_not_cells?: number;
  gex_features?: number;
  n_features?: number;
  feature_types?: string;
  method: string;
  url: string;
  curve: RankCurve;
}

export interface SeqSample {
  sample: string;
  /** The obs column holding `sample` (merged datasets), or null for the whole dataset. */
  column: string | null;
  /** "ingest": the uploaded filtered matrix (Cell Ranger's call); "dataset": the cells when attached. */
  cells_source: "ingest" | "dataset";
  attached_at: string;
  files: { metrics: string | null; raw: string[] | null };
  metrics: ParsedMetrics | null;
  knee: KneeSummary | null;
  notes: string[];
  flags: SeqFlag[];
  summary: string;
}

export interface SeqMetricInfo {
  key: SeqMetricKey;
  label: string;
  kind: "int" | "frac";
  tip: string;
}

export interface SeqMetricsResponse {
  samples: SeqSample[];
  n_samples: number;
  metrics_info: SeqMetricInfo[];
}

async function failure(res: Response): Promise<Error> {
  const text = await readError(res);
  let detail = text;
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* not JSON: the text itself */
  }
  return new Error(detail || `Request failed (${res.status}).`);
}

/** GET /seq-metrics: the attached Cell Ranger metrics per sample, or null when none
 *  (asked with allow_empty, so a dataset without them is no 404 in the console). */
export async function getSeqMetrics(datasetId: string): Promise<SeqMetricsResponse | null> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/seq-metrics?allow_empty=true`);
  if (res.status === 404) return null;
  if (!res.ok) throw await failure(res);
  const body = (await res.json()) as SeqMetricsResponse;
  return body.samples.length ? body : null;
}

/** POST /seq-metrics: attach metrics_summary.csv and/or a raw matrix (to one sample of `column`). */
export async function attachSeqMetrics(
  datasetId: string,
  files: File[],
  target: { sample?: string; column?: string } = {},
): Promise<SeqMetricsResponse> {
  const form = new FormData();
  for (const f of files) form.append("files", f);
  if (target.column) form.append("column", target.column);
  if (target.sample) form.append("sample", target.sample);
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/seq-metrics`, { method: "POST", body: form });
  if (!res.ok) throw await failure(res);
  return res.json();
}

/** DELETE /seq-metrics?sample=: remove one sample's record; null when none is left. */
export async function removeSeqMetrics(datasetId: string, sample: string): Promise<SeqMetricsResponse | null> {
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/seq-metrics?sample=${encodeURIComponent(sample)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw await failure(res);
  const body = (await res.json()) as SeqMetricsResponse;
  return body.samples.length ? body : null;
}
