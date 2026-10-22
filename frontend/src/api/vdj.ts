import { API_BASE } from "@/lib/constants";
import { apiFetch, readError } from "./client";

/** The sample choice that matches cells by their barcode suffix ("*-2", "*-1:sampleA"). */
export const SUFFIX_COLUMN = "__barcode_suffix__";

/** One attached contig table and what it matched (backend core/vdj.py attach_file). */
export interface VdjFile {
  name: string;
  sha256: string;
  sample_column: string | null;
  sample: string | null;
  stored_as: string | null;
  attached_at: string;
  format: "10x" | "airr";
  n_rows: number;
  n_contigs: number;
  n_kept_chains: number;
  /** Contigs left out, by the first test each failed. */
  dropped: Record<string, number>;
  kept_by_locus: Record<string, number>;
  n_barcodes: number;
  /** Barcodes with at least one kept chain. */
  n_barcodes_usable: number;
  matched: number;
  matched_exact: number;
  matched_core: number;
  unmatched: number;
  unmatched_usable: number;
  ambiguous: number;
  unmatched_examples: string[];
  n_candidate_cells: number;
  cells_without_contigs: number;
}

export interface SampleChoice {
  name: string;
  label?: string;
  values: string[];
}

export interface VdjMatch {
  cells_profiled: number;
  cells_with_contigs: number;
  cells_with_chain: number;
  cells_without_contigs: number;
  cells_not_profiled: number;
  barcodes_without_cells: number;
  barcodes_without_cells_usable: number;
  ambiguous_barcodes: number;
}

export interface VdjSummary {
  has_vdj: boolean;
  n_cells: number;
  how_to: string;
  sample_choices: SampleChoice[];
  files?: VdjFile[];
  match?: VdjMatch;
  /** Cells per vdj_status, in the column's order. */
  status?: Record<string, number>;
  receptor?: Record<string, number>;
  /** scirpy's chain_pairing labels. */
  pairing?: Record<string, number>;
  /** Cells with a kept chain of each locus. */
  loci?: Record<string, number>;
  n_clonotypes?: number;
  n_cells_in_clonotypes?: number;
  n_expanded?: number;
  n_cells_expanded?: number;
  expansion?: Record<string, { clonotypes: number; cells: number }>;
  same_v_gene?: boolean;
  definition?: string;
  stale?: boolean;
  stale_note?: string | null;
  columns?: string[];
}

export interface ClonotypeChain {
  arm: "VJ" | "VDJ";
  locus: string;
  cdr3: string;
  v: string | null;
  j: string | null;
}

export interface ClonotypeRow {
  id: string;
  n_cells: number;
  /** Cells in the requested group (when one was named). */
  n_in_group: number | null;
  expansion: string;
  receptor: string;
  chains: ClonotypeChain[];
  by_group: Record<string, number>;
  top_group: string | null;
  top_share: number | null;
  indices: number[];
}

export interface ClonotypeTable {
  group_by: string | null;
  group: string | null;
  n_clonotypes: number;
  n_shown: number;
  n_cells_in_clonotypes: number;
  rows: ClonotypeRow[];
}

export type DiversityMetric = "richness" | "shannon" | "simpson" | "clonality" | "gini";

export interface DiversityGroup {
  name: string;
  n_cells: number;
  n_with_clonotype: number;
  richness: number | null;
  shannon: number | null;
  simpson: number | null;
  clonality: number | null;
  gini: number | null;
  note: string | null;
  /** Means (and SDs as <metric>_sd) over the rarefaction draws. */
  rarefied?: Record<string, number | null>;
}

export interface Diversity {
  group_by: string;
  rarefy: boolean;
  depth: number | null;
  seed: number;
  n_draws: number;
  min_cells: number;
  groups: DiversityGroup[];
  caption: string;
}

export interface Overlap {
  group_by: string;
  metric: "morisita_horn" | "jaccard";
  groups: string[];
  n_cells: number[];
  n_clonotypes: number[];
  matrix: (number | null)[][];
  shared: number[][];
  n_groups_total: number;
  n_groups_shown: number;
}

export interface Usage {
  group_by: string;
  chain: string;
  gene: "v" | "j";
  count: "clonotypes" | "cells";
  chains_present: Record<string, number>;
  genes: string[];
  n_genes_total: number;
  groups: string[];
  n: number[];
  counts: number[][];
  share: number[][];
  n_groups_total: number;
  n_groups_shown: number;
}

export function getVdj(datasetId: string): Promise<VdjSummary> {
  return apiFetch(`/datasets/${datasetId}/vdj`);
}

export interface AttachOptions {
  sampleColumn?: string;
  sample?: string;
  replace?: boolean;
}

export interface AttachResult {
  attached: VdjFile;
  summary: VdjSummary;
}

/** POST /vdj/upload (multipart): attach a contig table; errors carry the server's sentence. */
export async function attachVdj(datasetId: string, file: File, opts: AttachOptions = {}): Promise<AttachResult> {
  const form = new FormData();
  form.append("file", file);
  if (opts.sampleColumn) form.append("sample_column", opts.sampleColumn);
  if (opts.sample) form.append("sample", opts.sample);
  if (opts.replace) form.append("replace", "true");
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/vdj/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`API error ${res.status}: ${await readError(res)}`);
  return res.json();
}

export function redefineClonotypes(datasetId: string, sameVGene: boolean): Promise<{ summary: VdjSummary }> {
  return apiFetch(`/datasets/${datasetId}/vdj/clonotypes`, {
    method: "POST",
    body: JSON.stringify({ same_v_gene: sameVGene }),
  });
}

export function removeVdj(datasetId: string): Promise<{ removed_columns: string[]; summary: VdjSummary }> {
  return apiFetch(`/datasets/${datasetId}/vdj`, { method: "DELETE" });
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  const s = q.toString();
  return s ? `?${s}` : "";
}

export function getClonotypes(
  datasetId: string,
  opts: { groupBy?: string; group?: string | null; n?: number } = {},
): Promise<ClonotypeTable> {
  return apiFetch(`/datasets/${datasetId}/vdj/clonotypes${query({ group_by: opts.groupBy, group: opts.group, n: opts.n })}`);
}

export function getDiversity(
  datasetId: string,
  groupBy: string,
  rarefy: boolean,
  opts: { seed?: number; minCells?: number } = {},
): Promise<Diversity> {
  return apiFetch(
    `/datasets/${datasetId}/vdj/diversity${query({ group_by: groupBy, rarefy, seed: opts.seed, min_cells: opts.minCells })}`,
  );
}

export function getOverlap(datasetId: string, groupBy: string, metric: Overlap["metric"]): Promise<Overlap> {
  return apiFetch(`/datasets/${datasetId}/vdj/overlap${query({ group_by: groupBy, metric })}`);
}

export function getUsage(
  datasetId: string,
  groupBy: string,
  opts: { chain?: string; gene?: Usage["gene"]; count?: Usage["count"] } = {},
): Promise<Usage> {
  return apiFetch(
    `/datasets/${datasetId}/vdj/usage${query({ group_by: groupBy, chain: opts.chain, gene: opts.gene, count: opts.count })}`,
  );
}
