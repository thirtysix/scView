import { API_BASE } from "@/lib/constants";
import { apiFetch } from "./client";
import { runStream, type AtacStreamHandlers } from "./atac";

// Peak-gene links (backend api/v1/atac_links.py, core/atac_links.py).

export type LinksGenes = "variable" | "all" | "list";

export interface LinksBody {
  genes: LinksGenes;
  gene_list?: string[];
  cells_column?: string | null;
  cells_value?: string[] | null;
}

export interface LinksEstimate {
  genes: LinksGenes;
  n_genes: number;
  n_asked: number;
  missing: string[];
  n_cells: number;
  n_dataset_cells: number;
  n_peaks: number;
  estimate_seconds: number;
  estimate_bytes: number;
  estimate_disk_bytes: number;
  genome: { genome: string; source: string; size: number; downloaded: boolean };
}

export interface LinkRow {
  gene: string;
  peak: string;
  distance_to_tss: number;
  score: number;
  z: number;
  p: number;
  padj: number;
}

export interface LinksCounts {
  n_genes_asked: number;
  n_genes_tested: number;
  n_genes_few_cells: number;
  n_genes_few_peaks: number;
  n_pairs_cis: number;
  n_pairs_tested: number;
  n_links: number;
  n_links_padj: number;
  n_peaks_kept: number;
  n_peaks_unknown: number;
  n_cells: number;
}

export interface LinksPage {
  id: string;
  total: number;
  offset: number;
  limit: number;
  sort: string;
  descending: boolean;
  rows: LinkRow[];
  counts: LinksCounts;
  params: { genes: LinksGenes; cells_column?: string | null; cells_value?: string[] | null; seed?: number };
  method: string;
  caption: string;
  seconds: number;
  created_at: string;
  missing: string[];
  stale: boolean;
}

export interface LinksRun {
  id: string;
  label: string;
  created_at: number;
  stale: boolean;
}

export interface LinkArc {
  peak: string;
  start: number;
  end: number;
  middle: number;
  tss: number;
  score: number;
  z: number;
  p: number;
  padj: number;
  distance_to_tss: number;
}

export interface LinksLocus {
  gene: string;
  chrom: string;
  tss: number;
  start: number;
  end: number;
  arcs: LinkArc[];
  peaks: { start: number; end: number }[];
  genes: { name: string; start: number; end: number; strand: string }[];
  note: string | null;
}

export function getLinksEstimate(datasetId: string, body: LinksBody): Promise<LinksEstimate> {
  return apiFetch(`/datasets/${datasetId}/atac/links/estimate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getLatestLinks(datasetId: string): Promise<{ result: LinksPage | null; runs: LinksRun[] }> {
  return apiFetch(`/datasets/${datasetId}/atac/links`);
}

export function getLinksPage(
  datasetId: string,
  id: string,
  opts: { search?: string; sort?: string; descending?: boolean; limit?: number } = {},
): Promise<LinksPage> {
  const q = new URLSearchParams();
  if (opts.search) q.set("search", opts.search);
  if (opts.sort) q.set("sort", opts.sort);
  if (opts.descending) q.set("descending", "true");
  if (opts.limit) q.set("limit", String(opts.limit));
  return apiFetch(`/datasets/${datasetId}/atac/links/${id}?${q.toString()}`);
}

export function getLinksLocus(datasetId: string, id: string, gene: string): Promise<LinksLocus> {
  return apiFetch(`/datasets/${datasetId}/atac/links/${id}/locus?${new URLSearchParams({ gene }).toString()}`);
}

export function linksCsvUrl(datasetId: string, id: string): string {
  return `${API_BASE}/datasets/${datasetId}/atac/links/${id}/csv`;
}

/** The links, streamed; aborting `signal` stops the run before its next gene, nothing saved. */
export function runAtacLinks(datasetId: string, body: LinksBody, h: AtacStreamHandlers = {}): Promise<LinksPage> {
  return runStream(`/datasets/${datasetId}/atac/links`, body, "Peak-gene links", h);
}
