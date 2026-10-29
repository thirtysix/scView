import { API_BASE } from "@/lib/constants";

import { apiFetch, readError } from "./client";

// --- GWAS trait links (backend core/gwas_catalog.py, api/v1/gwas.py) ---

export interface GwasStatus {
  release: string;
  downloaded: boolean;
  /** Both files are in DATA_DIR/gwas (placed by hand or fetched) */
  raw_present: boolean;
  auto_download: boolean;
  n_traits: number | null;
  n_genes: number | null;
  n_loci: number | null;
  built_at: string | null;
  download_mb: number;
  assoc_url: string;
  mappings_url: string;
  attribution: string;
  citation: string;
  licence: string;
  caveat: string;
}

export interface GwasTraitRow {
  uri: string;
  trait: string;
  parent: string;
  /** Background genes the catalog maps to the trait */
  n_trait_genes: number;
  n_overlap: number;
  /** Distinct loci (lead SNPs within 1 Mb merged) the overlap genes sit in */
  n_loci: number;
  /** Overlap genes as the catalog names them (human symbols) */
  genes: string[];
  /** The same genes as this dataset names them (mouse symbols on mouse data) */
  dataset_genes: string[];
  odds_ratio: number;
  p: number;
  p_adj: number;
  p_empirical: number;
  n_draws: number;
  /** Adjusted p < 0.05 and empirical p < 0.05 */
  lead: boolean;
}

export interface GwasGroup {
  group: string;
  n_markers: number;
  /** Markers that are in the background */
  n_query: number;
  n_tested: number;
  n_with_loci: number;
  n_leads: number;
  rows: GwasTraitRow[];
}

export interface GwasResult {
  result_id: string;
  column: string;
  include_measurements: boolean;
  n_markers: number;
  species: string;
  mapping: string | null;
  n_background: number;
  n_measured: number;
  min_loci: number;
  trait_size: [number, number];
  n_draws: number;
  refine_draws: number;
  seed: number;
  max_rows: number;
  release: string;
  attribution: string;
  citation: string;
  caveat: string;
  method: string;
  groups: GwasGroup[];
  elapsed_seconds: number;
  created_at: number;
}

export interface GeneTrait {
  uri: string;
  trait: string;
  parent: string;
  mlog: number;
  p: number;
  n_loci: number;
  pmids: number[];
}

export interface GeneTraits {
  gene: string;
  ensembl: string | null;
  query: string;
  via_ortholog: boolean;
  n_traits: number;
  shown: number;
  traits: GeneTrait[];
  release: string;
  attribution: string;
  citation: string;
}

export interface GwasRunBody {
  column: string;
  include_measurements?: boolean;
  n_markers?: number;
  species?: "" | "human" | "mouse";
}

export function getGwasStatus(): Promise<GwasStatus> {
  return apiFetch("/gwas/status");
}

function is404(e: unknown): boolean {
  return e instanceof Error && /API error 404/.test(e.message);
}

/** The last run on the dataset (on `column`), or null when none has run (allow_empty, so no 404). */
export async function getGwasClusters(datasetId: string, column?: string): Promise<GwasResult | null> {
  const q = `?allow_empty=true${column ? `&column=${encodeURIComponent(column)}` : ""}`;
  try {
    return await apiFetch<GwasResult | null>(`/datasets/${datasetId}/gwas/clusters${q}`);
  } catch (e) {
    if (is404(e)) return null;
    throw e;
  }
}

export function runGwasClusters(datasetId: string, body: GwasRunBody): Promise<GwasResult> {
  return apiFetch(`/datasets/${datasetId}/gwas/clusters`, { method: "POST", body: JSON.stringify(body) });
}

/** One gene's traits, or null when the catalog maps nothing to it (404). */
export async function getGeneTraits(symbol: string, species?: string): Promise<GeneTraits | null> {
  const q = species ? `?species=${encodeURIComponent(species)}` : "";
  try {
    return await apiFetch<GeneTraits>(`/gwas/gene/${encodeURIComponent(symbol)}${q}`);
  } catch (e) {
    if (is404(e)) return null;
    throw e;
  }
}

export interface GwasProgress {
  stage: "download" | "check" | "parse";
  file?: string;
  bytes?: number;
  total?: number;
  rows?: number;
}

/** Fetch and prepare the catalog, calling `onProgress` per SSE progress event.
 *  Resolves with the new status; rejects with the server's message. */
export async function downloadGwas(onProgress: (p: GwasProgress) => void): Promise<GwasStatus> {
  const res = await fetch(`${API_BASE}/gwas/download`, { method: "POST" });
  if (!res.ok) throw new Error(await readError(res));
  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response stream");
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
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
        const data = JSON.parse(line.slice(6)) as Record<string, unknown>;
        if (event === "error") throw new Error(String(data.error));
        if (event === "complete") return data as unknown as GwasStatus;
        if (event === "progress") onProgress(data as unknown as GwasProgress);
        event = "";
      }
    }
  }
  throw new Error("The download ended without completing.");
}

// --- Link-outs (no request is made from scView: the browser opens the page) ---

export function gwasCatalogGeneUrl(symbol: string): string {
  return `https://www.ebi.ac.uk/gwas/genes/${encodeURIComponent(symbol)}`;
}

/** The Open Targets target page by Ensembl id, else its search for the symbol. */
export function openTargetsUrl(symbol: string, ensembl?: string | null): string {
  return ensembl
    ? `https://platform.opentargets.org/target/${encodeURIComponent(ensembl)}`
    : `https://platform.opentargets.org/search?q=${encodeURIComponent(symbol)}`;
}

export function gtexGeneUrl(symbol: string): string {
  return `https://gtexportal.org/home/gene/${encodeURIComponent(symbol)}`;
}

export function pubmedUrl(pmid: number): string {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`;
}

export function efoUrl(uri: string): string {
  return uri.startsWith("http") ? uri : `https://www.ebi.ac.uk/ols4/search?q=${encodeURIComponent(uri)}`;
}
