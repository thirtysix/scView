import { readError } from "@/api/client";
import { API_BASE } from "@/lib/constants";

/** Marker off-target expression (backend core/offtarget.py): where else in the body
 *  a group's markers are expressed, from the Human Protein Atlas. */

export type HpaCategory = "enriched" | "group_enriched" | "enhanced" | "low_specificity" | "not_detected";
export type OffTargetVerdict = "restricted" | "enhanced" | "broad" | "not_detected" | "not_in_hpa";

export interface HpaSide {
  category: HpaCategory;
  category_text: string;
  /** Tau over log2(value + 1): 0 even everywhere, 1 in one tissue only */
  tau: number | null;
  max: number;
  top: { name: string; value: number }[];
  /** The tissues / cell types that make it enriched, group enriched or enhanced */
  elevated: string[];
}

export interface OffTargetGene {
  gene: string;
  in_dataset_tau: number | null;
  found: boolean;
  hpa_symbol: string | null;
  ensg: string | null;
  url: string | null;
  tissue: HpaSide | null;
  cell_type: HpaSide | null;
  verdict: OffTargetVerdict;
  verdict_text: string;
  /** Broad in HPA but specific in the dataset */
  flag: boolean;
}

export interface OffTargetResult {
  species: string;
  note: string | null;
  genes: OffTargetGene[];
  heatmap: { genes: string[]; tissues: string[]; log2_ntpm: number[][]; unit: string };
  summary: {
    n_genes: number;
    counts: Record<OffTargetVerdict, number>;
    n_flagged: number;
    flagged: string[];
  };
  attribution: string;
  version: string;
  licence: string;
  cutoffs?: { fold: number; min_level: number; group_tissues: number; group_cell_types: number; tau_specific: number };
  group_col?: string;
  group?: string;
  n?: number;
  marker_rule?: string;
}

async function fail(res: Response): Promise<never> {
  const text = await readError(res);
  let detail = text;
  try {
    const body = JSON.parse(text);
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // not JSON: the text itself
  }
  throw new Error(detail || `HPA lookup failed (${res.status}).`);
}

/** A group's markers (the app's one marker choice) looked up in HPA. Downloads the
 *  HPA tables on first use (~21 MB). */
export async function getOffTarget(
  datasetId: string,
  groupCol: string,
  group: string,
  n = 20,
): Promise<OffTargetResult> {
  const q = new URLSearchParams({ group_col: groupCol, group, n: String(n) });
  const res = await fetch(`${API_BASE}/datasets/${datasetId}/markers/offtarget?${q}`);
  if (!res.ok) return fail(res);
  return res.json();
}

/** An explicit gene list (e.g. a saved gene list) looked up in HPA. */
export async function offTargetForGenes(genes: string[], species: "human" | "mouse" = "human"): Promise<OffTargetResult> {
  const res = await fetch(`${API_BASE}/offtarget/genes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ genes, species }),
  });
  if (!res.ok) return fail(res);
  return res.json();
}
