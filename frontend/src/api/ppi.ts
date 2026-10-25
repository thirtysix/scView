import { apiFetch } from "./client";

/** STRING combined-score cut-offs: medium / high / highest confidence. */
export type PpiScore = 400 | 700 | 900;
export type PpiDirection = "up" | "down" | "both";

/** One protein of the subnetwork (backend core/ppi.py analyse). */
export interface PpiNode {
  /** The gene as the list wrote it. */
  gene: string;
  /** STRING's preferred name. */
  name: string;
  string_id: string;
  /** Partners within the list. */
  degree: number;
  /** Normalised betweenness within the subnetwork (0-1). */
  betweenness: number;
  /** 1 = the largest community; null for a protein with no partner in the list. */
  community: number | null;
  logfc: number | null;
  /** Layout position; null for a protein with no partner (not drawn). */
  x: number | null;
  y: number | null;
  hub: boolean;
}

export interface PpiLink {
  source: number;
  target: number;
  score: number;
}

export interface PpiResult {
  species: string;
  score: PpiScore;
  version: string;
  attribution: string;
  citation: string;
  n_given: number;
  n_used: number;
  max_genes: number;
  truncated: boolean;
  n_mapped: number;
  unmapped: string[];
  n_unmapped: number;
  n_proteins: number;
  n_links: number;
  expected_links: number;
  /** Null with fewer than two proteins. */
  p_value: number | null;
  n_draws: number;
  seed: number;
  n_isolated: number;
  n_communities: number;
  nodes: PpiNode[];
  links: PpiLink[];
  hubs: string[];
  method: string;
  /** Dataset queries only. */
  label?: string;
  source?: string;
  direction?: PpiDirection;
  species_evidence?: string;
}

/** The subnetwork of a list the backend takes from the dataset:
 *  `de:<result id>`, `markers:<column>/<group>` or `program:<k>`. */
export function getDatasetPpi(
  datasetId: string,
  source: string,
  opts: { n?: number; direction?: PpiDirection; score?: PpiScore } = {},
): Promise<PpiResult> {
  const q = new URLSearchParams({
    source,
    n: String(opts.n ?? 100),
    direction: opts.direction ?? "up",
    score: String(opts.score ?? 400),
  });
  return apiFetch<PpiResult>(`/datasets/${datasetId}/ppi?${q.toString()}`);
}

/** The subnetwork of any gene list. */
export function postPpiNetwork(
  species: "human" | "mouse",
  genes: string[],
  score: PpiScore = 400,
  logfc?: Record<string, number>,
): Promise<PpiResult> {
  return apiFetch<PpiResult>("/ppi/network", {
    method: "POST",
    body: JSON.stringify({ species, genes, score, logfc }),
  });
}
