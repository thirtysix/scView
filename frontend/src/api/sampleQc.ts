import { apiFetch } from "./client";

/** A sample's sex from its expression (backend core/sex_check.py). */
export type SexCall = "female" | "male" | "mixed" | "ambiguous" | "undetermined";

export interface SexGroup {
  name: string;
  n_cells: number;
  /** Share of cells with any female-marker (XIST/TSIX) count; null: no such gene in the data. */
  frac_female: number | null;
  /** Share of cells with any Y-gene count; null: no Y gene in the data. */
  frac_male: number | null;
  frac_both: number;
  /** Pseudobulk counts per million of the sample's counts; null without raw counts. */
  cpm_female: number | null;
  cpm_male: number | null;
  call: SexCall;
  confidence: "high" | "low" | "none";
  /** One sentence with the numbers behind the call. */
  evidence: string;
  /** The sample's value in the sex metadata column ("mixed" when it varies), if there is one. */
  metadata: "female" | "male" | "mixed" | null;
  /** The metadata says the other sex: a possible sample swap or mislabel. */
  mismatch: boolean;
}

export interface SexConfounding {
  condition: string;
  kind: "confounded" | "imbalanced";
  levels: Record<string, { female: number; male: number }>;
  message: string;
}

export interface SexCheck {
  /** The sample column; null: no sample column, all cells are one group. */
  column: string | null;
  reason: string | null;
  n_cells: number;
  species: "human" | "mouse" | "unknown";
  species_evidence: string;
  female_genes: string[];
  male_genes: string[];
  female_genes_missing: string[];
  male_genes_missing: string[];
  source: string;
  counts: boolean;
  thresholds: { on: number; off: number; min_cells: number };
  metadata_column: string | null;
  groups: SexGroup[];
  summary: Record<SexCall, number>;
  confounding: SexConfounding[];
  warnings: string[];
  notes: string[];
}

/** Per-sample sex from XIST vs Y-chromosome genes; ``column`` names the samples (default: detected). */
export async function getSexCheck(datasetId: string, column?: string | null): Promise<SexCheck> {
  const q = column ? `?column=${encodeURIComponent(column)}` : "";
  return apiFetch<SexCheck>(`/datasets/${datasetId}/qc/sex${q}`);
}

/** One pseudobulk profile: a sample, or a sample x condition, optionally within one cell type. */
export interface PseudobulkProfile {
  id: string;
  sample: string;
  split: string | null;
  n_cells: number;
  total_counts: number;
  /** Scores on PC1..PCk. */
  pcs: number[];
  /** Leave-one-out distance on the first PCs (absent with too few profiles). */
  distance?: number;
  outlier?: boolean;
}

export interface PseudobulkCovariate {
  name: string;
  kind: "categorical" | "numeric";
  /** Computed by scView (cells, library size, inferred sex), not an obs column. */
  derived: boolean;
  /** Other obs columns that split the profiles the same way (Kang's orig_ident = stim). */
  same_as: string[];
  /** One value per profile, in `profiles` order. */
  values: (string | number)[];
}

export interface PcAssociation {
  pc: number;
  r2: number | null;
  p: number | null;
  /** Kruskal-Wallis p for a categorical covariate. */
  p_rank: number | null;
  rho?: number;
}

export interface PcAssociationRow {
  covariate: string;
  kind: string;
  same_as: string[];
  test: string;
  pcs: PcAssociation[];
}

export interface PseudobulkQc {
  sample_column: string | null;
  reason: string | null;
  split_by: string | null;
  celltype_column: string | null;
  celltype: string | null;
  celltype_levels: { name: string; n_cells: number }[];
  n_cells: number;
  n_cells_used: number;
  min_cells: number;
  profiles: PseudobulkProfile[];
  dropped: { sample: string; split: string | null; n_cells: number }[];
  n_genes: number;
  n_top: number;
  explained_variance: number[];
  covariates: PseudobulkCovariate[];
  association: PcAssociationRow[];
  drivers: Record<string, { covariate: string; r2: number; p: number } | null>;
  loadings: Record<string, { positive: string[]; negative: string[] }>;
  outliers: { id: string; distance: number; cutoff: number; pc: number; few_cells: boolean; reason: string }[];
  outlier_rule: string | null;
  counts_source: string | null;
  notes: string[];
  /** Why there is no PCA (no samples, no counts, too few profiles). */
  message: string | null;
}

export interface PseudobulkQcParams {
  /** undefined: detected; "": none. */
  splitBy?: string;
  celltypeColumn?: string | null;
  celltype?: string | null;
}

/** PCA of the samples' pseudobulk profiles, PC-covariate association and outliers. */
export async function getPseudobulkQc(datasetId: string, p: PseudobulkQcParams = {}): Promise<PseudobulkQc> {
  const q = new URLSearchParams();
  if (p.splitBy !== undefined) q.set("split_by", p.splitBy);
  if (p.celltypeColumn) q.set("celltype_column", p.celltypeColumn);
  if (p.celltype) q.set("celltype", p.celltype);
  const qs = q.toString();
  return apiFetch<PseudobulkQc>(`/datasets/${datasetId}/qc/pseudobulk${qs ? `?${qs}` : ""}`);
}
