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
