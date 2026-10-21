import { apiFetch } from "./client";
import type { PairedDesign, PairedTest } from "@/lib/donorPlot";

export interface CompositionRequest {
  condition_key: string;
  group_key: string;
  /** Donor / sample column; left out, the backend picks the most donor-like one. */
  sample_key?: string;
  /** Values of the condition column to compare (default: all). */
  conditions?: string[];
  /** The baseline condition (default: the first). */
  reference?: string;
  transform?: "logit" | "asin";
  /** Left out: paired when every donor is in every condition. */
  paired?: boolean;
  /** limma's robust empirical Bayes (propeller's default); false: the plain fit. */
  robust?: boolean;
}

export interface CompositionGroup {
  group: string;
  /** Condition -> mean share of the cells over its samples (0-1); null without samples. */
  props: Record<string, number | null>;
  n_cells: Record<string, number>;
  /** Mean share in the other condition / the reference (two conditions only). */
  fold_change: number | null;
  /** Moderated t (two conditions) or F (more). */
  stat: number | null;
  pval: number | null;
  fdr: number | null;
  /** This group's prior df; smaller for an outlying variance (robust). null: infinite. */
  df_prior: number | null;
  /** Wilcoxon signed-rank + paired t on the raw shares (%) of the donors in both conditions; null unless paired with two. */
  paired_test: PairedTest | null;
}

export interface CompositionSample {
  sample: string;
  condition: string;
  n_cells: number;
  /** Share of each group, in `groups` order. */
  props: number[];
}

export interface CompositionResult {
  condition_key: string;
  sample_key: string | null;
  sample_key_detected: boolean;
  group_key: string;
  /** Reference first. */
  conditions: string[];
  reference: string;
  transform: "logit" | "asin";
  robust: boolean;
  /** Robust: groups whose variance is an outlier, given a smaller prior df. */
  n_outliers: number;
  tested: boolean;
  design: "paired" | "unpaired" | null;
  test: "moderated t" | "moderated F" | null;
  df_residual: number | null;
  /** Empirical-Bayes prior df (robust: the non-outliers'); null when tested means infinite. */
  df_prior: number | null;
  n_samples: Record<string, number>;
  n_cells: Record<string, number>;
  n_significant: number;
  fdr_threshold: number;
  /** How many groups a chi-square on the pooled cells would call significant. */
  n_significant_cell_level: number | null;
  groups: CompositionGroup[];
  samples: CompositionSample[];
  warnings: string[];
  notes: string[];
  dropped_samples: string[];
  /** Whether any donor occurs in more than one condition, and how many are in all. */
  pairing: PairedDesign;
}

/**
 * Replicate-aware composition test (propeller-style): per-sample shares of each
 * group, logit (or arcsin-sqrt), a moderated t / F test with the samples as
 * replicates, Benjamini-Hochberg across groups.
 */
export async function testComposition(
  datasetId: string,
  body: CompositionRequest,
): Promise<CompositionResult> {
  return apiFetch<CompositionResult>(`/datasets/${datasetId}/composition/test`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}
