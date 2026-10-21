import { apiFetch } from "./client";
import type { DonorPoint, PairedDesign, PairedTest } from "@/lib/donorPlot";

/** mean: mean log-normalised expression over the donor's cells; cpm: log2(CPM + 1) of their summed counts. */
export type DonorMethod = "mean" | "cpm";

export interface PerDonorExpression {
  gene: string;
  group_key: string;
  /** null: no donor / sample column, see `message`. */
  sample_key: string | null;
  sample_key_detected: boolean;
  condition_key: string | null;
  method: DonorMethod;
  value_label: string;
  /** Where the values came from, e.g. "log-normalised .raw" or "layers['counts']". */
  source: string | null;
  /** Samples with fewer cells of a group are left out. */
  min_cells: number;
  groups: string[];
  /** [""] when not split by a condition. */
  conditions: string[];
  points: DonorPoint[];
  n_donors: Record<string, number>;
  n_dropped: number;
  warnings: string[];
  message: string | null;
  /** Whether the donors recur across the conditions (the same people before and after). */
  design: PairedDesign;
  /** Group -> Wilcoxon signed-rank + paired t over its pairs; only paired with two conditions. */
  paired_tests: Record<string, PairedTest>;
}

/** One gene's pseudobulk value per donor (x condition) in each group (core/donor_expression.py). */
export async function getPerDonorExpression(
  datasetId: string,
  params: { gene: string; groupby: string; sample?: string; condition?: string; method?: DonorMethod },
): Promise<PerDonorExpression> {
  const q = new URLSearchParams({ gene: params.gene, groupby: params.groupby });
  if (params.sample) q.set("sample", params.sample);
  if (params.condition) q.set("condition", params.condition);
  if (params.method) q.set("method", params.method);
  return apiFetch<PerDonorExpression>(`/datasets/${datasetId}/expression/per-donor?${q.toString()}`);
}
