/**
 * Regress-out before PCA: the scaling step's `regress_out` (backend core/scaling.py), obs
 * columns whose least-squares fit is taken off the variable genes before their z-scores
 * (`sc.pp.regress_out`). X keeps the log values. Off (empty) by default.
 */

import type { ObsColumnInfo } from "@/api/types";

/** Covariates offered whether or not the dataset has them yet: QC metrics come from the
 *  run's QC step, and the scaling step scores the cell cycle itself. */
export const REGRESS_COVARIATES: { key: string; label: string }[] = [
  { key: "total_counts", label: "Total counts" },
  { key: "pct_counts_mt", label: "% mito" },
  { key: "S_score", label: "S score" },
  { key: "G2M_score", label: "G2M score" },
  { key: "cc_difference", label: "S − G2M (cell-cycle difference)" },
];

const LABELS: Record<string, string> = Object.fromEntries(REGRESS_COVARIATES.map((c) => [c.key, c.label]));
const NUMERIC = /^(u?int|float)\d*$/i;
// Columns the app writes that are numbers but no covariate (cell-cycle phase is a label).
const SKIP = new Set(["doublet_score", "predicted_doublet", "phase"]);

export const REGRESS_HELP =
  "Takes each variable gene's straight-line fit on these per-cell numbers off before PCA " +
  "(scanpy regress_out); X and plots keep the log values. It helps when a technical score " +
  "drives a leading PC inside clusters. It can hurt: regressing the cell cycle removes real " +
  "biology in proliferating tissue (S − G2M keeps dividing cells apart from resting ones), " +
  "and regressing total counts is usually redundant after normalisation and can over-correct " +
  "(scanpy issue #526).";

/** A covariate key in words. */
export function regressLabel(key: string): string {
  return LABELS[key] ?? key;
}

/** The choices: the standard covariates, then the dataset's other numeric columns. */
export function regressOptions(columns: ObsColumnInfo[] | undefined): { key: string; label: string }[] {
  const extra = (columns ?? [])
    .filter((c) => NUMERIC.test(c.dtype) && !(c.name in LABELS) && !SKIP.has(c.name) && c.n_unique > 1)
    .map((c) => ({ key: c.name, label: c.name }));
  return [...REGRESS_COVARIATES, ...extra];
}

/** A recorded `regress_out` in words, for the History chip. */
export function regressOutText(value: unknown): string {
  const keys = Array.isArray(value) ? value.map(String) : [];
  return keys.length ? `regressed out: ${keys.map(regressLabel).join(", ")}` : "regress-out: off";
}
