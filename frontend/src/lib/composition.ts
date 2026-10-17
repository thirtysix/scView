/** Helpers for the replicate-aware composition test (Compare > Composition). */
import type { CompositionGroup, CompositionResult } from "@/api/composition";
import type { ObsColumnInfo } from "@/api/types";
import type { CsvCell } from "@/lib/csv";
import { isCategoricalDtype } from "@/lib/obsColumns";

/** The backend compares at most this many conditions at once. */
export const MAX_CONDITIONS = 8;
const MAX_CONDITION_VALUES = 20;

// As the backend's core/confounding.py CONDITION_RE, plus a joint dataset's
// 'dataset' column (each source dataset often is one condition).
const CONDITION_RE =
  /condition|stim|treat|disease|status|genotype|timepoint|time_point|infect|perturb|group/i;

/** Columns that can hold the conditions: categorical, 2-20 values. */
export function conditionColumns(cols: ObsColumnInfo[] | undefined, exclude: string[] = []): ObsColumnInfo[] {
  return (cols ?? []).filter(
    (c) =>
      !exclude.includes(c.name) &&
      isCategoricalDtype(c.dtype) &&
      c.n_unique >= 2 &&
      c.n_unique <= MAX_CONDITION_VALUES,
  );
}

/** The most condition-like column by name (2-8 values first), else "dataset", else "". */
export function suggestConditionColumn(cols: ObsColumnInfo[]): string {
  const named = cols.filter((c) => CONDITION_RE.test(c.name));
  const hit =
    named.find((c) => c.n_unique <= MAX_CONDITIONS) ?? named[0] ?? cols.find((c) => c.name === "dataset");
  return hit?.name ?? "";
}

/** Rows best first: by p when tested, else by overall share. */
export function sortedGroups(r: CompositionResult): CompositionGroup[] {
  const share = (g: CompositionGroup) =>
    r.conditions.reduce((a, k) => a + (g.props[k] ?? 0), 0);
  return [...r.groups].sort((a, b) =>
    r.tested
      ? (a.pval ?? Infinity) - (b.pval ?? Infinity) || share(b) - share(a)
      : share(b) - share(a),
  );
}

/** Is this group's shift called at the result's FDR threshold? */
export function isShifted(g: CompositionGroup, r: CompositionResult): boolean {
  return r.tested && g.fdr != null && g.fdr < r.fdr_threshold;
}

/** One line on what was tested and how, for under the table and the CSV name. */
export function describeTest(r: CompositionResult): string {
  if (!r.tested) return "Proportions only: not tested.";
  const reps = r.conditions.map((k) => `${k} ${r.n_samples[k] ?? 0}`).join(", ");
  const scale = r.transform === "logit" ? "logit" : "arcsin-square-root";
  const prior =
    r.df_prior == null ? "variances fully pooled across groups" : `prior df ${r.df_prior.toFixed(1)}`;
  const design =
    r.design === "paired" ? `paired by ${r.sample_key}` : `samples from ${r.sample_key} treated as independent`;
  return (
    `${r.test === "moderated t" ? "Moderated t-test" : "Moderated F-test"} on ${scale}-transformed ` +
    `per-sample shares, ${design} (${reps} samples), empirical-Bayes variances (${prior}), ` +
    `Benjamini-Hochberg across ${r.groups.length} groups.`
  );
}

/** The table as CSV: one row per group, shares as percentages. */
export function compositionCsv(r: CompositionResult): { headers: string[]; rows: CsvCell[][] } {
  const two = r.conditions.length === 2;
  const headers = [
    r.group_key,
    ...r.conditions.map((k) => `pct_${k}`),
    ...(two ? ["fold_change", "log2_fold_change"] : []),
    r.test === "moderated F" ? "F" : "t",
    "pval",
    "fdr",
    ...r.conditions.map((k) => `n_cells_${k}`),
    ...r.conditions.map((k) => `n_samples_${k}`),
  ];
  const num = (v: number | null | undefined, digits: number): CsvCell =>
    v == null || !Number.isFinite(v) ? "" : Number(v.toPrecision(digits));
  const rows = sortedGroups(r).map((g) => [
    g.group,
    ...r.conditions.map((k) => num(g.props[k] == null ? null : g.props[k]! * 100, 6)),
    ...(two ? [num(g.fold_change, 6), num(g.fold_change ? Math.log2(g.fold_change) : null, 6)] : []),
    num(g.stat, 6),
    num(g.pval, 6),
    num(g.fdr, 6),
    ...r.conditions.map((k) => g.n_cells[k] ?? 0),
    ...r.conditions.map((k) => r.n_samples[k] ?? 0),
  ]);
  return { headers, rows };
}
