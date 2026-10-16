import type { MarkerGene } from "@/api/markers";
import type { CsvCell } from "@/lib/csv";
import { formatFoldChange } from "@/lib/formatting";

/**
 * Marker specificity (backend core/marker_specificity.py): tau over the groups'
 * mean expression, and the group's detection fraction against the highest other
 * group. A strong marker can still be shared with one neighbouring group; these
 * say whether it is.
 */

/** Header tooltips, shared by both marker tables. */
export const SPECIFICITY_HELP = {
  tau:
    "Tau: how much of the gene's mean expression belongs to one group. 0 = even across " +
    "all groups, 1 = expressed in this group only (Yanai et al. 2005).",
  detection:
    "Share of this group's cells that detect the gene, against the highest share in any " +
    "other single group (not the pooled rest).",
  ratio:
    "Detection in this group / detection in the next-best group. Above 1 the group detects " +
    "it more often than any other; ∞ means no other group detects it.",
} as const;

/** The guides on the specificity plot: |log2FC| and tau a marker should clear. */
export const STRONG_LOG2FC = 1;
export const SPECIFIC_TAU = 0.8;

/**
 * Detection ratio from the two fractions: Infinity when only this group detects the
 * gene (the backend sends that as null), null when either fraction is missing.
 */
export function detectionRatio(m: Pick<MarkerGene, "pct_in" | "pct_max_other">): number | null {
  const own = m.pct_in;
  const other = m.pct_max_other;
  if (own == null || other == null || !Number.isFinite(own) || !Number.isFinite(other)) return null;
  if (other > 0) return own / other;
  return own > 0 ? Infinity : null;
}

export function formatRatio(r: number | null): string {
  if (r == null || Number.isNaN(r)) return "–";
  if (r === Infinity) return "∞";
  return r >= 10 ? r.toFixed(0) : r.toFixed(1);
}

/** A fraction as a whole percentage, "<1%" for a few cells (not "0%"), "–" when missing. */
export function formatPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  if (v > 0 && v < 0.005) return "<1%";
  return `${Math.round(v * 100)}%`;
}

export function formatTau(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "–" : v.toFixed(2);
}

/** True when the markers came with specificity columns (older backends send none). */
export function hasSpecificity(markers: readonly MarkerGene[]): boolean {
  return markers.some((m) => m.tau != null || m.pct_max_other != null);
}

export type MarkerSortField =
  | "gene"
  | "logfoldchange"
  | "pval"
  | "pval_adj"
  | "pct_in"
  | "score"
  | "tau"
  | "pct_ratio";

function sortValue(m: MarkerGene, field: Exclude<MarkerSortField, "gene">): number | null {
  const v = field === "pct_ratio" ? detectionRatio(m) : m[field];
  return v == null || Number.isNaN(v) ? null : v;
}

/** A sorted copy for a marker table; a missing value sorts last either way. */
export function sortMarkers(
  markers: readonly MarkerGene[],
  field: MarkerSortField,
  dir: "asc" | "desc",
): MarkerGene[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...markers].sort((a, b) => {
    if (field === "gene") return sign * a.gene.localeCompare(b.gene);
    const av = sortValue(a, field);
    const bv = sortValue(b, field);
    if (av == null || bv == null) return (av == null ? 1 : 0) - (bv == null ? 1 : 0);
    if (av === bv) return 0;
    return sign * (av < bv ? -1 : 1);
  });
}

/** CSV columns for a marker table's export, specificity included. */
export const MARKER_CSV_HEADERS = [
  "gene", "group", "log_fold_change", "p_value", "p_value_adjusted", "pct_in", "pct_out",
  "score", "tau", "pct_max_other", "max_other_group", "pct_ratio",
];

const fixed = (v: number | null | undefined, digits: number) =>
  v == null || !Number.isFinite(v) ? "" : v.toFixed(digits);

export function markerCsvRow(m: MarkerGene): CsvCell[] {
  const ratio = detectionRatio(m);
  return [
    m.gene,
    m.group,
    formatFoldChange(m.logfoldchange, 4, ""),
    m.pval.toExponential(4),
    m.pval_adj.toExponential(4),
    fixed(m.pct_in, 3),
    fixed(m.pct_out, 3),
    fixed(m.score, 4),
    fixed(m.tau, 3),
    fixed(m.pct_max_other, 3),
    m.max_other_group ?? "",
    ratio === Infinity ? "inf" : fixed(ratio, 3),
  ];
}
