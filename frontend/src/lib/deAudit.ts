/**
 * Display helpers for the DE audit (api/deAudit.ts, backend core/de_audit.py):
 * when it can run, the shuffles' histogram, and the plain-text lines.
 */
import type { DEResponse } from "@/api/de";
import type { DEAuditProgress, DEAuditPseudobulk, DEAuditResult, DEAuditStart } from "@/api/deAudit";

/** Shuffles need this many donors on each side (backend MIN_DONORS). */
export const AUDIT_MIN_DONORS = 3;

/**
 * Why the audit cannot run on this result, or null when it can.
 * `donorColumns`: obs columns that could hold donors (lib/de.ts replicateColumns).
 */
export function auditBlocker(
  result: DEResponse | null,
  betweenGroups: boolean,
  donorColumns: number,
): string | null {
  if (!result) return "Run a DE first.";
  if (!betweenGroups)
    return "Needs a between-groups result (Between groups above): the audit shuffles condition labels across donors.";
  if (!result.result_id) return "This result was not kept by the server; run the DE again.";
  if (result.method === "logreg")
    return "Logistic regression gives no p-values, and the audit counts genes at adjusted p < 0.05.";
  if (!result.replicate_key && donorColumns === 0)
    return "Needs a donor or sample column, and this dataset has none.";
  return null;
}

/** A note when the shuffles will be skipped (fewer than 3 donors a side). */
export function fewDonorsNote(result: DEResponse | null): string | null {
  const a = result?.n_donors_a ?? result?.n_samples_a;
  const b = result?.n_donors_b ?? result?.n_samples_b;
  if (a == null || b == null || Math.min(a, b) >= AUDIT_MIN_DONORS) return null;
  return `Only ${a} and ${b} donors: the audit will compare with pseudobulk but skip the donor shuffles (they need ${AUDIT_MIN_DONORS} a side).`;
}

export interface NullBins {
  /** Bin centres and counts of shuffles. */
  x: number[];
  y: number[];
  width: number;
  /** x-axis range that holds the bins and the observed line. */
  range: [number, number];
}

/** The shuffles' gene counts in up to `nBins` equal bins from 0 to past the observed count. */
export function nullBins(counts: number[], observed: number, nBins = 12): NullBins {
  const top = Math.max(observed, ...counts, 1);
  const width = Math.max(1, Math.ceil((top + 1) / nBins));
  const n = Math.ceil((top + 1) / width);
  const y = new Array<number>(n).fill(0);
  for (const c of counts) {
    const k = Math.min(n - 1, Math.floor(c / width));
    y[k] = (y[k] ?? 0) + 1;
  }
  const x = y.map((_, i) => i * width + width / 2);
  return { x, y, width, range: [0, n * width] };
}

/** "12" or "0-12 (median 3)" of the shuffles' counts. */
export function nullRange(counts: number[]): string {
  if (counts.length === 0) return "none";
  const sorted = [...counts].sort((p, q) => p - q);
  const at = (i: number) => sorted[i] ?? 0;
  const n = sorted.length;
  const lo = at(0).toLocaleString("en-US");
  const hi = at(n - 1).toLocaleString("en-US");
  const mid = n % 2 ? at((n - 1) / 2) : (at(n / 2 - 1) + at(n / 2)) / 2;
  return lo === hi ? lo : `${lo}-${hi} (median ${Math.round(mid).toLocaleString("en-US")})`;
}

const SCHEMES: Record<DEAuditStart["scheme"], string> = {
  paired: "half the donors' two samples swapped",
  donors: "donors moved between the sides",
  samples: "donor samples moved between the sides",
};

export function schemeLabel(scheme: DEAuditStart["scheme"]): string {
  return SCHEMES[scheme] ?? scheme;
}

/** The running line: what step, and how far through the shuffles. */
export function progressText(start: DEAuditStart | null, p: DEAuditProgress | null): string {
  if (!p) return start ? `Preparing ${start.label}...` : "Starting...";
  return p.message;
}

/** Concordance of cell level and pseudobulk in one sentence. */
export function concordanceText(pb: DEAuditPseudobulk, alpha: number): string {
  const f = (v: number) => v.toLocaleString("en-US");
  const jac = pb.jaccard == null ? "no overlap to measure" : `Jaccard ${pb.jaccard.toFixed(2)}`;
  const rho = pb.spearman_lfc == null ? "" : `; Spearman ${pb.spearman_lfc.toFixed(2)} of the log2 fold changes over ${f(pb.n_genes_common)} genes in both`;
  return `Adj. p < ${alpha}: ${f(pb.n_sig_cell)} at cell level, ${f(pb.n_sig_pseudobulk)} in pseudobulk (${pb.name}, ${pb.n_samples_a} vs ${pb.n_samples_b} donor samples), ${f(pb.n_shared)} in both (${jac})${rho}.`;
}

/** "62% (SNG-1015)", or "–". */
export function shareText(share: number | null, donor: string | null): string {
  if (share == null) return "–";
  return `${Math.round(share * 100)}%${donor ? ` (${donor})` : ""}`;
}

/** Plot captions' numbers. */
export function auditSummary(r: DEAuditResult): { shuffles: string; empirical: string | null } {
  return {
    shuffles: nullRange(r.null_counts),
    empirical: r.empirical_p == null ? null : `empirical p ${r.empirical_p.toPrecision(2)}`,
  };
}
