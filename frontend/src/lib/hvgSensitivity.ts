/** Helpers for the variable-gene sensitivity check (components/unified/HvgSensitivity). */

import type { HvgClusterMatch, HvgClusterStatus, HvgFlavor, HvgRow } from "@/api/clustering";

/** Jaccard at or above which a cluster counts as found again (Hennig 2007; the backend's STABLE). */
export const STABLE = 0.75;
/** At or below this a cluster has dissolved (clusterboot's rule). */
export const DISSOLVED = 0.5;

/** robust: stable under every other setting; depends: under some; unstable: under none. */
export type Verdict = "robust" | "depends" | "unstable";

export interface ClusterStability {
  cluster: string;
  n_cells: number;
  /** One per other setting, in the rows' order; null when that row lacks the cluster */
  matches: (HvgClusterMatch | null)[];
  /** Mean best-match Jaccard over the other settings */
  mean: number | null;
  nStable: number;
  verdict: Verdict | null;
}

/** The rows other than the control (the current setting re-run). */
export function otherSettings<T extends { is_current: boolean }>(rows: readonly T[]): T[] {
  return rows.filter((r) => !r.is_current);
}

/**
 * Each current cluster's stability across the other settings, least stable
 * first (clusters in the same state keep the column's order).
 */
export function clusterStability(rows: readonly HvgRow[]): ClusterStability[] {
  const others = otherSettings(rows);
  const byRow = others.map((r) => new Map(r.clusters.map((c) => [c.cluster, c])));
  const order: string[] = [];
  const cells = new Map<string, number>();
  for (const r of rows) {
    for (const c of r.clusters) {
      if (!cells.has(c.cluster)) {
        order.push(c.cluster);
        cells.set(c.cluster, c.n_cells);
      }
    }
  }
  const out = order.map((cluster): ClusterStability => {
    const matches = byRow.map((m) => m.get(cluster) ?? null);
    const found = matches.filter((m): m is HvgClusterMatch => m !== null);
    const nStable = found.filter((m) => m.status === "stable").length;
    const mean = found.length ? found.reduce((s, m) => s + m.jaccard, 0) / found.length : null;
    const verdict: Verdict | null = !found.length
      ? null
      : nStable === found.length
        ? "robust"
        : nStable === 0
          ? "unstable"
          : "depends";
    return { cluster, n_cells: cells.get(cluster) ?? 0, matches, mean, nStable, verdict };
  });
  const rank = new Map(order.map((c, i) => [c, i]));
  return out.sort((a, b) => (a.mean ?? 2) - (b.mean ?? 2) || rank.get(a.cluster)! - rank.get(b.cluster)!);
}

/**
 * Seconds left once `done` settings (their times) of `total` have finished. The
 * first setting also pays for compiling the neighbour search after a server
 * start, so it is left out once there are others, and capped by the estimate
 * while it is the only one.
 */
export function secondsLeft(done: readonly number[], total: number, estimatePer: number | null): number | null {
  const left = total - done.length;
  if (left <= 0) return 0;
  if (done.length === 0) return estimatePer == null ? null : estimatePer * left;
  const sample = done.length >= 2 ? done.slice(1) : done;
  let per = sample.reduce((s, v) => s + v, 0) / sample.length;
  if (done.length === 1 && estimatePer != null) per = Math.min(per, estimatePer);
  return per * left;
}

/** "45 s", "1 min 10 s", "3 min": rounded to 5 s under ten minutes. */
export function formatDuration(seconds: number): string {
  const s = Math.max(5, Math.round(seconds / 5) * 5);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const rest = s % 60;
  if (m >= 10 || rest === 0) return `${Math.round(s / 60)} min`;
  return `${m} min ${rest} s`;
}

/** Per-cell values from per-cluster ones: NaN where a cell's cluster has none. */
export function perCellValues(
  codes: ArrayLike<number>,
  categories: readonly string[],
  values: ReadonlyMap<string, number>,
): Float32Array {
  const byCode = categories.map((c) => values.get(c) ?? NaN);
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) {
    const code = codes[i]!;
    out[i] = code >= 0 && code < byCode.length ? byCode[code]! : NaN;
  }
  return out;
}

/** Background + text classes for a Jaccard value. */
export function jaccardTone(j: number | null): string {
  if (j == null) return "bg-slate-50 text-slate-400";
  if (j >= STABLE) return "bg-emerald-100 text-emerald-800";
  if (j > DISSOLVED) return "bg-amber-100 text-amber-800";
  return "bg-red-100 text-red-700";
}

const STATUS_TEXT: Record<HvgClusterStatus, string> = {
  stable: "found again",
  splits: "splits",
  merges: "merges into a bigger cluster",
  dissolves: "dissolves (splits and merges)",
};

/** A tooltip saying what became of a cluster under one setting. */
export function describeMatch(m: HvgClusterMatch): string {
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const parts = m.parts > 1 ? `, spread over ${m.parts} clusters` : "";
  return (
    `Jaccard ${m.jaccard.toFixed(2)}: ${STATUS_TEXT[m.status]}. ` +
    `${pct(m.contained)} of its cells are in one new cluster, which is ${pct(m.purity)} this cluster${parts}.`
  );
}

const SHORT_FLAVOR: Record<HvgFlavor, string> = {
  seurat: "Seurat",
  cell_ranger: "CR",
  seurat_v3: "v3",
  pearson_residuals: "Pearson",
};

/** A column header for a setting: "Seurat 1k", "Pearson 2k", "CR 1.5k", "Seurat 800". */
export function shortSettingLabel(s: { flavor: HvgFlavor; n_top_genes: number }): string {
  const n = s.n_top_genes;
  const genes = n >= 1000 ? `${Number((n / 1000).toFixed(1))}k` : String(n);
  return `${SHORT_FLAVOR[s.flavor] ?? s.flavor} ${genes}`;
}

/** A cluster's verdict for its row: "robust", "stable in 3 of 4", "unstable". */
export function verdictLabel(s: Pick<ClusterStability, "verdict" | "nStable" | "matches">): string {
  if (s.verdict === "robust") return "robust";
  if (s.verdict === "unstable") return "unstable";
  if (s.verdict === "depends") return `stable in ${s.nStable} of ${s.matches.filter((m) => m).length}`;
  return "—";
}
