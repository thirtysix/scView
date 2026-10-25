/**
 * The Communication tab's "Compare conditions" data shaping: the compared
 * rows, their filters and sort, the sender x receiver difference matrix and
 * its colour range (diverging about 0, from lib/colorScale.ts). Pure functions,
 * tested without a browser.
 */
import type { CompareCall, CompareNum, CompareResult, CompareText } from "@/api/communication";
import type { DivergingColormapName } from "@/lib/colors";
import { overlayStats, resolveColorScale, DEFAULT_CLIP } from "@/lib/colorScale";
import type { Weight } from "@/lib/communication";

export type CompareRow = Record<CompareText, string> &
  Record<CompareNum, number | null> & {
    /** Position in the server's table (the donors endpoint takes it) */
    index: number;
    n_donors: number;
    composition_note: boolean;
  };

const TEXT: CompareText[] = ["source", "target", "ligand_complex", "receptor_complex", "call"];
const NUM: CompareNum[] = [
  "lr_means_a", "lr_means_b", "delta", "log2_fold", "magnitude_rank_a", "magnitude_rank_b",
  "specificity_rank_a", "specificity_rank_b", "p_value", "p_adj",
];

export function toCompareRows(res: CompareResult): CompareRow[] {
  const n = res.columns.source.length;
  const rows: CompareRow[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const r = { index: i } as CompareRow;
    for (const c of TEXT) r[c] = res.columns[c][i] ?? "";
    for (const c of NUM) r[c] = res.columns[c][i] ?? null;
    r.n_donors = res.columns.n_donors[i] ?? 0;
    r.composition_note = Boolean(res.columns.composition_note[i]);
    rows[i] = r;
  }
  return rows;
}

/** "all" = every call; "moved" = all but unchanged. */
export type CallFilter = "moved" | "all" | CompareCall;

export interface CompareFilter {
  call: CallFilter;
  source: string; // "" = any
  target: string;
  query: string; // in the ligand or receptor name, ignoring case
  /** Only rows with an adjusted p under this (null = no p filter) */
  maxPadj: number | null;
}

export const DEFAULT_COMPARE_FILTER: CompareFilter = { call: "moved", source: "", target: "", query: "", maxPadj: null };

export function filterCompareRows(rows: CompareRow[], f: CompareFilter): CompareRow[] {
  const q = f.query.trim().toUpperCase();
  return rows.filter(
    (r) =>
      (f.call === "all" || (f.call === "moved" ? r.call !== "unchanged" : r.call === f.call)) &&
      (!f.source || r.source === f.source) &&
      (!f.target || r.target === f.target) &&
      (!q || r.ligand_complex.toUpperCase().includes(q) || r.receptor_complex.toUpperCase().includes(q)) &&
      (f.maxPadj === null || (r.p_adj !== null && r.p_adj < f.maxPadj)),
  );
}

export type CompareSortKey = CompareText | CompareNum | "abs_delta" | "n_donors";

/** Sorted copy; nulls last either way. */
export function sortCompareRows(rows: CompareRow[], key: CompareSortKey, ascending: boolean): CompareRow[] {
  const dir = ascending ? 1 : -1;
  const get = (r: CompareRow): string | number | null =>
    key === "abs_delta" ? (r.delta === null ? null : Math.abs(r.delta)) : r[key];
  return [...rows].sort((a, b) => {
    const x = get(a);
    const y = get(b);
    if (x === null) return y === null ? 0 : 1;
    if (y === null) return -1;
    if (typeof x === "string" || typeof y === "string") return dir * String(x).localeCompare(String(y));
    return dir * (x - y);
  });
}

/** Sender x receiver: B minus A, in interactions (count) or summed LR mean (strength). */
export function diffMatrix(res: CompareResult, weight: Weight): number[][] {
  const a = weight === "count" ? res.counts_a : res.strength_a;
  const b = weight === "count" ? res.counts_b : res.strength_b;
  return res.groups.map((_, i) => res.groups.map((__, j) => (b[i]?.[j] ?? 0) - (a[i]?.[j] ?? 0)));
}

/**
 * The colour range of a difference matrix: symmetric about 0, so equal gains
 * and losses get equally strong colours and 0 is the map's midpoint (white on
 * RdBu). Taken through resolveColorScale (signed values: diverging) and kept
 * symmetric when every difference has one sign.
 */
export function diffColorRange(
  matrix: number[][],
  diverging: DivergingColormapName,
): { zmin: number; zmax: number; colormap: DivergingColormapName } {
  const stats = overlayStats(matrix.flat());
  const scale = resolveColorScale("score", stats, {
    sequential: "viridis",
    diverging,
    signedMode: "diverging",
    clip: DEFAULT_CLIP,
  });
  const m = Math.max(Math.abs(scale.min), Math.abs(scale.max), stats.n ? 0 : 1) || 1;
  return { zmin: -m, zmax: m, colormap: diverging };
}

/** "Mono → T: CXCL10 → CXCR3" */
export const compareLabel = (r: Pick<CompareRow, "source" | "target" | "ligand_complex" | "receptor_complex">) =>
  `${r.source} → ${r.target}: ${r.ligand_complex} → ${r.receptor_complex}`;

export const CALL_LABELS: Record<CompareCall, string> = {
  gained: "Gained",
  lost: "Lost",
  changed: "Changed",
  unchanged: "Unchanged",
};

/** The one-sentence summary under the result. */
export function compareSummary(res: CompareResult): string {
  const c = res.calls;
  return (
    `${res.b} vs ${res.a}: ${c.gained.toLocaleString()} gained, ${c.lost.toLocaleString()} lost, ` +
    `${c.changed.toLocaleString()} changed, ${c.unchanged.toLocaleString()} unchanged of the ` +
    `${res.n_rows.toLocaleString()} interactions specific in either condition`
  );
}
