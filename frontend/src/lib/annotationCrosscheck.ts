/** Helpers for the annotation cross-check card (backend core/annotation_crosscheck.py). */
import type {
  CrosscheckContingency,
  CrosscheckInfo,
  CrosscheckLabelRow,
  CrosscheckResult,
  LabelTable,
} from "@/api/annotation";
import type { DatasetInfo } from "@/api/types";
import { isConfidenceOutput } from "@/lib/annotationConfidence";
import { rankGroupingColumns, type RankedGrouping } from "@/lib/obsColumns";
import { pct } from "@/lib/qcThresholds";

export const MAX_CROSSCHECK_COLUMNS = 6;

type DatasetColumns = Pick<DatasetInfo, "obs_columns" | "n_cells" | "active_clustering" | "user_annotation_columns">;

/**
 * Label columns the cross-check can take, best first: categorical groupings
 * (2–100 values) that are not confidence-check outputs. Boolean flags are left out.
 */
export function crosscheckColumns(dataset: DatasetColumns | null): RankedGrouping[] {
  if (!dataset) return [];
  const bools = new Set((dataset.obs_columns ?? []).filter((c) => c.dtype === "bool").map((c) => c.name));
  return rankGroupingColumns(dataset.obs_columns ?? [], {
    nCells: dataset.n_cells,
    activeClustering: dataset.active_clustering,
    userColumns: dataset.user_annotation_columns,
  }).filter((c) => !isConfidenceOutput(c.name) && !bools.has(c.name));
}

/**
 * The columns the card opens with: those an annotation method made (CellTypist,
 * LLM, marker score) and the active clustering; topped up to two from the
 * ranking (cell types and hand labels come first there). At most `max`.
 */
export function defaultSelection(
  available: readonly RankedGrouping[],
  info: Pick<CrosscheckInfo, "suggested" | "active_clustering"> | null,
  max = MAX_CROSSCHECK_COLUMNS,
): string[] {
  const names = new Set(available.map((c) => c.name));
  const picked = available.filter((c) => info?.suggested[c.name]).map((c) => c.name);
  const active = info?.active_clustering;
  if (active && names.has(active) && !picked.includes(active)) picked.push(active);
  for (const c of available) {
    if (picked.length >= 2) break;
    if (!picked.includes(c.name)) picked.push(c.name);
  }
  return picked.slice(0, max);
}

/**
 * The reference among `selected`: the first annotation that is not a
 * clustering (hand labels, then cell types by the ranking), else the first.
 */
export function defaultReference(
  selected: readonly string[],
  available: readonly RankedGrouping[],
  info: Pick<CrosscheckInfo, "suggested"> | null,
): string {
  const kind = new Map(available.map((c) => [c.name, c.kind]));
  const isClustering = (name: string) =>
    info?.suggested[name] === "clustering" || (kind.get(name) === "clustering" && !info?.suggested[name]);
  const ranked = available.map((c) => c.name).filter((n) => selected.includes(n));
  return ranked.find((n) => !isClustering(n)) ?? selected[0] ?? "";
}

/**
 * Rows and columns of a contingency table ordered so best matches sit on the
 * diagonal: reference rows keep their order, and each other label goes under
 * the row holding most of its cells (bigger overlaps first within a row).
 */
export function diagonalOrder(c: Pick<CrosscheckContingency, "rows" | "cols" | "counts">): {
  rows: string[];
  cols: string[];
} {
  const rowIndex = new Map(c.rows.map((r, i) => [r, i]));
  const best = new Map<string, { row: number; n: number }>();
  for (const r of c.rows) {
    for (const [col, n] of Object.entries(c.counts[r] ?? {})) {
      const cur = best.get(col);
      const row = rowIndex.get(r)!;
      if (!cur || n > cur.n || (n === cur.n && row < cur.row)) best.set(col, { row, n });
    }
  }
  const cols = c.cols
    .map((col, i) => ({ col, i, b: best.get(col) ?? { row: c.rows.length, n: 0 } }))
    .sort((a, b) => a.b.row - b.b.row || b.b.n - a.b.n || a.i - b.i)
    .map((x) => x.col);
  return { rows: [...c.rows], cols };
}

/** A label-keyed table as a matrix in the given order; missing entries are 0. */
export function tableMatrix(table: LabelTable, rows: readonly string[], cols: readonly string[]): number[][] {
  return rows.map((r) => cols.map((c) => table[r]?.[c] ?? 0));
}

/** A pairwise score (ARI/AMI) as a matrix over `columns`, null where missing. */
export function pairMatrix(
  scores: Record<string, Record<string, number | null>>,
  columns: readonly string[],
): (number | null)[][] {
  return columns.map((a) => columns.map((b) => scores[a]?.[b] ?? null));
}

/** Nodes and links of a reference → other Sankey: reference labels left, the other column's right. */
export function sankeyData(c: Pick<CrosscheckContingency, "counts">, rows: readonly string[], cols: readonly string[]) {
  const colIndex = new Map(cols.map((col, i) => [col, rows.length + i]));
  const source: number[] = [];
  const target: number[] = [];
  const value: number[] = [];
  rows.forEach((r, i) => {
    for (const col of cols) {
      const n = c.counts[r]?.[col] ?? 0;
      if (n > 0) {
        source.push(i);
        target.push(colIndex.get(col)!);
        value.push(n);
      }
    }
  });
  return { labels: [...rows, ...cols], source, target, value };
}

export type LabelSortKey = "label" | "n_cells" | "mean_agreement" | "disagreeing";

export interface LabelTableRow extends CrosscheckLabelRow {
  label: string;
}

/** The per-label rows in the reference's order, then sorted by `key` (ties keep that order). */
export function sortLabelTable(
  labels: Record<string, CrosscheckLabelRow>,
  order: readonly string[],
  key: LabelSortKey,
  descending: boolean,
): LabelTableRow[] {
  const pos = new Map(order.map((l, i) => [l, i]));
  const rows = Object.entries(labels)
    .map(([label, row]) => ({ label, ...row }))
    .sort((a, b) => (pos.get(a.label) ?? Infinity) - (pos.get(b.label) ?? Infinity));
  const value = (r: LabelTableRow): number | string => {
    if (key === "label") return r.label;
    if (key === "disagreeing") return r.disagreeing.length;
    if (key === "mean_agreement") return r.mean_agreement ?? -1;
    return r.n_cells;
  };
  const sign = descending ? -1 : 1;
  return rows
    .map((r, i) => ({ r, i }))
    .sort((a, b) => {
      const va = value(a.r);
      const vb = value(b.r);
      const cmp = typeof va === "string" ? va.localeCompare(vb as string) : va - (vb as number);
      return sign * cmp || a.i - b.i;
    })
    .map((x) => x.r);
}

/**
 * What one column calls a reference label's cells, with where a call maps when
 * that is another reference label: "Monocyte (80.0%), T cell (15.0%, → CD4 T)".
 * The share sits in brackets, so a numeric label ("3") is not read as part of it.
 */
export function callsText(row: LabelTableRow, column: string): string {
  const a = row.annotators[column];
  if (!a) return "–";
  return a.calls
    .map((c) => `${c.label} (${pct(c.share)}${c.maps_to && c.maps_to !== row.label ? `, → ${c.maps_to}` : ""})`)
    .join(", ");
}

/** The headline: mapped agreement, and exact-name agreement apart from it. */
export function agreementSummary(r: CrosscheckResult): string {
  const parts = [
    r.mean_agreement === null
      ? "No cell has labels to compare."
      : `Mean agreement with ${r.reference}: ${pct(r.mean_agreement)} of ${r.n_compared.toLocaleString()} cells ` +
        "(other columns' labels mapped to the reference label they mostly fall in).",
  ];
  const shared = Object.values(r.comparisons).some((c) => c.exact_name.n_shared_labels > 0);
  if (r.exact_name_agreement !== null) {
    parts.push(
      shared
        ? `Same label name: ${pct(r.exact_name_agreement)} of cells.`
        : "The columns share no label names, so same-name agreement is 0 however well they agree.",
    );
  }
  if (r.n_unresolved > 0) parts.push(`${r.n_unresolved.toLocaleString()} cells have no majority (Unresolved).`);
  return parts.join(" ");
}

/** A saved-columns name the backend accepts (letters, digits, _ . -; up to 60). */
export function validCrosscheckName(name: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9_.-]{0,59}$/.test(name.trim());
}
