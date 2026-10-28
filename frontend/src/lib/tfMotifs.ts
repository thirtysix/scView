import type { TfMotifResult, TfMotifRow } from "@/api/tfMotifs";
import type { CsvCell } from "@/lib/csv";

export type TfSortKey = "padj" | "odds_ratio" | "k";

/** Rows in the chosen order: adjusted p ascending, or odds ratio / overlap descending
 *  (ties by p). */
export function sortTfRows(rows: TfMotifRow[], key: TfSortKey): TfMotifRow[] {
  const out = [...rows];
  if (key === "padj") out.sort((a, b) => a.padj - b.padj || a.p - b.p);
  else out.sort((a, b) => b[key] - a[key] || a.p - b.p);
  return out;
}

/** A p-value in a few characters: 0.031, 2.4e-7. */
export function formatP(p: number): string {
  if (!Number.isFinite(p)) return "–";
  if (p === 0) return "0";
  return p < 0.001 ? p.toExponential(1) : p.toPrecision(2);
}

/** The gene to colour the cells by for a row: the expressed member, else the first TF. */
export function tfGene(row: TfMotifRow): string {
  return row.tf_expressed?.gene ?? row.tfs[0] ?? row.motif;
}

export const TF_CSV_HEADERS = [
  "group", "motif_id", "motif", "k", "n", "K", "N", "odds_ratio", "p", "padj", "gc_rich",
  "tf_gene", "tf_pct_expressing", "tf_logfc", "activity_agrees", "overlap_genes",
];

/** Every returned row of every group, for the CSV. */
export function tfCsvRows(result: TfMotifResult): CsvCell[][] {
  return result.groups.flatMap((g) =>
    g.rows.map((r) => [
      g.group, r.motif_id, r.motif, r.k, r.n, r.K, r.N, r.odds_ratio, r.p, r.padj, r.gc_rich ? "yes" : "no",
      r.tf_expressed?.gene ?? "", r.tf_expressed?.pct ?? "", r.tf_expressed?.logfc ?? "",
      r.activity_agrees === null ? "" : r.activity_agrees ? "yes" : "no", r.overlap.join(";"),
    ]),
  );
}
