/**
 * "Import labels from a file" (Data Assessment): a per-cell label table (an author
 * annotation from GEO, a collaborator's file) into a new user label column. The
 * backend matches and checks it (core/label_import.py); this reads the table's header
 * so the user can pick its columns, with the backend's own separator rule.
 */
import { columnNameError } from "@/lib/labels";

/** The barcode choice that means "no barcode column: row i labels cell i". */
export const BY_ORDER = "\u0000order";

export interface TableHeader {
  columns: string[];
  sep: string;
}

/** The header row's separator, as the backend reads it: a tab, else a comma, else a semicolon;
 *  a one-column header goes by the file's extension. */
export function separatorOf(firstLine: string, fileName: string): string {
  for (const sep of ["\t", ",", ";"]) if (firstLine.includes(sep)) return sep;
  return fileName.toLowerCase().replace(/\.gz$/, "").endsWith(".csv") ? "," : "\t";
}

/** The column names pandas gives a header row (an empty name is "Unnamed: i", as R's write.csv leaves the first). */
export function parseHeader(firstLine: string, fileName: string): TableHeader {
  const line = firstLine.replace(/^\uFEFF/, "").replace(/\r$/, "");
  const sep = separatorOf(line, fileName);
  const columns = line.split(sep).map((c, i) => {
    const name = c.trim().replace(/^"(.*)"$/, "$1");
    return name === "" ? `Unnamed: ${i}` : name;
  });
  return { columns, sep };
}

const HEAD_BYTES = 64 * 1024;

/** The first line of a (gzipped) text file, or null when it cannot be read here. */
export async function readFirstLine(file: Blob): Promise<string | null> {
  try {
    const head = new Uint8Array(await file.slice(0, HEAD_BYTES).arrayBuffer());
    let bytes = head;
    if (head[0] === 0x1f && head[1] === 0x8b) {
      if (typeof DecompressionStream === "undefined") return null;
      // The head alone: the first decompressed chunk holds the header line.
      const body = new Response(head).body;
      if (!body) return null;
      const stream = body.pipeThrough(new DecompressionStream("gzip"));
      const reader = stream.getReader();
      const { value } = await reader.read();
      await reader.cancel().catch(() => undefined);
      if (!value) return null;
      bytes = value;
    }
    const text = new TextDecoder().decode(bytes);
    return text.split("\n", 1)[0] ?? null;
  } catch {
    return null;
  }
}

/** The table's barcode and label columns as the backend would take them: the first column,
 *  and the one other column (or, with more, the second, which the user should check). */
export function defaultColumns(columns: readonly string[]): { barcode: string; label: string } {
  const barcode = columns[0] ?? BY_ORDER;
  const label = columns.length >= 2 ? (columns[1] ?? "") : (columns[0] ?? "");
  return { barcode: columns.length >= 2 ? barcode : BY_ORDER, label };
}

/** A free name for the new column: the label column's own name when it can be one, else
 *  `imported_labels` (`_2`, ...). */
export function suggestImportName(labelColumn: string, existing: readonly string[]): string {
  const own = labelColumn.trim().replace(/[^A-Za-z0-9 _.-]/g, "_");
  if (own && !columnNameError(own, existing)) return own;
  for (let i = 1; ; i++) {
    const name = i === 1 ? "imported_labels" : `imported_labels_${i}`;
    if (!columnNameError(name, existing)) return name;
  }
}

/** The import's answer (POST /labels/import-file). */
export interface LabelImportReport {
  column: string;
  created: boolean;
  match: "name" | "order";
  n_rows: number;
  n_blank: number;
  n_cells: number;
  n_matched: number;
  n_extra: number;
  n_duplicates: number;
  share_matched?: number;
  extra_examples?: string[];
  n_unmatched_cells: number;
  unmatched_examples?: string[];
  n_labels: number;
  labels: { label: string; n_cells: number }[];
  n_imported_labels?: number;
  hint?: string;
  note?: string;
  dry_run?: boolean;
}

const n = (v: number) => v.toLocaleString("en-US");
const examples = (xs?: string[]) => (xs && xs.length ? ` (e.g. ${xs.slice(0, 3).join(", ")})` : "");

/** The match report in sentences, matched first. */
export function describeReport(r: LabelImportReport): string[] {
  const labelled = r.n_rows - r.n_blank;
  const lines = [
    r.match === "order"
      ? `${n(r.n_matched)} of ${n(r.n_cells)} cells get a label, row by row in the dataset's order.`
      : `${n(r.n_matched)} of ${n(labelled)} labelled rows name a cell of this dataset` +
        (r.share_matched != null ? ` (${(100 * r.share_matched).toFixed(1)}%).` : "."),
  ];
  if (r.n_extra > 0) lines.push(`${n(r.n_extra)} rows name no cell here${examples(r.extra_examples)}.`);
  if (r.n_unmatched_cells > 0)
    lines.push(`${n(r.n_unmatched_cells)} cells are not in the table and stay Unassigned${examples(r.unmatched_examples)}.`);
  if (r.n_blank > 0) lines.push(`${n(r.n_blank)} rows have no label and are skipped.`);
  if (r.n_duplicates > 0) lines.push(`${n(r.n_duplicates)} repeated rows (same cell, same label) count once.`);
  return lines;
}
