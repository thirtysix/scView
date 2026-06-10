/**
 * CSV export helpers shared by every results table.
 *
 * Values are RFC-4180-escaped: a cell containing a comma, double-quote, or
 * newline is wrapped in double-quotes with internal quotes doubled. (The old
 * per-table `[...].join(",")` left commas in e.g. enrichment terms unescaped,
 * which silently corrupted those rows.)
 */

export type CsvCell = string | number | null | undefined;

function escapeCell(value: CsvCell): string {
  const s = value == null ? "" : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** Build an RFC-4180 CSV string from a header row + data rows. */
export function buildCsv(headers: string[], rows: CsvCell[][]): string {
  return [headers, ...rows].map((row) => row.map(escapeCell).join(",")).join("\n");
}

/** Build a CSV and trigger a browser download. */
export function downloadCsv(filename: string, headers: string[], rows: CsvCell[][]): void {
  const blob = new Blob([buildCsv(headers, rows)], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename.endsWith(".csv") ? filename : `${filename}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
