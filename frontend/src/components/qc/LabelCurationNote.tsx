/** Curated vs author cell-type labels (backend core/label_curation.py `check`). */
export interface LabelCurationRow {
  author: string;
  n_cells: number;
  /** The curated label most of the author label's cells carry, and their share. */
  curated: string;
  share: number;
  others: { curated: string; n_cells: number }[];
  /** Why the mapping looks implausible; absent when it does not. */
  flag?: string;
}

export interface LabelCuration {
  curated_column: string;
  author_column: string;
  n_author_labels: number;
  n_cells: number;
  /** Largest author labels first (at most 100). */
  rows: LabelCurationRow[];
  n_flagged: number;
  flags: string[];
  note: string | null;
}

/**
 * A cautious note when the curated cell types (CELLxGENE's `cell_type`) look like they lost the
 * authors' meaning for some labels, with the cross-tab to check. Silent when nothing is flagged.
 */
export function LabelCurationNote({ info }: { info?: LabelCuration | null }) {
  if (!info || !info.n_flagged || !info.note) return null;
  return (
    <div className="mt-2 rounded border border-amber-200 bg-amber-50 px-2.5 py-2 text-[11px] leading-snug text-slate-600" data-testid="label-curation-note">
      <p className="font-medium text-amber-800">{info.note}</p>
      <ul className="mt-1 list-disc space-y-0.5 pl-4">
        {info.flags.map((f) => (
          <li key={f}>{f}.</li>
        ))}
      </ul>
      <details className="mt-1">
        <summary className="cursor-pointer text-slate-500">
          Cross-tab: {info.author_column} against {info.curated_column} ({info.rows.length} of {info.n_author_labels} author
          labels, {info.n_cells.toLocaleString()} cells)
        </summary>
        <div className="mt-1 max-h-64 overflow-y-auto">
          <table className="w-full text-[11px]" aria-label="Author labels against curated labels">
            <thead className="sticky top-0 bg-amber-50">
              <tr className="text-left text-slate-400">
                <th className="font-medium">{info.author_column}</th>
                <th className="text-right font-medium" title="Cells with both labels">Cells</th>
                <th className="pl-2 font-medium" title="The curated label most of these cells carry, and the share that do">
                  Mostly curated as
                </th>
                <th className="font-medium" title="The next curated labels among these cells">Also</th>
              </tr>
            </thead>
            <tbody>
              {info.rows.map((r) => (
                <tr key={r.author} className={r.flag ? "font-medium text-amber-800" : undefined}>
                  <td className="max-w-[10rem] truncate pr-2">{r.author}</td>
                  <td className="text-right">{r.n_cells.toLocaleString()}</td>
                  <td className="pl-2">
                    {r.curated} ({Math.round(r.share * 100)}%)
                  </td>
                  <td className="text-slate-400">
                    {r.others.map((o) => `${o.curated} (${o.n_cells.toLocaleString()})`).join(", ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
