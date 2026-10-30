/** What a targeted panel changes in the run (backend core/targeted_panel.py `assessment`). */
export interface TargetedPanelInfo {
  n_genes: number;
  technology: string;
  panel_name: string;
  sentences: string[];
  control: { median_pct: number; share_above: number; above_pct: number; flag: string | null } | null;
  /** The filter and run forms show these; only what the user changes is sent. */
  defaults: Record<string, number | null>;
  all_genes: boolean;
}

/**
 * The Data Assessment's line on the panel path: which defaults the run uses instead, and the
 * share of control counts. Anything the user sets in the steps below wins.
 */
export function TargetedPanelNote({ info }: { info?: TargetedPanelInfo | null }) {
  if (!info) return null;
  const [lead, ...rest] = info.sentences;
  return (
    <div className="mt-2 rounded border border-sky-200 bg-sky-50 px-2.5 py-2 text-[11px] leading-snug text-slate-600" data-testid="targeted-panel-note">
      <p className="font-medium text-slate-700">
        {lead}
        {info.panel_name ? ` (${info.panel_name})` : ""}
      </p>
      <ul className="mt-1 list-disc space-y-0.5 pl-4">
        {rest.map((s) => (
          <li key={s}>{s}</li>
        ))}
        {info.all_genes && <li>Setting a Variable Genes parameter below turns the selection back on.</li>}
      </ul>
      {info.control && (
        <p className="mt-1">
          Control counts per cell: median {info.control.median_pct.toFixed(2)}%;{" "}
          {(info.control.share_above * 100).toFixed(1)}% of cells above {info.control.above_pct}%.
        </p>
      )}
      {info.control?.flag && <p className="mt-1 text-amber-700">{info.control.flag}</p>}
    </div>
  );
}
