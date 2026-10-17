import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { getFilterPreview, type FilterGrouping, type FilterPreview, type FilterThresholds } from "@/api/qc";
import { pct } from "@/lib/qcThresholds";

const DEBOUNCE_MS = 150;
const ROWS = 6; // per table, more behind "show all"
const WARNINGS = 4; // whole groups first, then the worst; more behind "show all"

/** One table of groups (samples or clusters): most affected first. */
function LossTable({ title, grouping }: { title: string; grouping: FilterGrouping }) {
  const [all, setAll] = useState(false);
  if (grouping.note) return <p className="text-[11px] text-slate-400">{grouping.note}</p>;
  const rows = [...grouping.groups].sort((a, b) => b.fraction - a.fraction || b.n_cells - a.n_cells);
  const shown = all ? rows : rows.slice(0, ROWS);
  return (
    <div>
      <p className="mb-0.5 text-[11px] font-medium text-slate-600">
        {title} <span className="font-normal text-slate-400">({grouping.column})</span>
      </p>
      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-400">
            <th className="font-medium">{grouping.column}</th>
            <th className="text-right font-medium">Cells</th>
            <th className="text-right font-medium">Removed</th>
            <th className="text-right font-medium">Share</th>
          </tr>
        </thead>
        <tbody className="text-slate-600">
          {shown.map((g) => (
            <tr key={g.name}>
              <td className="max-w-[10rem] truncate pr-2" title={g.name}>
                {g.name}
              </td>
              <td className="text-right">{g.n_cells.toLocaleString()}</td>
              <td className="text-right">{g.n_removed.toLocaleString()}</td>
              <td
                className={`text-right ${
                  g.fraction >= 1 ? "font-semibold text-red-700" : g.fraction > 0.5 ? "font-semibold text-amber-700" : ""
                }`}
              >
                {pct(g.fraction)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > ROWS && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-[11px] text-primary hover:underline">
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}

/**
 * What the QC thresholds set on the histograms would remove, asked of the backend
 * on every change (debounced): overall, by each threshold on its own, per sample and
 * per cluster, with a warning for a sample or cluster removed whole or more than half.
 */
export function QcFilterSummary({
  datasetId,
  thresholds,
  refreshKey = 0,
  onDropDoublets,
}: {
  datasetId: string;
  thresholds: FilterThresholds;
  refreshKey?: number;
  onDropDoublets?: (on: boolean) => void;
}) {
  const [preview, setPreview] = useState<FilterPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [allWarnings, setAllWarnings] = useState(false);
  const request = useRef(0);
  const key = JSON.stringify(thresholds);

  useEffect(() => {
    const id = ++request.current;
    setLoading(true);
    const timer = setTimeout(() => {
      getFilterPreview(datasetId, JSON.parse(key) as FilterThresholds)
        .then((p) => {
          if (id !== request.current) return; // a newer threshold was asked for
          setPreview(p);
          setError(null);
        })
        .catch((e) => {
          if (id === request.current) setError(String(e?.message ?? e));
        })
        .finally(() => {
          if (id === request.current) setLoading(false);
        });
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [datasetId, key, refreshKey]);

  if (!preview) {
    return error ? (
      <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Cell-loss summary unavailable: {error}
      </p>
    ) : null;
  }

  const parts = Object.values(preview.by_criterion).map(
    (c) => `${c!.label.toLowerCase()} ${c!.n_removed.toLocaleString()}`,
  );
  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3" data-testid="qc-loss-summary">
      <div className="flex items-center justify-between gap-2">
        <h4 className="text-xs font-semibold text-slate-700">Cells these thresholds remove</h4>
        {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>
      <p className="text-[11px] text-slate-700">
        <span className="font-semibold">
          {preview.n_removed.toLocaleString()} of {preview.n_cells.toLocaleString()} cells (
          {pct(preview.fraction_removed)})
        </span>
        {parts.length > 0 ? `; each threshold on its own: ${parts.join(", ")}` : ": no threshold is on"}.
      </p>
      {preview.warnings.length > 0 && (
        <ul className="space-y-1" aria-label="Cell-loss warnings">
          {(allWarnings ? preview.warnings : preview.warnings.slice(0, WARNINGS)).map((w) => (
            <li
              key={`${w.kind}:${w.group}`}
              className={`flex items-start gap-1.5 rounded-md px-2 py-1 text-[11px] ${
                w.whole ? "bg-red-50 text-red-800" : "bg-amber-50 text-amber-800"
              }`}
            >
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{w.message}</span>
            </li>
          ))}
        </ul>
      )}
      {preview.warnings.length > WARNINGS && (
        <button
          type="button"
          onClick={() => setAllWarnings((v) => !v)}
          className="text-[11px] text-primary hover:underline"
        >
          {allWarnings ? "Fewer warnings" : `Show all ${preview.warnings.length} warnings`}
        </button>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        {preview.samples && <LossTable title="Per sample" grouping={preview.samples} />}
        {preview.clusters && <LossTable title="Per cluster" grouping={preview.clusters} />}
      </div>
      {preview.doublet_calls && onDropDoublets && (
        <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
          <input
            type="checkbox"
            checked={thresholds.drop_doublets}
            onChange={(e) => onDropDoublets(e.target.checked)}
            className="h-3 w-3"
          />
          Also drop predicted doublets
        </label>
      )}
      <p className="text-[11px] text-slate-400">
        {preview.has_mt ? "" : "% mito can't be measured here, so no mito threshold applies. "}
        These are the Cell/Gene Filtering step&apos;s parameters: they apply when that step runs (tick it,
        or Reprocess from Scratch).
      </p>
    </div>
  );
}
