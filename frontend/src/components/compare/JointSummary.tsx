import { useEffect, useState } from "react";
import { getJointSummary, type BatchMixing, type JointSummary as Summary } from "@/api/integration";

/** A score as a share of the ideal, for bar widths: 0 = datasets apart, 1 = ideal. */
export function shareOfIdeal(value: number | null | undefined, ideal: number): number {
  if (value == null || !(ideal > 0)) return 0;
  return Math.max(0, Math.min(1, value / ideal));
}

/**
 * Quality checks for a joint dataset: how well its source datasets mix before and
 * after batch correction. Renders nothing for a dataset that isn't joint.
 */
export function JointSummary({ datasetId, title }: { datasetId: string; title: string }) {
  const [summary, setSummary] = useState<Summary | null>(null);

  useEffect(() => {
    let cancelled = false;
    setSummary(null);
    getJointSummary(datasetId)
      .then((s) => !cancelled && setSummary(s))
      .catch(() => !cancelled && setSummary(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  if (!summary?.joint) return null;
  return (
    <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <h3 className="text-sm font-semibold text-slate-700">{title}</h3>
      {summary.mixing ? (
        <MixingScore mixing={summary.mixing} />
      ) : (
        <p className="text-sm text-slate-400">No batch-mixing score for this dataset.</p>
      )}
    </div>
  );
}

function MixingScore({ mixing }: { mixing: BatchMixing }) {
  const rows: { label: string; value: number | null }[] = [
    { label: "Before correction", value: mixing.uncorrected },
  ];
  if (mixing.corrected != null) rows.push({ label: "After correction", value: mixing.corrected });
  const fmt = (v: number | null) => (v == null ? "–" : v.toFixed(3));

  return (
    <div>
      <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
        Batch mixing
      </h4>
      <div className="space-y-1.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-3 text-sm">
            <span className="w-36 flex-shrink-0 text-slate-600">{r.label}</span>
            <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100">
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${shareOfIdeal(r.value, mixing.ideal) * 100}%` }}
              />
            </div>
            <span className="w-14 text-right font-mono text-slate-800">{fmt(r.value)}</span>
          </div>
        ))}
        <div className="flex items-center gap-3 text-sm">
          <span className="w-36 flex-shrink-0 text-slate-400">Ideal</span>
          <div className="flex-1" />
          <span className="w-14 text-right font-mono text-slate-500">{fmt(mixing.ideal)}</span>
        </div>
      </div>
      <p className="mt-2 text-xs text-slate-500">
        Share of each cell&apos;s {mixing.k} nearest neighbours that come from another dataset, averaged
        over {mixing.n_scored.toLocaleString()}
        {mixing.n_scored < mixing.n_cells ? ` of ${mixing.n_cells.toLocaleString()}` : ""} cells. Near 0,
        the datasets sit apart; the ideal is what perfect mixing gives for these dataset sizes. High
        mixing is only expected where the datasets share cell types.
      </p>
      {mixing.datasets.length > 0 && (
        <table className="mt-2 w-full text-xs">
          <thead>
            <tr className="text-left text-slate-400">
              <th className="py-1 font-medium">Dataset</th>
              <th className="py-1 text-right font-medium">Cells</th>
              <th className="py-1 text-right font-medium">Before</th>
              {mixing.corrected != null && <th className="py-1 text-right font-medium">After</th>}
              <th className="py-1 text-right font-medium">Ideal</th>
            </tr>
          </thead>
          <tbody className="font-mono text-slate-700">
            {mixing.datasets.map((d) => (
              <tr key={d.label} className="border-t border-slate-100">
                <td className="py-1 font-sans">{d.label}</td>
                <td className="py-1 text-right">{d.n_cells.toLocaleString()}</td>
                <td className="py-1 text-right">{fmt(d.uncorrected)}</td>
                {mixing.corrected != null && <td className="py-1 text-right">{fmt(d.corrected)}</td>}
                <td className="py-1 text-right text-slate-500">{fmt(d.ideal)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
