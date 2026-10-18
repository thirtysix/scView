import { useEffect, useState } from "react";
import { AlertTriangle, Palette } from "lucide-react";
import { getCellTypistSummary, type CellTypistSummary as Summary } from "@/api/annotation";
import { pct } from "@/lib/qcThresholds";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const ROWS = 8; // more behind "show all"
const NUM = "whitespace-nowrap pl-2 text-right";

const conf = (v: number | null) => (v == null ? "—" : v.toFixed(2));

/** Colour the Unified View scatter by `column` and go there. */
function colourBy(column: string) {
  useSettingsStore.getState().setColorBy(column);
  useViewStore.getState().setPanel("unified");
}

/**
 * How sure a CellTypist run was, per label (core/celltypist_fit.py): cells,
 * median confidence (highest class probability), share under 0.5, and how many
 * cells' own call agrees with the label; cautions for a species mismatch, low
 * gene overlap and a low median. `refreshKey` refetches after a new run.
 */
export function CellTypistSummary({
  datasetId,
  column,
  refreshKey = 0,
}: {
  datasetId: string;
  column: string;
  refreshKey?: number;
}) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let live = true;
    setSummary(null);
    getCellTypistSummary(datasetId, column)
      .then((info) => live && setSummary(info.result))
      .catch(() => live && setSummary(null));
    return () => {
      live = false;
    };
  }, [datasetId, column, refreshKey]);

  if (!summary) return null;
  const rows = showAll ? summary.labels : summary.labels.slice(0, ROWS);
  const overlap =
    summary.gene_overlap == null
      ? null
      : summary.n_shared_genes != null && summary.n_model_genes != null
        ? `${summary.n_shared_genes.toLocaleString()} of ${summary.n_model_genes.toLocaleString()} model genes (${pct(summary.gene_overlap)})`
        : pct(summary.gene_overlap);

  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50/60 p-3" data-testid="celltypist-summary">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold text-slate-700">
          CellTypist confidence{summary.model ? ` · ${summary.model.replace(/\.pkl$/, "")}` : ""}
        </p>
        <button
          type="button"
          onClick={() => colourBy(summary.confidence_column)}
          title={`Colour the Unified View scatter by ${summary.confidence_column}`}
          className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-700 hover:border-primary hover:text-primary"
        >
          <Palette className="h-3.5 w-3.5" />
          Colour embedding by confidence
        </button>
      </div>
      <p className="mb-2 text-[11px] text-slate-500">
        Median {conf(summary.median_confidence)}; {pct((summary.pct_below ?? 0) / 100)} of cells under{" "}
        {summary.low_confidence}
        {summary.n_low_labels > 0 &&
          `; ${summary.n_low_labels} label${summary.n_low_labels === 1 ? "" : "s"} with a low median`}
        {overlap && <>. Genes shared: {overlap}</>}.
      </p>

      {summary.cautions.length > 0 && (
        <ul className="mb-2 space-y-1">
          {summary.cautions.map((c) => (
            <li key={c} role="alert" className="flex gap-1.5 rounded bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              <AlertTriangle className="mt-0.5 h-3 w-3 flex-shrink-0" />
              <span>{c}</span>
            </li>
          ))}
        </ul>
      )}

      <table className="w-full text-[11px]">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="font-medium">Label</th>
            <th className={`${NUM} font-medium`}>Cells</th>
            <th className={`${NUM} font-medium`} title="Median of each cell's highest class probability">
              Median
            </th>
            <th className={`${NUM} font-medium`} title={`Cells with confidence under ${summary.low_confidence}`}>
              &lt; {summary.low_confidence}
            </th>
            {summary.percell_column && (
              <th className={`${NUM} font-medium`} title="Cells whose own best class is this label (before majority voting)">
                Own call agrees
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className={r.low_confidence ? "text-amber-800" : "text-slate-700"}>
              <td className="max-w-[14rem] truncate" title={r.label}>
                {r.label}
                {r.low_confidence && (
                  <span className="ml-1 rounded bg-amber-100 px-1 text-[10px] font-medium">low</span>
                )}
              </td>
              <td className={NUM}>{r.n_cells.toLocaleString()}</td>
              <td className={NUM}>{conf(r.median_confidence)}</td>
              <td className={NUM}>{r.pct_below == null ? "—" : `${r.pct_below.toFixed(0)}%`}</td>
              {summary.percell_column && (
                <td className={NUM}>{r.pct_agree == null ? "—" : `${r.pct_agree.toFixed(0)}%`}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      {summary.labels.length > ROWS && (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="mt-1 text-[11px] text-primary hover:underline"
        >
          {showAll ? "Show fewer" : `Show all ${summary.labels.length}`}
        </button>
      )}
      <p className="mt-2 text-[10px] text-slate-400">{summary.note}</p>
    </div>
  );
}
