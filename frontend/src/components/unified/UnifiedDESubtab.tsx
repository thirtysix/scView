import { useCallback, useMemo, useState } from "react";
import { Loader2, Download, Lasso, FlaskConical } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { computeDE, type DEResponse } from "@/api/de";
import { VolcanoPlot } from "@/components/plots/VolcanoPlot";
import { downloadCsv } from "@/lib/csv";
import { formatPValue } from "@/lib/formatting";

interface Props {
  onGeneClick: (gene: string) => void;
}

/**
 * Differential expression for the current cell selection (lasso or a clicked
 * cluster) vs the rest, shown as a volcano plot + a significant-gene table.
 */
export function UnifiedDESubtab({ onGeneClick }: Props) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);

  const [result, setResult] = useState<DEResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fcThreshold, setFcThreshold] = useState(1);
  const [pThreshold] = useState(0.05);

  const nSelected = selected?.size ?? 0;

  const runDE = useCallback(async () => {
    if (!datasetId || !selected || selected.size < 3) return;
    setLoading(true);
    setError(null);
    try {
      const label = highlighted?.value ?? "selection";
      const res = await computeDE(datasetId, Array.from(selected), label);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, selected, highlighted]);

  // Significant genes, most significant first, for the table.
  const sigGenes = useMemo(() => {
    if (!result) return [];
    return result.genes
      .filter((g) => g.pval_adj <= pThreshold && Math.abs(g.logfoldchange) >= fcThreshold)
      .sort((a, b) => a.pval_adj - b.pval_adj || Math.abs(b.logfoldchange) - Math.abs(a.logfoldchange));
  }, [result, pThreshold, fcThreshold]);

  const exportCSV = useCallback(() => {
    if (!result) return;
    downloadCsv(
      `de_${result.label}_${datasetId}.csv`,
      ["gene", "log2_fold_change", "p_value", "p_value_adjusted"],
      result.genes.map((g) => [
        g.gene,
        g.logfoldchange.toFixed(4),
        g.pval.toExponential(4),
        g.pval_adj.toExponential(4),
      ]),
    );
  }, [result, datasetId]);

  if (!datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Selection status + run */}
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs">
        {nSelected < 3 ? (
          <div className="flex items-start gap-2 text-slate-500">
            <Lasso className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
            <span>
              Lasso-select cells on the scatter (or click a cluster in the legend), then compute
              differential expression of that group vs the rest.
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-2">
            <span className="text-slate-600">
              <span className="font-semibold text-slate-800">{nSelected.toLocaleString()}</span> cells
              selected{highlighted?.value ? ` (${highlighted.value})` : ""}
            </span>
            <button
              onClick={runDE}
              disabled={loading}
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <FlaskConical className="h-3 w-3" />}
              Compute DE
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">{error}</div>
      )}

      {result && (
        <>
          {/* Threshold + export */}
          <div className="flex items-center gap-2 text-xs">
            <label className="whitespace-nowrap font-medium text-slate-500">|log2FC| &ge;</label>
            <input
              type="range"
              min={0}
              max={5}
              step={0.25}
              value={fcThreshold}
              onChange={(e) => setFcThreshold(parseFloat(e.target.value))}
              className="h-1 flex-1 cursor-pointer appearance-none rounded-lg bg-slate-200 accent-primary"
            />
            <span className="w-8 text-right tabular-nums text-slate-600">{fcThreshold.toFixed(2)}</span>
            <button
              onClick={exportCSV}
              title="Export DE table as CSV"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
            >
              <Download className="h-3 w-3" />
              CSV
            </button>
          </div>

          <div className="text-[10px] text-slate-400">
            {result.label} (n={result.n_selected.toLocaleString()}) vs rest (n=
            {result.n_rest.toLocaleString()}) · {sigGenes.length.toLocaleString()} significant of{" "}
            {result.genes.length.toLocaleString()} genes · click a point or row to overlay it
          </div>

          {/* Volcano */}
          <div className="rounded-lg border border-slate-200">
            <VolcanoPlot
              genes={result.genes}
              fcThreshold={fcThreshold}
              pThreshold={pThreshold}
              onGeneClick={onGeneClick}
            />
          </div>

          {/* Top significant genes */}
          <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="px-2 py-1.5 text-left font-medium text-slate-600">Gene</th>
                  <th className="px-2 py-1.5 text-right font-medium text-slate-600">log2FC</th>
                  <th className="px-2 py-1.5 text-right font-medium text-slate-600">Adj.P</th>
                </tr>
              </thead>
              <tbody>
                {sigGenes.slice(0, 200).map((g) => (
                  <tr
                    key={g.gene}
                    onClick={() => onGeneClick(g.gene)}
                    className="cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50"
                  >
                    <td className="px-2 py-1.5 font-mono font-medium text-slate-800">{g.gene}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      <span className={g.logfoldchange > 0 ? "text-red-600" : "text-blue-600"}>
                        {g.logfoldchange.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">
                      {formatPValue(g.pval_adj)}
                    </td>
                  </tr>
                ))}
                {sigGenes.length === 0 && (
                  <tr>
                    <td colSpan={3} className="px-2 py-4 text-center text-slate-400">
                      No genes pass the thresholds.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
