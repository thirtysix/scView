import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Panel } from "@/components/common/Panel";
import { InfoTip } from "@/components/common/InfoTip";
import { getMeanVariance, type MeanVariance } from "@/api/normalization";
import { meanVarianceTraces, splitGenes, type BeforeMode } from "@/lib/meanVariance";
import { NORMALIZATION_LABELS } from "@/lib/pipelineParams";
import { plotConfig } from "@/lib/plotConfig";

const FILL_STYLE = { width: "100%", height: "100%" } as const;

function axisLayout(yTitle: string) {
  return {
    autosize: true,
    margin: { l: 52, r: 8, t: 8, b: 40 },
    showlegend: true,
    // Inside the plot, top left (empty there: variance rises with the mean).
    legend: { x: 0.01, y: 0.99, bgcolor: "rgba(255,255,255,0.75)", font: { size: 9 } },
    // dtick 1 = one tick per decade; Plotly otherwise labels 2..9 within each.
    xaxis: { type: "log" as const, dtick: 1, title: { text: "mean count", font: { size: 10 } }, tickfont: { size: 9 } },
    yaxis: { type: "log" as const, dtick: 1, title: { text: yTitle, font: { size: 10 } }, tickfont: { size: 9 } },
  };
}

/**
 * Each gene's mean against its variance, before and after normalisation: the
 * counts' variance grows faster than a Poisson's, and Pearson residuals bring a
 * gene that varies by sampling alone to about 1, whatever its mean. Computed on
 * request (residuals of every gene take a few seconds).
 */
export function MeanVariancePlot({ datasetId, refreshKey = 0 }: { datasetId: string; refreshKey?: number }) {
  const [data, setData] = useState<MeanVariance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [before, setBefore] = useState<BeforeMode>("counts");

  // A new dataset or a pipeline run makes the shown diagnostic stale.
  useEffect(() => {
    setData(null);
    setError(null);
  }, [datasetId, refreshKey]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getMeanVariance(datasetId));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  const hidden = data ? splitGenes(data.mean, before === "counts" ? data.var_counts : data.var_log, null).hidden : 0;

  return (
    <div className="space-y-2 rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-sm font-semibold text-slate-700">Mean–variance</h3>
          <InfoTip label="About the mean–variance plot">
            Counts vary more than a Poisson (var = mean) as the mean grows. The shifted log damps
            that trend; analytic Pearson residuals put a gene that varies by sampling alone at
            about 1 at any mean, so genes above the line vary for a reason. Red: the dataset&apos;s
            current variable genes.
          </InfoTip>
        </div>
        {data ? (
          <div className="flex items-center gap-1 text-[11px]">
            <span className="text-slate-500">Before:</span>
            {(["counts", "log"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setBefore(m)}
                className={`rounded px-1.5 py-0.5 ${
                  before === m ? "bg-primary/10 font-medium text-primary" : "text-slate-500 hover:bg-slate-100"
                }`}
              >
                {m === "counts" ? "Counts" : "Shifted log"}
              </button>
            ))}
          </div>
        ) : (
          <button
            onClick={load}
            disabled={loading}
            className="inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
          >
            {loading && <Loader2 className="h-3 w-3 animate-spin" />}
            {loading ? "Computing…" : "Show mean–variance"}
          </button>
        )}
      </div>
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}
      {data && (
        <>
          <div className="grid grid-cols-1 gap-2 xl:grid-cols-2">
            <Panel title={before === "counts" ? "Before: counts" : "Before: shifted log"} className="h-[300px]">
              <Plot
                data={meanVarianceTraces(data, "before", before)}
                layout={axisLayout(before === "counts" ? "variance" : "variance (log values)")}
                config={plotConfig(`mean_variance_${before}`)}
                style={FILL_STYLE}
                useResizeHandler
              />
            </Panel>
            <Panel title="After: Pearson residuals" className="h-[300px]">
              <Plot
                data={meanVarianceTraces(data, "after")}
                layout={axisLayout("residual variance")}
                config={plotConfig("mean_variance_residuals")}
                style={FILL_STYLE}
                useResizeHandler
              />
            </Panel>
          </div>
          <p className="text-[11px] text-slate-400">
            {data.n_genes.toLocaleString()} genes on {data.n_cells_used.toLocaleString()}
            {data.n_cells_used < data.n_cells ? ` sampled of ${data.n_cells.toLocaleString()}` : ""} cells;
            counts from {data.counts_source}; θ = {data.theta}.
            {data.hvg
              ? ` Variable genes as last chosen${
                  data.normalization_method
                    ? ` (${NORMALIZATION_LABELS[data.normalization_method] ?? data.normalization_method})`
                    : ""
                }.`
              : " No variable genes chosen yet."}
            {hidden > 0 ? ` ${hidden} genes with zero variance are off the log scale.` : ""}
          </p>
        </>
      )}
    </div>
  );
}
