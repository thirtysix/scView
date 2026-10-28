import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { Caption } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import {
  getDeconvolutionCheck,
  runDeconvolutionCheck,
  type CheckProgress,
  type DeconvCheck,
  type DeconvOptions,
} from "@/api/deconvolution";
import { aboutSeconds, checkScatterTraces, num } from "@/lib/deconvolution";
import { errorMessage } from "@/lib/errorMessage";
import { prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";

/**
 * The deconvolution self-check (backend deconvolution.iter_check): each donor held out
 * in turn, the reference without it estimates its pseudobulk and mixtures of its cells
 * whose fractions are known. Per-type r / RMSE / bias, and truth against estimate.
 */
export function DeconvolutionCheck({
  datasetId,
  options,
  nDonors,
  minDonors,
  estimateSeconds,
  colorOf,
}: {
  datasetId: string;
  options: DeconvOptions;
  nDonors: number;
  minDonors: number;
  estimateSeconds: number | null;
  colorOf: (type: string) => string;
}) {
  const [check, setCheck] = useState<DeconvCheck | null>(null);
  const [progress, setProgress] = useState<CheckProgress | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [method, setMethod] = useState<string>("");
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    getDeconvolutionCheck(datasetId)
      .then((c) => !cancelled && setCheck(c))
      .catch(() => undefined); // none yet, or the server is busy: the button still works
    return () => {
      cancelled = true;
      abort.current?.abort();
    };
  }, [datasetId]);

  const shownMethod = check && check.methods.includes(method as "weighted") ? method : (check?.methods[0] ?? "");
  const traces = useMemo(() => (check ? checkScatterTraces(check, shownMethod, colorOf) : []), [check, shownMethod, colorOf]);
  const perType = check?.per_type.filter((p) => p.method === shownMethod) ?? [];

  const run = () => {
    const ctl = new AbortController();
    abort.current = ctl;
    setRunning(true);
    setError(null);
    setProgress(null);
    runDeconvolutionCheck(datasetId, options, { onProgress: setProgress, signal: ctl.signal })
      .then((c) => {
        setCheck(c);
        setMethod(c.methods[0] ?? "");
      })
      .catch((e) => {
        if (!ctl.signal.aborted) setError(errorMessage(e));
      })
      .finally(() => {
        setRunning(false);
        setProgress(null);
      });
  };

  const tooFew = nDonors < minDonors;
  const time = aboutSeconds(estimateSeconds);

  return (
    <section className="space-y-2 rounded-lg border border-slate-200 p-2 dark:border-slate-700" aria-label="Self-check">
      <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-200">
        Self-check
        <InfoTip label="About the self-check" width={300}>
          Each donor is held out in turn. The reference built from the other donors estimates the held-out
          donor&apos;s own pseudobulk and mixtures of its cells with known fractions (random fractions, 2,000 to
          10,000 cells each). The mixtures come from this dataset, so this is a best case: real bulk samples add
          platform and protocol differences.
        </InfoTip>
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
        <button
          type="button"
          onClick={run}
          disabled={running || tooFew}
          className="flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-800"
        >
          {running && <Loader2 className="h-3 w-3 animate-spin" />}
          {running ? "Checking…" : check ? "Check again" : "Run self-check"}
        </button>
        {running && (
          <button type="button" onClick={() => abort.current?.abort()} className="text-primary hover:underline">
            Cancel
          </button>
        )}
        {running && progress && (
          <span data-testid="check-progress">
            {progress.done} of {progress.total} donors held out
          </span>
        )}
        {!running && !tooFew && time && <span>{nDonors} donors: {time}.</span>}
        {tooFew && (
          <span>
            Needs {minDonors}+ donors ({nDonors ? `${nDonors} in the donor column` : "pick a donor column"}).
          </span>
        )}
      </div>
      <RequestError message={error} />

      {check && (
        <div className="space-y-2">
          <table className="w-full text-[11px]" aria-label="Self-check overall">
            <thead>
              <tr className="border-b border-slate-200 text-left text-slate-600 dark:border-slate-700 dark:text-slate-300">
                <th className="px-1 py-0.5 font-medium">Method</th>
                <th className="px-1 py-0.5 text-right font-medium" title="Pearson correlation of estimate and truth, all types">r</th>
                <th className="px-1 py-0.5 text-right font-medium" title="Root mean squared error, in fraction units (0.05 = 5 points)">RMSE</th>
              </tr>
            </thead>
            <tbody>
              {check.overall.map((o) => (
                <tr key={o.method} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="px-1 py-0.5 text-slate-700 dark:text-slate-200">{check.method_names[o.method] ?? o.method}</td>
                  <td className="px-1 py-0.5 text-right tabular-nums">{num(o.r)}</td>
                  <td className="px-1 py-0.5 text-right tabular-nums">{num(o.rmse, 3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {check.methods.length > 1 && (
            <label className="flex items-center gap-1 text-[11px] text-slate-500">
              Per type for
              <select
                value={shownMethod}
                onChange={(e) => setMethod(e.target.value)}
                className="rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
              >
                {check.methods.map((m) => (
                  <option key={m} value={m}>
                    {check.method_names[m] ?? m}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="max-h-56 overflow-auto rounded border border-slate-200 dark:border-slate-700">
            <table className="w-full text-[11px]" aria-label="Self-check per cell type">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800">
                <tr className="border-b border-slate-200 text-left text-slate-600 dark:border-slate-700 dark:text-slate-300">
                  <th className="px-1 py-0.5 font-medium">Cell type</th>
                  <th className="px-1 py-0.5 text-right font-medium" title="Pearson correlation of estimate and truth">r</th>
                  <th className="px-1 py-0.5 text-right font-medium" title="Root mean squared error, in fraction units">RMSE</th>
                  <th className="px-1 py-0.5 text-right font-medium" title="Mean of estimate minus truth: > 0 over-estimates the type">Bias</th>
                </tr>
              </thead>
              <tbody>
                {perType.map((p) => (
                  <tr key={p.cell_type} className="border-b border-slate-100 dark:border-slate-800">
                    <td className="px-1 py-0.5 text-slate-700 dark:text-slate-200">{p.cell_type}</td>
                    <td className="px-1 py-0.5 text-right tabular-nums">{num(p.r)}</td>
                    <td className="px-1 py-0.5 text-right tabular-nums">{num(p.rmse, 3)}</td>
                    <td className="px-1 py-0.5 text-right tabular-nums">{num(p.bias, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ height: 220 }}>
            <Plot
              data={traces}
              layout={{
                autosize: true,
                margin: { l: 40, r: 6, t: 6, b: 34 },
                showlegend: false,
                xaxis: { title: { text: "true fraction (%)", font: { size: 10 } }, range: [0, 100], tickfont: { size: 9 } },
                yaxis: { title: { text: "estimated (%)", font: { size: 10 } }, range: [0, 100], tickfont: { size: 9 } },
              }}
              config={plotConfig("deconvolution_self_check")}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </div>
          <Caption
            n={[
              { value: check.donors.length, unit: "donors held out" },
              { value: check.n_mixtures, unit: "mixtures each" },
            ]}
          >
            Reference {prettyObsLabel(check.celltype_column)} by {prettyObsLabel(check.donor_column)}. Truth against
            estimate for {check.method_names[shownMethod] ?? shownMethod}, each point one cell type in one mixture or
            held-out donor
            {check.points_capped ? ` (a sample of the ${(check.n_points[shownMethod] ?? 0).toLocaleString()} points)` : ""}. Points on
            the dotted line are exact.
          </Caption>
          {check.notes.map((n) => (
            <p key={n} className="text-[11px] text-slate-500">
              {n}
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
