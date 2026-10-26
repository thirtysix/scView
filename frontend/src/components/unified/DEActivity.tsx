import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Download, Loader2 } from "lucide-react";
import type { DEResponse } from "@/api/de";
import {
  getActivityCells,
  getDEActivity,
  NETWORK_LABELS,
  type ActivityNetwork,
  type DEActivityResponse,
  type ActivityRef,
} from "@/api/activity";
import { InfoTip } from "@/components/common/InfoTip";
import { activityLabel, deProfile } from "@/lib/activity";
import { downloadCsv } from "@/lib/csv";
import { formatPct } from "@/lib/deDetection";
import { formatPValue } from "@/lib/formatting";
import { errorMessage } from "@/lib/errorMessage";

interface Props {
  datasetId: string;
  result: DEResponse;
  /** Display names of the two sides: a positive score = targets higher in `labelA`. */
  labelA: string;
  labelB: string;
  onScore: (scores: Float32Array, name: string, genes?: string[], activity?: ActivityRef) => void;
}

const SHOWN = 50;

/**
 * Which TFs (CollecTRI) and pathways (PROGENy) a DE comparison moves: ULM on the
 * DE statistic of every tested gene, not just the significant ones. A row click
 * colours the cells by that source's per-cell activity.
 */
export function DEActivity({ datasetId, result, labelA, labelB, onScore }: Props) {
  const [network, setNetwork] = useState<ActivityNetwork>("collectri");
  const [data, setData] = useState<Partial<Record<ActivityNetwork, DEActivityResponse>>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [colouring, setColouring] = useState<string | null>(null);

  const profile = useMemo(() => deProfile(result.genes), [result]);

  // A new DE result makes the old activity meaningless.
  useEffect(() => {
    setData({});
    setError(null);
  }, [result]);

  const run = useCallback(
    async (n: ActivityNetwork) => {
      setLoading(true);
      setError(null);
      try {
        const res = await getDEActivity(datasetId, n, profile.genes, profile.stats, result.result_id);
        setData((d) => ({ ...d, [n]: res }));
      } catch (e) {
        setError(errorMessage(e));
      } finally {
        setLoading(false);
      }
    },
    [datasetId, profile, result.result_id],
  );

  const pick = (n: ActivityNetwork) => {
    setNetwork(n);
    if (!data[n] && Object.keys(data).length > 0) run(n);
  };

  const colourBy = async (source: string) => {
    if (colouring) return;
    setColouring(source);
    try {
      const res = await getActivityCells(datasetId, network, source);
      onScore(new Float32Array(res.scores), activityLabel(network, source), undefined, { network, source });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setColouring(null);
    }
  };

  const shown = data[network];
  const exportCsv = () => {
    if (!shown) return;
    downloadCsv(
      `de_activity_${network}_${result.label}_${datasetId}.csv`,
      ["source", "activity_t", "p_value", "adj_pvalue", "n_targets"],
      shown.results.map((r) => [r.source, r.score.toFixed(4), r.pvalue.toExponential(4), r.padj.toExponential(4), String(r.n_targets)]),
    );
  };
  const noStat = profile.genes.length === 0;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-200 p-2.5 text-xs">
      <div className="flex items-center gap-2">
        <Activity className="h-3.5 w-3.5 flex-shrink-0 text-slate-400" />
        <span className="font-semibold text-slate-700">TF / pathway activity</span>
        <InfoTip width={300}>
          decoupler&apos;s univariate linear model on this comparison&apos;s test statistic for all{" "}
          {profile.genes.length.toLocaleString()} tested genes: a positive score means the
          source&apos;s activated targets are higher in {labelA} and its repressed targets lower.
          Benjamini-Hochberg adjusted across sources. Human networks only.
        </InfoTip>
        <span className="flex-1" />
        <button
          onClick={() => run(network)}
          disabled={loading || noStat}
          title={noStat ? "This DE result carries no test statistic (re-run the DE)" : "Score the DE statistic"}
          className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
        >
          {loading && <Loader2 className="h-3 w-3 animate-spin" />}
          Score
        </button>
      </div>

      <div className="flex rounded-md border border-slate-200 p-0.5">
        {(Object.keys(NETWORK_LABELS) as ActivityNetwork[]).map((n) => (
          <button
            key={n}
            onClick={() => pick(n)}
            className={`flex-1 rounded px-2 py-1 transition ${
              network === n ? "bg-primary font-medium text-white" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {NETWORK_LABELS[n].long}
          </button>
        ))}
      </div>

      {error && (
        <p role="alert" className="rounded border border-amber-200 bg-amber-50 p-2 text-amber-800">
          {error}
        </p>
      )}

      {shown && (
        <>
          <div className="flex items-center justify-between text-[10px] text-slate-500">
            <span>
              {shown.results.filter((r) => r.padj < 0.05).length} of {shown.results.length} with adj. p &lt; 0.05;
              red = targets up in {labelA}, blue = up in {labelB}.
            </span>
            <button
              onClick={exportCsv}
              title="Export as CSV"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-0.5 text-slate-600 hover:bg-slate-50"
            >
              <Download className="h-3 w-3" />
              CSV
            </button>
          </div>
          <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full">
              <thead className="sticky top-0 z-10 bg-slate-50">
                <tr className="border-b border-slate-200 text-slate-600">
                  <th className="px-2 py-1.5 text-left font-medium" title={shown.kind === "tf" ? "Transcription factor (CollecTRI)" : "Pathway (PROGENy)"}>
                    {shown.kind === "tf" ? "TF" : "Pathway"}
                  </th>
                  <th
                    className="px-2 py-1.5 text-right font-medium"
                    title="ULM activity on the DE statistics: the t-value of the slope of each gene's DE statistic on its weight in the source's network (decoupler's ulm); positive when the activated targets are up"
                  >
                    Activity
                  </th>
                  <th className="px-2 py-1.5 text-right font-medium" title="Two-sided p of the ULM slope, Benjamini-Hochberg adjusted over the sources scored">
                    Adj.P
                  </th>
                  <th className="px-2 py-1.5 text-right font-medium" title="Targets among the tested genes (at least 5 to be scored)">
                    Targets
                  </th>
                </tr>
              </thead>
              <tbody>
                {shown.results.slice(0, SHOWN).map((r) => (
                  <tr
                    key={r.source}
                    onClick={() => colourBy(r.source)}
                    title="Colour the cells by this source's activity"
                    className="cursor-pointer border-b border-slate-50 hover:bg-blue-50"
                  >
                    <td className="px-2 py-1.5 font-mono font-medium text-slate-800">
                      {r.source}
                      {colouring === r.source && <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />}
                    </td>
                    <td className={`px-2 py-1.5 text-right tabular-nums ${r.score > 0 ? "text-red-600" : "text-blue-600"}`}>
                      {r.score.toFixed(2)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">{formatPValue(r.padj)}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{r.n_targets}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-slate-400">
            Strongest {Math.min(SHOWN, shown.results.length)} by |activity|; the CSV has all. {shown.label}:{" "}
            {shown.citation}. Higher in {labelB} means a negative score.
            {shown.de_min_pct
              ? ` Scored on the ${shown.n_genes.toLocaleString()} genes the DE tested (detected in at least ` +
                `${formatPct(shown.de_min_pct)} of either group); targets it left out are not used.`
              : ""}
          </p>
        </>
      )}
    </div>
  );
}
