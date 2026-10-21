import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Search } from "lucide-react";
import type { PlotMouseEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { InfoTip } from "@/components/common/InfoTip";
import { CapNote } from "@/components/common/Caption";
import { useDatasetStore } from "@/stores/datasetStore";
import {
  getActivityCells,
  getActivitySummary,
  NETWORK_LABELS,
  type ActivityNetwork,
  type ActivityRef,
  type ActivitySummary,
} from "@/api/activity";
import { activityLabel, filterRows, heatmapData, sourceRows } from "@/lib/activity";
import { plotConfig } from "@/lib/plotConfig";

interface Props {
  /** Colour the embedding by one source's per-cell activity. */
  onScore: (scores: Float32Array, name: string, genes?: string[], activity?: ActivityRef) => void;
  groupByColumn: string;
  setGroupByColumn: (col: string) => void;
}

const TOP_CHOICES = [10, 20, 40] as const;
// Diverging, white at 0: red = active (targets up), blue = repressed. Plotly's own
// "RdBu" runs blue to red with a grey middle.
const ACTIVITY_SCALE: [number, string][] = [
  [0, "#2166ac"],
  [0.5, "#f7f7f7"],
  [1, "#b2182b"],
];
const LIST_LIMIT = 200;

/**
 * Per-cell TF (CollecTRI) and pathway (PROGENy) activity: a group x source heatmap
 * of the mean, and any one source on the embedding. Scored on the fly by the
 * backend (ULM t-values on log-normalised expression); the first use of a network
 * downloads it.
 */
export function UnifiedActivitySubtab({ onScore, groupByColumn, setGroupByColumn }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const [network, setNetwork] = useState<ActivityNetwork>("progeny");
  const [summary, setSummary] = useState<ActivitySummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [topN, setTopN] = useState<number>(20);
  const [colouring, setColouring] = useState<string | null>(null);
  const [shown, setShown] = useState<string | null>(null);

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 1 &&
        c.n_unique <= 100 &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset]);
  const groupby = groupColumns.some((c) => c.name === groupByColumn) ? groupByColumn : "";

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    getActivitySummary(datasetId, network, groupby || undefined)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch((e) => {
        if (!cancelled) {
          setSummary(null);
          setError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, network, groupby]);

  const colourBy = useCallback(
    async (source: string) => {
      if (!datasetId || colouring) return;
      setColouring(source);
      try {
        const res = await getActivityCells(datasetId, network, source);
        onScore(new Float32Array(res.scores), activityLabel(network, source), undefined, { network, source });
        setShown(source);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setColouring(null);
      }
    },
    [datasetId, network, colouring, onScore],
  );

  const rows = useMemo(() => (summary ? sourceRows(summary) : []), [summary]);
  const matching = useMemo(() => filterRows(rows, query), [rows, query]);
  const listed = useMemo(() => matching.slice(0, LIST_LIMIT), [matching]);
  const heat = useMemo(
    () => (summary ? heatmapData(summary, network === "progeny" ? summary.sources.length : topN) : null),
    [summary, network, topN],
  );

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        {(Object.keys(NETWORK_LABELS) as ActivityNetwork[]).map((n) => (
          <button
            key={n}
            onClick={() => {
              setNetwork(n);
              setQuery("");
            }}
            className={`rounded-md px-2.5 py-1 text-xs font-medium ${
              network === n ? "bg-primary text-white" : "border border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {NETWORK_LABELS[n].long}
          </button>
        ))}
        <InfoTip label="About TF and pathway activity" align="right">
          Each cell gets one score per transcription factor (CollecTRI) or pathway (PROGENy): the
          t-value of a linear fit of its log-normalised expression on the source&apos;s signed
          target weights (decoupler&apos;s ULM). Positive: the targets it activates are high and the
          ones it represses low. Human networks only.
        </InfoTip>
      </div>

      <div>
        <label className="mb-1 block text-[11px] font-medium text-slate-500">Group by (heatmap)</label>
        <select
          value={groupby}
          onChange={(e) => setGroupByColumn(e.target.value)}
          className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700"
        >
          <option value="">None</option>
          {groupColumns.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name} ({c.n_unique} groups)
            </option>
          ))}
        </select>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          Scoring {NETWORK_LABELS[network].short.toLowerCase()}… (the first use downloads the network)
        </div>
      )}
      {error && (
        <p role="alert" className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {error}
        </p>
      )}

      {summary && !loading && (
        <>
          {heat && (
            <div>
              <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
                <span>Mean activity per group{network === "collectri" ? ", most variable TFs" : ""}</span>
                {network === "collectri" && (
                  <select
                    aria-label="TFs shown"
                    value={topN}
                    onChange={(e) => setTopN(Number(e.target.value))}
                    className="rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]"
                  >
                    {TOP_CHOICES.map((n) => (
                      <option key={n} value={n}>
                        top {n}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              <div style={{ height: Math.max(220, heat.y.length * 15 + 110) }}>
                <Plot
                  data={[
                    {
                      type: "heatmap",
                      x: heat.x,
                      y: heat.y,
                      z: heat.z,
                      zmin: -heat.zmax, // symmetric: white is 0
                      zmax: heat.zmax,
                      colorscale: ACTIVITY_SCALE,
                      colorbar: { title: { text: "t", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
                      hovertemplate: "%{y} in %{x}<br>mean activity %{z:.2f}<extra></extra>",
                    },
                  ]}
                  layout={{
                    autosize: true,
                    margin: { l: 70, r: 6, t: 4, b: 90 },
                    // category: cluster ids "0".."13" would otherwise read as numbers
                    // and get every other label.
                    xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true },
                    yaxis: { type: "category", tickfont: { size: 9 }, automargin: true },
                  }}
                  config={plotConfig(`activity_${network}_heatmap`)}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                  onClick={(e: Readonly<PlotMouseEvent>) => {
                    const y = e.points[0]?.y;
                    if (typeof y === "string") colourBy(y);
                  }}
                />
              </div>
              <p className="text-[10px] text-slate-400">Click a row to colour the cells by it.</p>
            </div>
          )}

          <div>
            <div className="relative mb-1.5">
              <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={network === "collectri" ? "Find a TF, e.g. STAT1" : "Find a pathway"}
                className="w-full rounded border border-slate-300 py-1.5 pl-7 pr-2 text-xs"
              />
            </div>
            <CapNote
              shown={LIST_LIMIT}
              total={matching.length}
              by={summary.mean ? "spread of the group means" : "name"}
              rest="search to find the others"
              className="mb-1"
            />
            <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-slate-50">
                  <tr className="border-b border-slate-200 text-slate-600">
                    <th className="px-2 py-1.5 text-left font-medium">{network === "collectri" ? "TF" : "Pathway"}</th>
                    <th className="px-2 py-1.5 text-right font-medium" title="Targets among the measured genes">
                      Targets
                    </th>
                    {summary.mean && <th className="px-2 py-1.5 text-left font-medium">Highest in</th>}
                  </tr>
                </thead>
                <tbody>
                  {listed.map((r) => (
                    <tr
                      key={r.source}
                      onClick={() => colourBy(r.source)}
                      className={`cursor-pointer border-b border-slate-50 hover:bg-blue-50 ${
                        shown === r.source ? "bg-blue-50" : ""
                      }`}
                    >
                      <td className="px-2 py-1.5 font-mono font-medium text-slate-800">
                        {r.source}
                        {colouring === r.source && <Loader2 className="ml-1 inline h-3 w-3 animate-spin" />}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-500">{r.n_targets}</td>
                      {summary.mean && (
                        <td className="truncate px-2 py-1.5 text-slate-600" title={r.top_group ?? ""}>
                          {r.top_group} {r.top_mean !== null && <span className="text-slate-400">({r.top_mean.toFixed(1)})</span>}
                        </td>
                      )}
                    </tr>
                  ))}
                  {listed.length === 0 && (
                    <tr>
                      <td colSpan={3} className="px-2 py-4 text-center text-slate-400">
                        No source matches.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-[10px] leading-snug text-slate-400">
            {summary.sources.length} {network === "collectri" ? "TFs" : "pathways"} with at least 5 of
            their targets among {summary.n_genes_measured.toLocaleString()} measured genes; scored on the{" "}
            {summary.matrix_source} ({summary.elapsed_seconds}s). {summary.label}: {summary.citation}.
          </p>
        </>
      )}
    </div>
  );
}
