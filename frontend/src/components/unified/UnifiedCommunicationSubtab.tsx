import { useEffect, useMemo, useRef, useState } from "react";
import { Download, Loader2, Play, Search, Square } from "lucide-react";
import type { Data, PlotMouseEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { InfoTip } from "@/components/common/InfoTip";
import {
  communicationCsvUrl,
  getCommunicationOptions,
  runCommunication,
  type CommunicationOptions,
  type CommunicationStart,
} from "@/api/communication";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { rgbCss } from "@/lib/categoryColors";
import {
  circleLayout,
  dotPlotData,
  filterRows,
  lrLabel,
  pairMatrix,
  roleBars,
  sortRows,
  toRows,
  type SortKey,
  type Weight,
} from "@/lib/communication";
import { plotConfig } from "@/lib/plotConfig";
import { useCommunicationStore } from "@/stores/communicationStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { RequestError } from "@/components/common/RequestError";

interface Props {
  /** The Unified View's group-by column: the default sender / receiver groups */
  groupByColumn: string;
}

type View = "dot" | "circle" | "heatmap" | "roles";
const VIEWS: { id: View; label: string }[] = [
  { id: "dot", label: "Dot plot" },
  { id: "circle", label: "Circle" },
  { id: "heatmap", label: "Heatmap" },
  { id: "roles", label: "Roles" },
];
const TABLE_LIMIT = 200;
/** Circle plot: edges under this share of the strongest are left out (a hairball otherwise). */
const CIRCLE_MIN_SHARE = 0.1;
const COLUMNS: { key: SortKey; label: string; title: string; num?: boolean }[] = [
  { key: "source", label: "Sender", title: "Source group (expresses the ligand)" },
  { key: "target", label: "Receiver", title: "Target group (expresses the receptor)" },
  { key: "ligand_complex", label: "Ligand", title: "Ligand (a complex as A_B)" },
  { key: "receptor_complex", label: "Receptor", title: "Receptor (a complex as A_B)" },
  { key: "magnitude_rank", label: "Magn.", title: "Magnitude rank (RRA of the expression-strength scores); lower is stronger", num: true },
  { key: "specificity_rank", label: "Spec.", title: "Specificity rank (RRA of the group-specificity scores); lower is more specific", num: true },
  { key: "lr_means", label: "LR mean", title: "Mean of the ligand's mean (sender) and the receptor's mean (receiver)", num: true },
  { key: "cellphone_pvals", label: "p", title: "CellPhoneDB permutation p-value", num: true },
];

const fmt = (v: number | null, digits = 2) =>
  v === null ? "–" : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(1) : v.toFixed(digits);

/**
 * Ligand-receptor communication between the groups of one column (LIANA's
 * consensus rank aggregate, computed by the backend): the strongest interactions
 * as a table and a dot plot, interactions between groups as a circle plot or
 * heatmap, and each group's role as sender and receiver.
 */
export function UnifiedCommunicationSubtab({ groupByColumn }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const settings = useCommunicationStore();
  const { update, setResult } = settings;
  const stored = settings.result;
  const result = stored && stored.datasetId === datasetId ? stored.result : null;

  const [options, setOptions] = useState<CommunicationOptions | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [start, setStart] = useState<CommunicationStart | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [view, setView] = useState<View>("dot");
  const [weight, setWeight] = useState<Weight>("count");
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "magnitude_rank", asc: true });

  const groupColumns = useMemo(() => {
    if (!dataset) return [];
    const nCells = dataset.n_cells ?? 0;
    return dataset.obs_columns.filter(
      (c) =>
        (c.dtype === "category" || c.dtype === "object" || c.dtype === "bool") &&
        c.n_unique >= 2 &&
        c.n_unique <= (options?.max_groups ?? 60) &&
        !(nCells > 0 && c.n_unique / nCells >= 0.9),
    );
  }, [dataset, options]);
  const wanted = settings.groupby || groupByColumn;
  const groupby = groupColumns.some((c) => c.name === wanted) ? wanted : "";
  const conditions = options?.conditions ?? [];
  const subsetOk = conditions.some(
    (c) => c.column === settings.subsetColumn && c.values.includes(settings.subsetValue),
  );
  const subsetColumn = subsetOk ? settings.subsetColumn : "";
  const subsetValue = subsetOk ? settings.subsetValue : "";

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    getCommunicationOptions(datasetId, {
      groupby: groupby || undefined,
      subset_column: subsetColumn || undefined,
      subset_value: subsetValue || undefined,
    })
      .then((o) => {
        if (!cancelled) setOptions(o);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, groupby, subsetColumn, subsetValue]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = async () => {
    if (!datasetId || !groupby || running) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    setStart(null);
    setProgress(null);
    try {
      const res = await runCommunication(
        datasetId,
        {
          groupby,
          subset_column: subsetColumn || null,
          subset_value: subsetValue || null,
          expr_prop: settings.exprProp,
        },
        { onStart: setStart, onProgress: (done, total) => setProgress({ done, total }), signal: ctrl.signal },
      );
      setResult(datasetId, res);
      setSource("");
      setTarget("");
      setQuery("");
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const colorOf = useCategoryColors(result?.groupby);
  const rows = useMemo(() => (result ? toRows(result) : []), [result]);
  const filtered = useMemo(() => filterRows(rows, { source, target, query }), [rows, source, target, query]);
  const sorted = useMemo(() => sortRows(filtered, sort.key, sort.asc), [filtered, sort]);
  const dot = useMemo(() => dotPlotData(filtered), [filtered]);

  if (!dataset || !datasetId) return <div className="text-xs text-slate-400">No dataset loaded.</div>;

  const pct = progress ? Math.round((100 * progress.done) / Math.max(1, progress.total)) : 0;
  const sel = "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600">Ligand-receptor communication</span>
        <InfoTip label="About cell-cell communication" align="right">
          For every sender group, receiver group and ligand-receptor pair in LIANA&apos;s consensus
          resource (human, or mouse for a mouse dataset), five methods score the ligand&apos;s
          expression in the sender and the receptor&apos;s in the receiver (CellPhoneDB, Connectome,
          log2FC, NATMI, SingleCellSignalR) on log-normalised values. Their ranks are combined (robust
          rank aggregation, as LIANA&apos;s rank_aggregate): magnitude rank = how strongly both are
          expressed, specificity rank = how specific to these two groups. Lower is stronger. Both
          genes must be detected in at least the set share of each group&apos;s cells; a complex
          counts by its weakest subunit. These are predictions from expression, not measured
          signalling. To compare conditions, run each condition on its own.
        </InfoTip>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Senders and receivers</label>
          <select value={groupby} onChange={(e) => update({ groupby: e.target.value })} className={sel}>
            <option value="">Choose a column</option>
            {groupColumns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.n_unique})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-[11px] font-medium text-slate-500">Cells</label>
          <select
            value={subsetColumn ? `${subsetColumn}\u0000${subsetValue}` : ""}
            onChange={(e) => {
              const [col, val] = e.target.value ? e.target.value.split("\u0000") : ["", ""];
              update({ subsetColumn: col ?? "", subsetValue: val ?? "" });
            }}
            className={sel}
          >
            <option value="">All cells</option>
            {conditions.flatMap((c) =>
              c.values.map((v) => (
                <option key={`${c.column}=${v}`} value={`${c.column}\u0000${v}`}>
                  {c.column} = {v}
                </option>
              )),
            )}
          </select>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-[11px] text-slate-500" htmlFor="ccc-expr-prop">
          Detected in at least
        </label>
        <select
          id="ccc-expr-prop"
          value={settings.exprProp}
          onChange={(e) => update({ exprProp: Number(e.target.value) })}
          className="rounded border border-slate-300 bg-white px-1 py-1 text-xs"
        >
          {[0.05, 0.1, 0.2, 0.3].map((v) => (
            <option key={v} value={v}>
              {Math.round(v * 100)}%
            </option>
          ))}
        </select>
        <span className="text-[11px] text-slate-500">of a group&apos;s cells</span>
        <div className="ml-auto">
          {running ? (
            <button
              onClick={() => abortRef.current?.abort()}
              className="flex items-center gap-1 rounded-md border border-slate-300 px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-50"
            >
              <Square className="h-3 w-3" /> Stop
            </button>
          ) : (
            <button
              onClick={run}
              disabled={!groupby}
              className="flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
            >
              <Play className="h-3 w-3" /> Run
            </button>
          )}
        </div>
      </div>
      {options && groupby && options.estimate_seconds != null && !running && (
        <p className="text-[10px] text-slate-400">
          {options.n_cells?.toLocaleString()} cells in {options.n_groups} groups, about{" "}
          {Math.max(1, Math.round(options.estimate_seconds))} s with {options.default_n_perms} permutations;{" "}
          {options.resource === "mouseconsensus" ? "mouse" : "human"} resource ({options.species_evidence}). The
          first run downloads the resource (2 MB).
        </p>
      )}
      {running && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {start
              ? `Scoring ${start.n_groups} groups, ${start.resource_label}${progress ? `: permutation ${progress.done} of ${progress.total}` : ""}`
              : "Preparing… (the first run downloads the resource)"}
          </div>
          <div className="h-1.5 w-full rounded bg-slate-100">
            <div className="h-1.5 rounded bg-primary transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      <RequestError message={error} />

      {result && (
        <>
          <p className="text-[11px] leading-snug text-slate-500">
            {result.n_rows.toLocaleString()} interactions scored between {result.groups.length}{" "}
            {result.groupby} groups ({result.subset ? `${result.subset.column} = ${result.subset.value}` : "all cells"},{" "}
            {result.n_cells.toLocaleString()} cells); {result.n_specific.toLocaleString()} with specificity rank ≤{" "}
            {result.specific_threshold}.
            {Object.keys(result.dropped_groups).length > 0 && (
              <> Left out (under 5 cells): {Object.keys(result.dropped_groups).join(", ")}.</>
            )}
          </p>

          <div className="grid grid-cols-3 gap-1.5">
            <select aria-label="Sender" value={source} onChange={(e) => setSource(e.target.value)} className={sel}>
              <option value="">Any sender</option>
              {result.groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <select aria-label="Receiver" value={target} onChange={(e) => setTarget(e.target.value)} className={sel}>
              <option value="">Any receiver</option>
              {result.groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <div className="relative">
              <Search className="absolute left-1.5 top-1/2 h-3 w-3 -translate-y-1/2 text-slate-400" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ligand / receptor"
                className="w-full rounded border border-slate-300 py-1.5 pl-5 pr-1 text-xs"
              />
            </div>
          </div>

          <div className="flex items-center gap-1">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                onClick={() => setView(v.id)}
                className={`rounded-md px-2 py-1 text-[11px] font-medium ${
                  view === v.id ? "bg-primary text-white" : "border border-slate-200 text-slate-600 hover:bg-slate-50"
                }`}
              >
                {v.label}
              </button>
            ))}
            {view !== "dot" && (
              <select
                aria-label="Weight"
                value={weight}
                onChange={(e) => setWeight(e.target.value as Weight)}
                className="ml-auto rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]"
              >
                <option value="count">Count</option>
                <option value="strength">Strength</option>
              </select>
            )}
          </div>

          {view === "dot" && <DotPlotView dot={dot} />}
          {view === "circle" && (
            <CirclePlot
              groups={result.groups}
              matrix={pairMatrix(result, weight)}
              weight={weight}
              color={(g) => rgbCss(colorOf(g))}
              onPick={(g) => setSource((s) => (s === g ? "" : g))}
            />
          )}
          {view === "heatmap" && (
            <PairHeatmap groups={result.groups} matrix={pairMatrix(result, weight)} weight={weight} />
          )}
          {view === "roles" && (
            <RolesPlot bars={roleBars(result, weight)} weight={weight} />
          )}
          {view !== "dot" && (
            <p className="text-[10px] text-slate-400">
              {weight === "count" ? "Interactions" : "Summed ligand-receptor mean of the interactions"} with
              specificity rank ≤ {result.specific_threshold}, from the full result.
              {view === "circle" &&
                ` Edges under ${CIRCLE_MIN_SHARE * 100}% of the strongest and autocrine ones (in the node's hover) are not drawn; click a group to filter the table by it as sender.`}
            </p>
          )}

          <div>
            <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
              <span>
                {filtered.length > TABLE_LIMIT
                  ? `Top ${TABLE_LIMIT} of ${filtered.length.toLocaleString()} rows by ${COLUMNS.find((c) => c.key === sort.key)?.label ?? sort.key}`
                  : `${filtered.length.toLocaleString()} rows`}
                {result.rows_shown < result.n_rows &&
                  ` (from the ${result.rows_shown.toLocaleString()} strongest of ${result.n_rows.toLocaleString()}; the CSV has all)`}
              </span>
              <a
                href={communicationCsvUrl(datasetId, result.result_id)}
                download
                className="flex items-center gap-1 text-primary hover:underline"
                title="Every scored row, with each method's score"
              >
                <Download className="h-3 w-3" /> CSV
              </a>
            </div>
            <div className="max-h-80 overflow-auto rounded-lg border border-slate-200">
              <table className="w-full text-[11px]">
                <thead className="sticky top-0 bg-slate-50">
                  <tr className="border-b border-slate-200 text-slate-600">
                    {COLUMNS.map((c) => (
                      <th
                        key={c.key}
                        title={c.title}
                        onClick={() =>
                          setSort((s) => (s.key === c.key ? { key: c.key, asc: !s.asc } : { key: c.key, asc: !!c.num && c.key !== "lr_means" }))
                        }
                        className={`cursor-pointer select-none px-1.5 py-1.5 font-medium ${c.num ? "text-right" : "text-left"}`}
                      >
                        {c.label}
                        {sort.key === c.key ? (sort.asc ? " ▲" : " ▼") : ""}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.slice(0, TABLE_LIMIT).map((r, i) => (
                    <tr key={i} className="border-b border-slate-50 hover:bg-blue-50" title={lrLabel(r)}>
                      <td className="max-w-[5rem] truncate px-1.5 py-1 text-slate-700">{r.source}</td>
                      <td className="max-w-[5rem] truncate px-1.5 py-1 text-slate-700">{r.target}</td>
                      <td className="max-w-[6rem] truncate px-1.5 py-1 font-mono text-slate-800">{r.ligand_complex}</td>
                      <td className="max-w-[6rem] truncate px-1.5 py-1 font-mono text-slate-800">{r.receptor_complex}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.magnitude_rank)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.specificity_rank)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.lr_means)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.cellphone_pvals, 3)}</td>
                    </tr>
                  ))}
                  {sorted.length === 0 && (
                    <tr>
                      <td colSpan={COLUMNS.length} className="px-2 py-4 text-center text-slate-400">
                        No interaction matches.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-[10px] leading-snug text-slate-400">
            {result.resource_label}, {result.n_interactions.toLocaleString()} of {result.n_resource.toLocaleString()}{" "}
            ligand-receptor pairs expressed; {result.n_perms} permutations (seed {result.seed}); scored on the{" "}
            {result.matrix_source} ({result.elapsed_seconds}s). {result.citation}.
          </p>
        </>
      )}
    </div>
  );
}

function DotPlotView({ dot }: { dot: ReturnType<typeof dotPlotData> }) {
  if (dot.points.length === 0) return <p className="text-xs text-slate-400">No interaction to plot.</p>;
  const scale = dot.maxSize > 0 ? 14 / dot.maxSize : 1;
  const trace: Data = {
    type: "scatter",
    mode: "markers",
    x: dot.points.map((p) => p.x),
    y: dot.points.map((p) => p.y),
    marker: {
      size: dot.points.map((p) => 4 + p.size * scale),
      color: dot.points.map((p) => p.color),
      colorscale: "Viridis",
      showscale: true,
      colorbar: { title: { text: "−log10 magn.", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
      line: { width: 0 },
    },
    customdata: dot.points.map((p) => [
      p.row.specificity_rank ?? NaN,
      p.row.magnitude_rank ?? NaN,
      p.row.lr_means ?? NaN,
    ]),
    hovertemplate:
      "%{x}<br>%{y}<br>magnitude rank %{customdata[1]:.2g}<br>specificity rank %{customdata[0]:.2g}" +
      "<br>LR mean %{customdata[2]:.2f}<extra></extra>",
  };
  return (
    <div>
      <div style={{ height: Math.max(240, dot.y.length * 17 + 130) }}>
        <Plot
          data={[trace]}
          layout={{
            autosize: true,
            margin: { l: 10, r: 6, t: 8, b: 10 },
            xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true, categoryarray: dot.x },
            yaxis: { type: "category", tickfont: { size: 9 }, automargin: true, categoryarray: dot.y },
          }}
          config={plotConfig("communication_dotplot")}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>
      <p className="text-[10px] text-slate-400">
        Size: −log10 specificity rank; colour: −log10 magnitude rank. The 15 strongest ligand-receptor pairs of the
        filtered rows.
      </p>
    </div>
  );
}

function CirclePlot({
  groups,
  matrix,
  weight,
  color,
  onPick,
}: {
  groups: string[];
  matrix: number[][];
  weight: Weight;
  color: (g: string) => string;
  onPick: (g: string) => void;
}) {
  const layout = useMemo(() => circleLayout(groups, matrix, 8, CIRCLE_MIN_SHARE), [groups, matrix]);
  const unit = weight === "count" ? "interactions" : "summed LR mean";
  const traces: Data[] = layout.edges.map((e) => ({
    type: "scatter",
    mode: "lines",
    x: e.x,
    y: e.y,
    line: { width: e.width, color: color(groups[e.source]!), shape: "spline" },
    opacity: 0.55,
    hoverinfo: "text",
    text: e.x.map(() => `${groups[e.source]} → ${groups[e.target]}: ${fmt(e.value, weight === "count" ? 0 : 1)} ${unit}`),
    showlegend: false,
  }));
  traces.push({
    type: "scatter",
    mode: "text+markers",
    x: layout.nodes.map((n) => n.x),
    y: layout.nodes.map((n) => n.y),
    text: layout.nodes.map((n) => n.group),
    // Plotly takes one position per point; the type allows only one for all.
    textposition: layout.nodes.map((n) => (n.y >= 0 ? "top center" : "bottom center")) as unknown as "top center",
    textfont: { size: 9 },
    marker: { size: 12, color: layout.nodes.map((n) => color(n.group)), line: { width: 1, color: "#fff" } },
    customdata: layout.nodes.map((n) => [n.out, n.in, n.autocrine]),
    hovertemplate: `%{text}<br>out %{customdata[0]:.3~g}, in %{customdata[1]:.3~g}, autocrine %{customdata[2]:.3~g} (${unit})<extra></extra>`,
    showlegend: false,
  });
  return (
    <div style={{ height: 420 }}>
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 4, r: 4, t: 4, b: 4 },
          xaxis: { visible: false, range: [-1.35, 1.35], fixedrange: true },
          yaxis: { visible: false, range: [-1.25, 1.25], scaleanchor: "x", fixedrange: true },
          hovermode: "closest",
        }}
        config={plotConfig("communication_circle")}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
        onClick={(e: Readonly<PlotMouseEvent>) => {
          const p = e.points[0];
          if (p && p.curveNumber === traces.length - 1) onPick(groups[p.pointIndex]!);
        }}
      />
    </div>
  );
}

function PairHeatmap({ groups, matrix, weight }: { groups: string[]; matrix: number[][]; weight: Weight }) {
  return (
    <div style={{ height: Math.max(260, groups.length * 18 + 130) }}>
      <Plot
        data={[
          {
            type: "heatmap",
            x: groups,
            y: groups,
            z: matrix,
            colorscale: "Viridis",
            colorbar: { title: { text: weight === "count" ? "n" : "sum", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
            hovertemplate: "%{y} → %{x}<br>%{z:.3~g}<extra></extra>",
          },
        ]}
        layout={{
          autosize: true,
          margin: { l: 10, r: 6, t: 8, b: 10 },
          xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true, title: { text: "Receiver", font: { size: 10 } } },
          yaxis: { type: "category", tickfont: { size: 9 }, automargin: true, autorange: "reversed", title: { text: "Sender", font: { size: 10 } } },
        }}
        config={plotConfig("communication_heatmap")}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

function RolesPlot({ bars, weight }: { bars: { group: string; out: number; in: number }[]; weight: Weight }) {
  const groups = bars.map((b) => b.group);
  return (
    <div style={{ height: Math.max(240, bars.length * 20 + 110) }}>
      <Plot
        data={[
          { type: "bar", orientation: "h", name: "Outgoing (sender)", y: groups, x: bars.map((b) => b.out), marker: { color: "#d95f02" } },
          { type: "bar", orientation: "h", name: "Incoming (receiver)", y: groups, x: bars.map((b) => b.in), marker: { color: "#1b9e77" } },
        ]}
        layout={{
          autosize: true,
          barmode: "group",
          margin: { l: 10, r: 6, t: 28, b: 30 },
          legend: { orientation: "h", x: 0, y: 1.08, font: { size: 9 } },
          xaxis: { tickfont: { size: 9 }, title: { text: weight === "count" ? "Interactions" : "Summed LR mean", font: { size: 10 } } },
          yaxis: { type: "category", tickfont: { size: 9 }, automargin: true, autorange: "reversed" },
        }}
        config={plotConfig("communication_roles")}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
