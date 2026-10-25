import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { GitCompare, Loader2 } from "lucide-react";
import sankey from "plotly.js/lib/sankey";
import Plot from "@/components/plots/Plot";
import {
  getCrosscheckInfo,
  runCrosscheck,
  saveCrosscheck,
  type CrosscheckInfo,
  type CrosscheckResult,
  type CrosscheckSaved,
} from "@/api/annotation";
import { MarkerDbCheck } from "@/components/annotation/MarkerDbCheck";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import {
  agreementSummary,
  callsText,
  crosscheckColumns,
  defaultReference,
  defaultSelection,
  diagonalOrder,
  MAX_CROSSCHECK_COLUMNS,
  pairMatrix,
  sankeyData,
  sortLabelTable,
  tableMatrix,
  validCrosscheckName,
  type LabelSortKey,
} from "@/lib/annotationCrosscheck";
import { categoryColor, rgbCss } from "@/lib/categoryColors";
import { plotlyColorscale } from "@/lib/colors";
import { prettyObsLabel } from "@/lib/formatting";
import Plotly from "@/lib/plotly";
import { plotConfig } from "@/lib/plotConfig";
import { pct } from "@/lib/qcThresholds";
import { useDatasetStore } from "@/stores/datasetStore";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

// The Sankey is drawn only here, so its trace type is registered here rather
// than in the shared bundle (lib/plotly.ts); registering twice is harmless.
Plotly.register([sankey]);

const ROWS = 10; // more behind "show all"
const MAX_LABELLED = 15; // cell values are written in a heatmap up to this many labels a side
const NUM = "whitespace-nowrap pl-2 text-right align-top";

/** The server's reason from an apiFetch error ("API error 400: {detail}"). */
function errorText(e: unknown): string {
  const body = (e instanceof Error ? e.message : String(e)).replace(/^API error \d+: /, "");
  try {
    return String(JSON.parse(body).detail ?? body);
  } catch {
    return body;
  }
}

/**
 * Do the annotation methods agree? (C005) Pick 2–6 label columns (CellTypist,
 * LLM, marker scores, a clustering, hand labels) and a reference: pairwise
 * ARI/AMI, the reference against one other column as a row-normalised
 * confusion heatmap and a Sankey, per-label agreement after mapping each
 * column's labels onto the reference's, and the consensus and agreement saved
 * as columns to colour the embedding by.
 */
export function AnnotationCrosscheckCard({
  datasetId,
  preset,
}: {
  datasetId: string;
  /** Columns to open on (reference first), e.g. from the reference mapping; a new key re-applies them */
  preset?: { columns: string[]; key: number } | null;
}) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const available = useMemo(() => crosscheckColumns(dataset), [dataset]);
  const palette = useSettingsStore((s) => s.categoricalPalette);
  const colormap = useSettingsStore((s) => s.continuousColormap);
  const [info, setInfo] = useState<CrosscheckInfo | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [reference, setReference] = useState("");
  const [result, setResult] = useState<CrosscheckResult | null>(null);
  const [pair, setPair] = useState("");
  const [metric, setMetric] = useState<"ari" | "ami">("ari");
  const [sort, setSort] = useState<{ key: LabelSortKey; desc: boolean }>({ key: "mean_agreement", desc: false });
  const [allRows, setAllRows] = useState(false);
  const [running, setRunning] = useState(false);
  const [name, setName] = useState("crosscheck");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<CrosscheckSaved | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refresh = useDatasetRefresh();
  const request = useRef(0); // answers for an older dataset or selection are dropped
  const touched = useRef(false); // the user changed the selection: stop defaulting it

  useEffect(() => {
    request.current++;
    touched.current = false;
    setInfo(null);
    setSelected([]);
    setReference("");
    setResult(null);
    setSaved(null);
    setError(null);
    let live = true;
    getCrosscheckInfo(datasetId)
      .then((i) => {
        if (!live || !i?.suggested) return; // an older backend without the route
        setInfo(i);
        setName(i.default_name);
      })
      .catch((e) => live && setError(errorText(e)));
    return () => {
      live = false;
    };
  }, [datasetId]);

  // Open on the columns annotation methods made; drop columns that went away.
  useEffect(() => {
    // Same columns: keep the array, so a dataset refresh (e.g. after saving) keeps the result.
    const names = new Set(available.map((c) => c.name));
    setSelected((s) => {
      const n =
        !touched.current && info ? defaultSelection(available, info, info.max_columns) : s.filter((c) => names.has(c));
      return n.length === s.length && n.every((c, i) => c === s[i]) ? s : n;
    });
  }, [available, info]);

  useEffect(() => {
    if (!selected.includes(reference)) setReference(defaultReference(selected, available, info));
  }, [selected, reference, available, info]);

  // A layout effect: it must run before anything can click Compare on the new
  // selection, or the answer to that click is dropped as an older one (CI race).
  useLayoutEffect(() => {
    request.current++;
    setResult(null);
    setSaved(null);
  }, [selected, reference]);

  const card = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!preset || preset.columns.length < 2) return;
    touched.current = true;
    setSelected(preset.columns);
    setReference(preset.columns[0] ?? "");
    card.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }, [preset]);

  const maxColumns = info?.max_columns ?? MAX_CROSSCHECK_COLUMNS;
  const toggle = (col: string) => {
    touched.current = true;
    setSelected((s) => (s.includes(col) ? s.filter((c) => c !== col) : s.length < maxColumns ? [...s, col] : s));
  };

  const run = async () => {
    const id = ++request.current;
    setRunning(true);
    setError(null);
    try {
      const r = await runCrosscheck(datasetId, selected, reference);
      if (id !== request.current) return;
      setResult(r);
      setAllRows(false);
      setPair((p) => (r.comparisons[p] ? p : (r.columns.find((c) => c !== r.reference) ?? "")));
    } catch (e) {
      if (id === request.current) setError(errorText(e));
    } finally {
      setRunning(false);
    }
  };

  const save = async () => {
    if (!result) return;
    setSaving(true);
    setError(null);
    try {
      const s = await saveCrosscheck(datasetId, result.columns, result.reference, name.trim());
      await refresh(datasetId, s.columns.consensus);
      useObsRevisionStore.getState().bump(datasetId, s.columns.agreement);
      setSaved(s);
      getCrosscheckInfo(datasetId).then((i) => i?.suggested && setInfo(i), () => undefined);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setSaving(false);
    }
  };

  const colourBy = (column: string) => {
    useSettingsStore.getState().setColorBy(column);
    useViewStore.getState().setPanel("unified");
  };

  const comparison = result && pair ? result.comparisons[pair] : undefined;
  const order = useMemo(() => (comparison ? diagonalOrder(comparison.contingency) : null), [comparison]);
  const colours = (column: string, labels: readonly string[]) => {
    const col = dataset?.obs_columns.find((c) => c.name === column);
    return labels.map((l) => categoryColor(datasetId, column, col?.values ?? [], l, palette, col?.color_slots));
  };

  const labelRows = useMemo(
    () => (result ? sortLabelTable(result.labels, result.reference_labels, sort.key, sort.desc) : []),
    [result, sort],
  );
  const shown = allRows ? labelRows : labelRows.slice(0, ROWS);
  const others = result ? result.columns.filter((c) => c !== result.reference) : [];

  const sortHeader = (key: LabelSortKey, label: string, className: string, title?: string) => (
    <th className={`${className} font-medium`} title={title}>
      <button
        type="button"
        className="hover:text-slate-600"
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "label" }))}
      >
        {label}
        {sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
      </button>
    </th>
  );

  return (
    <div ref={card} className="rounded-xl border border-slate-200 bg-white p-4" data-testid="annotation-crosscheck">
      <div className="mb-2 flex items-center gap-2">
        <GitCompare className="h-5 w-5 text-primary" />
        <h3 className="text-sm font-semibold text-slate-900">Annotation cross-check</h3>
      </div>
      <p className="mb-3 text-xs text-slate-500">
        Do the annotation methods agree? Compare 2–{maxColumns} label columns (CellTypist, the AI, marker scores, a
        clustering, hand labels) against a reference. Each column&apos;s labels are mapped to the reference label most
        of their cells carry, so different names for the same cells still agree.
      </p>

      {available.length < 2 ? (
        <p className="text-xs text-slate-500">
          Needs two label columns: annotate cell types into their own column, or cluster first.
        </p>
      ) : (
        <>
          <div className="mb-2 text-xs font-medium text-slate-600">
            Columns to compare <span className="font-normal text-slate-400">({selected.length} of up to {maxColumns})</span>
          </div>
          <div className="mb-3 flex flex-wrap gap-1.5" role="group" aria-label="Columns to compare">
            {available.map((c) => {
              const on = selected.includes(c.name);
              const method = info?.suggested[c.name];
              return (
                <button
                  key={c.name}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggle(c.name)}
                  disabled={!on && selected.length >= maxColumns}
                  className={`rounded-full border px-2.5 py-0.5 text-[11px] transition-colors disabled:opacity-40 ${
                    on ? "border-primary bg-primary text-white" : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  title={c.name}
                >
                  {prettyObsLabel(c.name)}
                  {method && <span className={on ? "text-white/80" : "text-slate-400"}> · {method}</span>}
                </button>
              );
            })}
          </div>
          <div className="flex flex-wrap items-end gap-3">
            <div>
              <label htmlFor="xc-reference" className="mb-1 block text-xs font-medium text-slate-600">
                Reference
              </label>
              <select
                id="xc-reference"
                value={reference}
                onChange={(e) => setReference(e.target.value)}
                disabled={selected.length === 0}
                className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
              >
                {selected.map((c) => (
                  <option key={c} value={c}>
                    {prettyObsLabel(c)}
                  </option>
                ))}
              </select>
            </div>
            <button
              type="button"
              onClick={() => void run()}
              disabled={running || selected.length < 2 || !reference}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitCompare className="h-4 w-4" />}
              Compare
            </button>
          </div>
        </>
      )}

      {result && (
        <div className="mt-3 space-y-3 border-t border-slate-100 pt-3" data-testid="crosscheck-result">
          <p className="text-xs text-slate-700" data-testid="crosscheck-summary">
            {agreementSummary(result)}
          </p>

          <div>
            <div className="mb-1 flex items-center gap-2 text-[11px] text-slate-500">
              <span className="font-medium text-slate-600">Between every pair</span>
              <span role="group" aria-label="Score" className="inline-flex overflow-hidden rounded border border-slate-200">
                {(["ari", "ami"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMetric(m)}
                    aria-pressed={metric === m}
                    className={`px-1.5 py-0.5 ${metric === m ? "bg-slate-600 text-white" : "bg-white text-slate-600 hover:bg-slate-50"}`}
                  >
                    {m.toUpperCase()}
                  </button>
                ))}
              </span>
              <span>1 = the same partition, 0 = no more alike than chance; names play no part.</span>
            </div>
            <Plot
              data={[
                {
                  type: "heatmap",
                  x: result.columns,
                  y: result.columns,
                  z: pairMatrix(result.pairwise[metric], result.columns),
                  zmin: 0,
                  zmax: 1,
                  colorscale: plotlyColorscale(colormap),
                  texttemplate: "%{z:.2f}",
                  textfont: { size: 10 },
                  hovertemplate: `%{y} vs %{x}<br>${metric.toUpperCase()} %{z:.3f}<extra></extra>`,
                  colorbar: { thickness: 8, len: 0.9 },
                  xgap: 1,
                  ygap: 1,
                },
              ]}
              layout={{
                height: 90 + 34 * result.columns.length,
                margin: { l: 10, r: 10, t: 28, b: 10 }, // room for the modebar above the colour bar
                xaxis: { type: "category", automargin: true, tickfont: { size: 10 }, tickangle: -30 },
                yaxis: { type: "category", automargin: true, tickfont: { size: 10 }, autorange: "reversed" },
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
              }}
              config={plotConfig(`crosscheck_${metric}`)}
              useResizeHandler
              style={{ width: "100%", height: 90 + 34 * result.columns.length }}
            />
          </div>

          {comparison && order && (
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-500">
                <span className="font-medium text-slate-600">{prettyObsLabel(result.reference)} against</span>
                <select
                  aria-label="Column to compare with the reference"
                  value={pair}
                  onChange={(e) => setPair(e.target.value)}
                  className="rounded border border-slate-200 bg-white px-1.5 py-0.5 text-xs text-slate-600"
                >
                  {others.map((c) => (
                    <option key={c} value={c}>
                      {prettyObsLabel(c)}
                    </option>
                  ))}
                </select>
                <span data-testid="crosscheck-pair-stats">
                  ARI {comparison.ari.toFixed(2)}, AMI {comparison.ami.toFixed(2)}; mapped agreement{" "}
                  {pct(comparison.mapped_agreement)}; same name{" "}
                  {comparison.exact_name.n_shared_labels > 0
                    ? `${pct(comparison.exact_name.agreement)} (${comparison.exact_name.n_shared_labels} shared names)`
                    : "– (no shared names)"}
                </span>
              </div>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: order.cols,
                    y: order.rows,
                    z: tableMatrix(comparison.contingency.row_fraction, order.rows, order.cols),
                    customdata: tableMatrix(comparison.contingency.counts, order.rows, order.cols) as unknown as number[],
                    zmin: 0,
                    zmax: 1,
                    colorscale: plotlyColorscale(colormap),
                    hovertemplate:
                      "<b>%{y}</b> → <b>%{x}</b><br>%{customdata:,} cells, %{z:.1%} of %{y}<extra></extra>",
                    colorbar: { title: { text: "share of row", font: { size: 10 } }, thickness: 8 },
                    xgap: 1,
                    ygap: 1,
                    ...(order.rows.length <= MAX_LABELLED && order.cols.length <= MAX_LABELLED
                      ? { texttemplate: "%{z:.2f}", textfont: { size: 9 } }
                      : {}),
                  },
                ]}
                layout={{
                  height: Math.min(560, Math.max(240, order.rows.length * 18 + 130)),
                  margin: { l: 10, r: 10, t: 28, b: 10 }, // room for the modebar above the colour bar
                  xaxis: {
                    type: "category",
                    automargin: true,
                    tickangle: -45,
                    tickfont: { size: 10 },
                    title: { text: pair, font: { size: 10 } },
                  },
                  yaxis: {
                    type: "category",
                    automargin: true,
                    autorange: "reversed",
                    tickfont: { size: 10 },
                    title: { text: result.reference, font: { size: 10 } },
                  },
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                }}
                config={plotConfig(`crosscheck_confusion_${pair}`)}
                useResizeHandler
                style={{ width: "100%", height: Math.min(560, Math.max(240, order.rows.length * 18 + 130)) }}
              />
              <CrosscheckSankey
                key={`${result.reference}\u0000${pair}`}
                reference={result.reference}
                other={pair}
                counts={comparison.contingency.counts}
                rows={order.rows}
                cols={order.cols}
                refColours={colours(result.reference, order.rows).map((c) => rgbCss(c))}
                otherColours={colours(pair, order.cols).map((c) => rgbCss(c))}
                linkColours={colours(result.reference, order.rows).map((c) => rgbCss(c, 0.35))}
              />
            </div>
          )}

          <div>
            <div className="mb-1 text-[11px] font-medium text-slate-600">Agreement per {prettyObsLabel(result.reference)} label</div>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]" data-testid="crosscheck-labels">
                <thead>
                  <tr className="text-left text-slate-400">
                    {sortHeader("label", "Label", "")}
                    {sortHeader("n_cells", "Cells", NUM)}
                    {sortHeader(
                      "mean_agreement",
                      "Agreement",
                      NUM,
                      "Mean share of the other columns whose mapped label is this one",
                    )}
                    {sortHeader("disagreeing", "Disagree", NUM, "Columns that map most of its cells elsewhere")}
                    {others.map((o) => (
                      <th key={o} className="pl-3 font-medium" title={`What ${o} calls these cells`}>
                        {prettyObsLabel(o)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="text-slate-600">
                  {shown.map((r) => (
                    <tr key={r.label}>
                      <td className="max-w-[10rem] truncate pr-2 align-top" title={r.label}>
                        {r.label}
                      </td>
                      <td className={NUM}>{r.n_cells.toLocaleString()}</td>
                      <td
                        className={`${NUM} ${r.mean_agreement !== null && r.mean_agreement < 0.5 ? "font-semibold text-amber-700" : ""}`}
                      >
                        {pct(r.mean_agreement)}
                      </td>
                      <td className={NUM}>{r.disagreeing.length}</td>
                      {others.map((o) => (
                        <td
                          key={o}
                          className={`pl-3 align-top ${r.annotators[o]?.disagrees ? "text-amber-700" : "text-slate-500"}`}
                        >
                          {callsText(r, o)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {labelRows.length > ROWS && (
              <button type="button" onClick={() => setAllRows((v) => !v)} className="text-[11px] text-primary hover:underline">
                {allRows ? "Show fewer" : `Show all ${labelRows.length} labels`}
              </button>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2 text-[11px]">
            <label htmlFor="xc-name" className="text-slate-600">
              Name
            </label>
            <input
              id="xc-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-32 rounded border border-slate-200 px-2 py-1 font-mono text-[11px] focus:border-primary focus:outline-none"
            />
            <button
              type="button"
              onClick={() => void save()}
              disabled={saving || !validCrosscheckName(name)}
              className="rounded border border-slate-200 px-2.5 py-1 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              title={`Writes ${name.trim()}_consensus (the majority label, Unresolved on a tie) and ${name.trim()}_agreement (0–1)`}
            >
              {saving ? "Saving…" : "Save consensus + agreement as columns"}
            </button>
            {saved && (
              <>
                <button type="button" onClick={() => colourBy(saved.columns.consensus)} className="text-primary hover:underline">
                  Colour by consensus
                </button>
                <button type="button" onClick={() => colourBy(saved.columns.agreement)} className="text-primary hover:underline">
                  Colour by agreement
                </button>
              </>
            )}
          </div>
          {saved && (
            <p className="text-[11px] text-emerald-700">
              Saved {saved.columns.consensus} and {saved.columns.agreement}
              {saved.n_unresolved > 0 ? ` (${saved.n_unresolved.toLocaleString()} cells Unresolved)` : ""}.
            </p>
          )}
          <p className="text-[11px] text-slate-400">
            {result.note} {result.elapsed_seconds} s.
          </p>
        </div>
      )}
      {reference && <MarkerDbCheck datasetId={datasetId} column={reference} />}
      {error && (
        <p role="alert" className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          {error}
        </p>
      )}
    </div>
  );
}

/** Reference labels (left) flowing into the other column's labels (right), in the heatmap's order. */
function CrosscheckSankey({
  reference,
  other,
  counts,
  rows,
  cols,
  refColours,
  otherColours,
  linkColours,
}: {
  reference: string;
  other: string;
  counts: Record<string, Record<string, number>>;
  rows: string[];
  cols: string[];
  refColours: string[];
  otherColours: string[];
  linkColours: string[];
}) {
  const s = useMemo(() => sankeyData({ counts }, rows, cols), [counts, rows, cols]);
  const height = Math.min(640, Math.max(260, Math.max(rows.length, cols.length) * 22 + 40));
  return (
    <Plot
      data={[
        {
          type: "sankey",
          arrangement: "snap",
          node: {
            label: s.labels,
            color: [...refColours, ...otherColours],
            pad: 6,
            thickness: 12,
            line: { width: 0 },
            hovertemplate: "%{label}: %{value:,} cells<extra></extra>",
          },
          link: {
            source: s.source,
            target: s.target,
            value: s.value,
            color: s.source.map((i) => linkColours[i] ?? "rgba(148,163,184,0.35)"),
            hovertemplate: "%{source.label} → %{target.label}: %{value:,} cells<extra></extra>",
          },
          textfont: { size: 10 },
        },
      ]}
      layout={{
        height,
        margin: { l: 10, r: 10, t: 28, b: 26 }, // column names below, clear of the modebar
        annotations: [
          { text: reference, x: 0, y: 0, yshift: -6, xref: "paper", yref: "paper", showarrow: false, xanchor: "left", yanchor: "top", font: { size: 10 } },
          { text: other, x: 1, y: 0, yshift: -6, xref: "paper", yref: "paper", showarrow: false, xanchor: "right", yanchor: "top", font: { size: 10 } },
        ],
        paper_bgcolor: "rgba(0,0,0,0)",
      }}
      config={plotConfig(`crosscheck_sankey_${other}`)}
      useResizeHandler
      style={{ width: "100%", height }}
    />
  );
}
