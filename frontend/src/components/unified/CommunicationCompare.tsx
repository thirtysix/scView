import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Download, Loader2, Play, Search, Square } from "lucide-react";
import type { PlotMouseEvent } from "plotly.js";
import Plot from "@/components/plots/Plot";
import { DonorBoxPlot } from "@/components/plots/DonorBoxPlot";
import { Caption, CapNote } from "@/components/common/Caption";
import { InfoTip } from "@/components/common/InfoTip";
import { RequestError } from "@/components/common/RequestError";
import {
  compareCsvUrl,
  getCompareDonors,
  runCommunicationCompare,
  type CommunicationOptions,
  type CompareDonors,
  type CompareStart,
} from "@/api/communication";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { rgbCss } from "@/lib/categoryColors";
import { plotlyColorscale } from "@/lib/colors";
import type { Weight } from "@/lib/communication";
import {
  CALL_LABELS,
  compareLabel,
  compareSummary,
  diffColorRange,
  diffMatrix,
  filterCompareRows,
  sortCompareRows,
  toCompareRows,
  DEFAULT_COMPARE_FILTER,
  type CallFilter,
  type CompareFilter,
  type CompareRow,
  type CompareSortKey,
} from "@/lib/communicationCompare";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import { plotConfig } from "@/lib/plotConfig";
import { useCommunicationStore } from "@/stores/communicationStore";
import { useSettingsStore } from "@/stores/settingsStore";

interface Props {
  datasetId: string;
  datasetName?: string;
  /** Sender / receiver column, chosen in the tab */
  groupby: string;
  options: CommunicationOptions | null;
}

const TABLE_LIMIT = 200;
const COLUMNS: { key: CompareSortKey; label: string; title: string; num?: boolean }[] = [
  { key: "source", label: "Sender", title: "Source group (expresses the ligand)" },
  { key: "target", label: "Receiver", title: "Target group (expresses the receptor)" },
  { key: "ligand_complex", label: "Ligand", title: "Ligand (a complex as A_B)" },
  { key: "receptor_complex", label: "Receptor", title: "Receptor (a complex as A_B)" },
  { key: "call", label: "Call", title: "Gained / lost: specific in one condition only, LR mean 1.5x or more higher there; changed: specific in both, LR means 1.5x apart" },
  { key: "lr_means_a", label: "A", title: "Ligand-receptor mean in the reference condition (A)", num: true },
  { key: "lr_means_b", label: "B", title: "Ligand-receptor mean in the compared condition (B)", num: true },
  { key: "abs_delta", label: "Δ", title: "B minus A (LR mean); sorted by its size", num: true },
  { key: "p_adj", label: "adj. p", title: "Donor-level test of the ligand x receptor product, Benjamini-Hochberg across the tested interactions; – without replicates", num: true },
];

const CALL_FILTERS: { value: CallFilter; label: string }[] = [
  { value: "moved", label: "Gained, lost or changed" },
  { value: "gained", label: "Gained" },
  { value: "lost", label: "Lost" },
  { value: "changed", label: "Changed" },
  { value: "unchanged", label: "Unchanged" },
  { value: "all", label: "All" },
];

const fmt = (v: number | null, digits = 2) =>
  v === null ? "–" : Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(1) : v.toFixed(digits);
const signed = (v: number | null) => (v === null ? "–" : `${v > 0 ? "+" : ""}${fmt(v)}`);

const CALL_CLASS: Record<string, string> = {
  gained: "text-emerald-700",
  lost: "text-rose-700",
  changed: "text-amber-700",
  unchanged: "text-slate-500",
};

/**
 * Communication compared between two conditions: both runs on the same groups,
 * resource and genes (backend core/ccc_compare.py), a sender x receiver heatmap
 * of the change in specific interactions, a ranked table of the changed ones,
 * and, with donors, one interaction's ligand x receptor product per donor.
 */
export function CommunicationCompare({ datasetId, datasetName, groupby, options }: Props) {
  const settings = useCommunicationStore();
  const { update, setCompareResult } = settings;
  const stored = settings.compareResult;
  const result = stored && stored.datasetId === datasetId ? stored.result : null;
  const diverging = useSettingsStore((s) => s.divergingColormap);

  const conditions = options?.conditions ?? [];
  const condition = conditions.find((c) => c.column === settings.compareColumn) ?? conditions[0] ?? null;
  const values = condition?.values ?? [];
  const a = values.includes(settings.compareA) ? settings.compareA : (values[0] ?? "");
  const bWanted = values.includes(settings.compareB) ? settings.compareB : (values.find((v) => v !== a) ?? "");
  const b = bWanted === a ? (values.find((v) => v !== a) ?? "") : bWanted;

  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [start, setStart] = useState<CompareStart | null>(null);
  const [progress, setProgress] = useState<{ condition: string; done: number; total: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [weight, setWeight] = useState<Weight>("count");
  const [filter, setFilter] = useState<CompareFilter>(DEFAULT_COMPARE_FILTER);
  const [sort, setSort] = useState<{ key: CompareSortKey; asc: boolean }>({ key: "abs_delta", asc: false });
  const [picked, setPicked] = useState<CompareRow | null>(null);
  const [donors, setDonors] = useState<CompareDonors | null>(null);
  const [donorError, setDonorError] = useState<string | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const run = async () => {
    if (!groupby || !condition || !a || !b || running) return;
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setRunning(true);
    setError(null);
    setStart(null);
    setProgress(null);
    try {
      const res = await runCommunicationCompare(
        datasetId,
        { groupby, condition: condition.column, a, b, expr_prop: settings.exprProp },
        {
          onStart: setStart,
          onProgress: (c, done, total) => setProgress({ condition: c, done, total }),
          signal: ctrl.signal,
        },
      );
      setCompareResult(datasetId, res);
      setFilter(DEFAULT_COMPARE_FILTER);
      setPicked(null);
      setDonors(null);
    } catch (e) {
      if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const rows = useMemo(() => (result ? toCompareRows(result) : []), [result]);
  const filtered = useMemo(() => filterCompareRows(rows, filter), [rows, filter]);
  const sorted = useMemo(() => sortCompareRows(filtered, sort.key, sort.asc), [filtered, sort]);
  const matrix = useMemo(() => (result ? diffMatrix(result, weight) : []), [result, weight]);
  const range = useMemo(() => diffColorRange(matrix, diverging), [matrix, diverging]);
  const condColor = useCategoryColors(result?.condition);

  // One interaction per donor, fetched when picked.
  useEffect(() => {
    if (!result || !picked || !result.replicates.test) return;
    let cancelled = false;
    setDonorError(null);
    getCompareDonors(datasetId, result.result_id, picked.index)
      .then((d) => {
        if (!cancelled) setDonors(d);
      })
      .catch((e) => {
        if (!cancelled) setDonorError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, result, picked]);

  const sel = "w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700";
  const progressPct = progress ? Math.round((100 * progress.done) / Math.max(1, progress.total)) : 0;

  if (conditions.length === 0) {
    return (
      <p className="text-xs text-slate-500">
        No condition column found (a column with a few values such as stim or treatment). Comparing needs one.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-2">
        <div>
          <label htmlFor="ccc-cmp-column" className="mb-1 block text-[11px] font-medium text-slate-500">
            Condition
          </label>
          <select
            id="ccc-cmp-column"
            value={condition?.column ?? ""}
            onChange={(e) => update({ compareColumn: e.target.value, compareA: "", compareB: "" })}
            className={sel}
          >
            {conditions.map((c) => (
              <option key={c.column} value={c.column}>
                {prettyObsLabel(c.column)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="ccc-cmp-a" className="mb-1 block text-[11px] font-medium text-slate-500">
            Reference (A)
          </label>
          <select id="ccc-cmp-a" value={a} onChange={(e) => update({ compareA: e.target.value })} className={sel}>
            {values.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="ccc-cmp-b" className="mb-1 block text-[11px] font-medium text-slate-500">
            Compared (B)
          </label>
          <select id="ccc-cmp-b" value={b} onChange={(e) => update({ compareB: e.target.value })} className={sel}>
            {values
              .filter((v) => v !== a)
              .map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
          </select>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-[11px] text-slate-500" htmlFor="ccc-cmp-expr-prop">
          Detected in at least
        </label>
        <select
          id="ccc-cmp-expr-prop"
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
              disabled={!groupby || !a || !b}
              className="flex items-center gap-1 rounded-md bg-primary px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
            >
              <Play className="h-3 w-3" /> Compare
            </button>
          )}
        </div>
      </div>
      {running && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            {start
              ? `Scoring ${start.n_groups} groups in ${start.a} and ${start.b} (about ${Math.max(1, Math.round(start.estimate_seconds))} s)` +
                (progress ? `: ${progress.condition}, permutation ${progress.done} of ${progress.total}` : "")
              : "Preparing… (the first run downloads the resource)"}
          </div>
          <div className="h-1.5 w-full rounded bg-slate-100">
            <div className="h-1.5 rounded bg-primary transition-all" style={{ width: `${progressPct}%` }} />
          </div>
        </div>
      )}
      <RequestError message={error} />

      {result && (
        <>
          <Caption n={[{ value: result.n_cells, unit: "cells" }, { value: result.groups.length, unit: "groups" }]}>
            {compareSummary(result)} ({prettyObsLabel(result.groupby)}; {prettyObsLabel(result.condition)}).
            {Object.keys(result.dropped_groups).length > 0 &&
              ` Left out of both (under ${result.min_cells} cells in a condition): ${Object.entries(result.dropped_groups)
                .map(([g, n]) => `${g} (${n.a} / ${n.b})`)
                .join(", ")}.`}
          </Caption>
          <p className="text-[11px] leading-snug text-slate-500" data-testid="ccc-cmp-replicates">
            {result.replicates.test ? (
              <>
                Tested per donor ({prettyObsLabel(result.replicates.sample_key)}:{" "}
                {Object.entries(result.replicates.n_donors)
                  .map(([k, n]) => `${n} in ${k}`)
                  .join(", ")}
                ) with the {result.replicates.test}, Benjamini-Hochberg across the interactions.{" "}
                {result.replicates.caution}
              </>
            ) : (
              result.replicates.reason
            )}{" "}
            <InfoTip label="About the comparison" align="right">
              Each condition is scored as a single run would score it, on the same groups (a group with fewer than{" "}
              {result.min_cells} cells in either condition is left out of both) and the same ligand-receptor pairs. An
              interaction counts in a condition when its specificity rank is at most {result.specific_threshold}.
              Gained: counts in B only, with its ligand-receptor mean at least {result.fold}x higher in B; lost: the
              reverse; changed: counts in both, means {result.fold}x apart. The heatmap is CellChat&apos;s differential
              number (or strength) of interactions. P-values come only from donors: each donor&apos;s mean expression of
              the ligand in the sender times the receptor in the receiver, compared between conditions. Cells are not
              replicates, so without donors there is no test.
            </InfoTip>
          </p>
          {result.composition.some((c) => c.flagged) && (
            <p role="note" className="flex gap-1.5 text-[11px] text-amber-800">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
              <span>
                Composition changed:{" "}
                {result.composition
                  .filter((c) => c.flagged)
                  .map((c) => `${c.group} ${(100 * c.share_a).toFixed(1)}% → ${(100 * c.share_b).toFixed(1)}% of the cells`)
                  .join(", ")}
                . Their pairs are marked in the table: a change there may follow the cells&apos; numbers or state, not
                signalling alone.
              </span>
            </p>
          )}

          <div>
            <div className="mb-1 flex items-center gap-2">
              <span className="text-[11px] font-medium text-slate-600">Change between groups ({result.b} − {result.a})</span>
              <select
                aria-label="Heatmap value"
                value={weight}
                onChange={(e) => setWeight(e.target.value as Weight)}
                className="ml-auto rounded border border-slate-200 bg-white px-1 py-0.5 text-[11px]"
              >
                <option value="count">Count</option>
                <option value="strength">Strength</option>
              </select>
            </div>
            <div style={{ height: Math.max(260, result.groups.length * 18 + 130) }}>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: result.groups,
                    y: result.groups,
                    z: matrix,
                    zmin: range.zmin,
                    zmax: range.zmax,
                    colorscale: plotlyColorscale(range.colormap),
                    colorbar: { title: { text: weight === "count" ? "Δ n" : "Δ sum", font: { size: 9 } }, thickness: 8, tickfont: { size: 8 } },
                    hovertemplate: `%{y} → %{x}<br>${result.b} − ${result.a}: %{z:.3~g}<extra></extra>`,
                  },
                ]}
                layout={{
                  autosize: true,
                  margin: { l: 10, r: 6, t: 8, b: 10 },
                  xaxis: { type: "category", tickangle: -45, tickfont: { size: 9 }, automargin: true, title: { text: "Receiver", font: { size: 10 } } },
                  yaxis: { type: "category", tickfont: { size: 9 }, automargin: true, autorange: "reversed", title: { text: "Sender", font: { size: 10 } } },
                }}
                config={plotConfig(`communication_compare_${weight}`, datasetName)}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
                onClick={(e: Readonly<PlotMouseEvent>) => {
                  const p = e.points[0];
                  if (p) setFilter((f) => ({ ...f, source: String(p.y), target: String(p.x) }));
                }}
              />
            </div>
            <Caption n={{ value: result.groups.length, unit: "groups" }}>
              {weight === "count" ? "Specific interactions" : "Summed ligand-receptor mean of the specific interactions"} in{" "}
              {result.b} minus {result.a}, per sender (rows) and receiver (columns); colours centred on 0. Click a cell to
              filter the table to that pair.
            </Caption>
          </div>

          <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
            <select
              aria-label="Call"
              value={filter.call}
              onChange={(e) => setFilter((f) => ({ ...f, call: e.target.value as CallFilter }))}
              className={sel}
            >
              {CALL_FILTERS.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
            <select aria-label="Sender" value={filter.source} onChange={(e) => setFilter((f) => ({ ...f, source: e.target.value }))} className={sel}>
              <option value="">Any sender</option>
              {result.groups.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
            <select aria-label="Receiver" value={filter.target} onChange={(e) => setFilter((f) => ({ ...f, target: e.target.value }))} className={sel}>
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
                aria-label="Ligand or receptor"
                value={filter.query}
                onChange={(e) => setFilter((f) => ({ ...f, query: e.target.value }))}
                placeholder="Ligand / receptor"
                className="w-full rounded border border-slate-300 py-1.5 pl-5 pr-1 text-xs"
              />
            </div>
          </div>
          <label className="flex items-center gap-1.5 text-[11px] text-slate-600">
            <input
              type="checkbox"
              checked={filter.maxPadj !== null}
              disabled={!result.replicates.test}
              onChange={(e) => setFilter((f) => ({ ...f, maxPadj: e.target.checked ? 0.05 : null }))}
            />
            Only adjusted p &lt; 0.05{!result.replicates.test && " (no donor test here)"}
          </label>

          <div>
            <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
              <span>{filtered.length.toLocaleString()} rows</span>
              <a
                href={compareCsvUrl(datasetId, result.result_id)}
                download
                className="flex items-center gap-1 text-primary hover:underline"
                title="Every compared row: LR means, ranks and calls in both conditions, donor p-values"
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
                        aria-sort={sort.key === c.key ? (sort.asc ? "ascending" : "descending") : undefined}
                        className={`px-1.5 py-1.5 font-medium ${c.num ? "text-right" : "text-left"}`}
                      >
                        <button
                          type="button"
                          className="select-none"
                          onClick={() =>
                            setSort((s) =>
                              s.key === c.key ? { key: c.key, asc: !s.asc } : { key: c.key, asc: c.key === "p_adj" || !c.num },
                            )
                          }
                        >
                          {c.label}
                          {sort.key === c.key ? (sort.asc ? " ▲" : " ▼") : ""}
                        </button>
                      </th>
                    ))}
                    <th className="px-1.5 py-1.5 text-right font-medium" title="Plot this interaction per donor">
                      Donors
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.slice(0, TABLE_LIMIT).map((r) => (
                    <tr
                      key={r.index}
                      className={`border-b border-slate-50 hover:bg-blue-50 ${picked?.index === r.index ? "bg-blue-50" : ""}`}
                      title={compareLabel(r)}
                    >
                      <td className="max-w-[5rem] truncate px-1.5 py-1 text-slate-700">{r.source}</td>
                      <td className="max-w-[5rem] truncate px-1.5 py-1 text-slate-700">{r.target}</td>
                      <td className="max-w-[6rem] truncate px-1.5 py-1 font-mono text-slate-800">{r.ligand_complex}</td>
                      <td className="max-w-[6rem] truncate px-1.5 py-1 font-mono text-slate-800">{r.receptor_complex}</td>
                      <td className={`px-1.5 py-1 ${CALL_CLASS[r.call] ?? ""}`}>
                        {CALL_LABELS[r.call as keyof typeof CALL_LABELS] ?? r.call}
                        {r.composition_note && (
                          <span
                            role="img"
                            aria-label="Sender or receiver changed share of the cells"
                            title="Sender or receiver changed share of the cells"
                          >
                            <AlertTriangle className="ml-0.5 inline h-3 w-3 text-amber-600" aria-hidden />
                          </span>
                        )}
                      </td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.lr_means_a)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{fmt(r.lr_means_b)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{signed(r.delta)}</td>
                      <td className="px-1.5 py-1 text-right tabular-nums">{r.p_adj === null ? "–" : formatPValue(r.p_adj)}</td>
                      <td className="px-1.5 py-1 text-right">
                        <button
                          type="button"
                          disabled={!result.replicates.test}
                          aria-pressed={picked?.index === r.index}
                          aria-label={`Plot per donor: ${compareLabel(r)}`}
                          onClick={() => {
                            setDonors(null);
                            setPicked((p) => (p?.index === r.index ? null : r));
                          }}
                          className="rounded border border-slate-200 px-1 text-[10px] text-slate-600 hover:bg-white disabled:opacity-40"
                        >
                          {r.n_donors || "–"}
                        </button>
                      </td>
                    </tr>
                  ))}
                  {sorted.length === 0 && (
                    <tr>
                      <td colSpan={COLUMNS.length + 1} className="px-2 py-4 text-center text-slate-400">
                        No interaction matches.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            <CapNote
              shown={Math.min(TABLE_LIMIT, sorted.length)}
              total={sorted.length}
              by={COLUMNS.find((c) => c.key === sort.key)?.label ?? sort.key}
              rest={result.rows_shown < result.n_rows ? "the CSV has all" : undefined}
            />
          </div>

          {picked && result.replicates.test && (
            <div data-testid="ccc-cmp-donors">
              <RequestError message={donorError} />
              {donors?.available && (
                <DonorBoxPlot
                  points={donors.points.map((p) => ({
                    group: compareLabel(picked),
                    condition: p.condition,
                    sample: p.donor,
                    n_cells: p.n_cells,
                    value: p.value,
                  }))}
                  groups={[compareLabel(picked)]}
                  conditions={[result.a, result.b]}
                  sampleKey={prettyObsLabel(result.replicates.sample_key)}
                  colorOf={(k) => rgbCss(condColor(k))}
                  valueLabel="ligand mean × receptor mean"
                  format={(v) => v.toFixed(2)}
                  plotName="communication_compare_donors"
                  datasetName={datasetName}
                  height={300}
                  design={{ paired: Boolean(donors.paired), n_pairs: donors.paired_test?.n_pairs ?? 0, conditions: [result.a, result.b] }}
                  pairedTests={donors.paired_test ? { [compareLabel(picked)]: donors.paired_test } : undefined}
                  unit="(product)"
                  unitHelp="The donor's mean ligand expression in the sender times its mean receptor expression in the receiver (log-normalised)."
                  note={
                    donors.p_adj != null
                      ? `Adjusted p across the interactions ${formatPValue(donors.p_adj)} (${donors.test}).`
                      : "Too few donors with enough cells for a test here."
                  }
                />
              )}
            </div>
          )}

          <p className="text-[10px] leading-snug text-slate-400">
            {result.resource_label}, {result.n_interactions.toLocaleString()} ligand-receptor pairs expressed;{" "}
            {result.n_perms} permutations per condition (seed {result.seed}); scored on the {result.matrix_source} (
            {result.elapsed_seconds}s). {result.citation}.
          </p>
        </>
      )}
    </div>
  );
}
