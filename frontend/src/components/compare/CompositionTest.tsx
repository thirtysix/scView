import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, AlertTriangle, Download, Loader2, Sparkles } from "lucide-react";
import Plot from "@/components/plots/Plot";
import { getDataset } from "@/api/datasets";
import { testComposition, type CompositionResult } from "@/api/composition";
import type { DatasetInfo } from "@/api/types";
import { getPalette } from "@/lib/colors";
import {
  MAX_CONDITIONS,
  compositionCsv,
  conditionColumns,
  describeTest,
  isShifted,
  sortedGroups,
  suggestConditionColumn,
} from "@/lib/composition";
import { downloadCsv } from "@/lib/csv";
import { replicateColumns } from "@/lib/de";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import { rankGroupingColumns } from "@/lib/obsColumns";
import { plotConfig } from "@/lib/plotConfig";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

interface Props {
  datasetId: string;
  /** The dataset's info when the caller has it; fetched otherwise. */
  info?: DatasetInfo | null;
  /** Pre-selected columns (e.g. a joint dataset's 'dataset' and cluster columns). */
  defaultCondition?: string | null;
  defaultGroup?: string | null;
}

const selectCls = "rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700";
const pct = (v: number | null | undefined) => (v == null ? "–" : `${(v * 100).toFixed(1)}`);

/**
 * Do a column's groups (cell types, clusters) make up different shares of the
 * cells between conditions, with the donors or samples as replicates? A
 * propeller-style test (POST /composition/test), its table, CSV and a per-sample
 * plot, and a warning against reading cell-level counts as a test.
 */
export function CompositionTest({ datasetId, info, defaultCondition, defaultGroup }: Props) {
  const palette = getPalette(useSettingsStore((s) => s.categoricalPalette));
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [fetched, setFetched] = useState<DatasetInfo | null>(null);
  const dataset = info ?? fetched;
  const [conditionKey, setConditionKey] = useState("");
  const [groupKey, setGroupKey] = useState("");
  const [sampleKey, setSampleKey] = useState(""); // "" = detected by the backend
  const [excluded, setExcluded] = useState<string[]>([]);
  const [reference, setReference] = useState(""); // "" = the first value
  const [transform, setTransform] = useState<"logit" | "asin">("logit");
  const [result, setResult] = useState<CompositionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (info) return;
    let cancelled = false;
    getDataset(datasetId)
      .then((d) => !cancelled && setFetched(d))
      .catch(() => !cancelled && setFetched(null));
    return () => {
      cancelled = true;
    };
  }, [datasetId, info]);

  const cols = useMemo(() => dataset?.obs_columns ?? [], [dataset]);
  const groupOptions = useMemo(
    () =>
      rankGroupingColumns(cols, {
        nCells: dataset?.n_cells,
        activeClustering: dataset?.active_clustering,
        userColumns: dataset?.user_annotation_columns,
      }).map((g) => g.name),
    [cols, dataset],
  );
  const conditionOptions = useMemo(() => conditionColumns(cols, [groupKey]), [cols, groupKey]);
  const sampleOptions = useMemo(
    () => replicateColumns(cols, [conditionKey, groupKey]),
    [cols, conditionKey, groupKey],
  );
  const conditionValues = useMemo(
    () => cols.find((c) => c.name === conditionKey)?.values ?? [],
    [cols, conditionKey],
  );

  // Defaults once the columns are known: cell types / clusters as the groups,
  // the most condition-like column as the condition.
  useEffect(() => {
    if (!cols.length) return;
    const names = new Set(cols.map((c) => c.name));
    const group =
      defaultGroup && names.has(defaultGroup) ? defaultGroup : (groupOptions[0] ?? "");
    setGroupKey(group);
    const conds = conditionColumns(cols, [group]);
    setConditionKey(
      defaultCondition && conds.some((c) => c.name === defaultCondition)
        ? defaultCondition
        : suggestConditionColumn(conds),
    );
  }, [cols, groupOptions, defaultCondition, defaultGroup]);

  // A new condition column starts with all its values, the first as reference.
  useEffect(() => {
    const next = conditionValues.length > MAX_CONDITIONS ? conditionValues.slice(2) : [];
    // Keep the same empty array, or the test would run twice.
    setExcluded((ex) => (ex.length === 0 && next.length === 0 ? ex : next));
    setReference("");
    setSampleKey("");
  }, [conditionKey, conditionValues]);

  const included = useMemo(
    () => conditionValues.filter((v) => !excluded.includes(v)),
    [conditionValues, excluded],
  );

  useEffect(() => {
    if (!conditionKey || !groupKey || conditionKey === groupKey) {
      setResult(null);
      return;
    }
    if (conditionValues.length && included.length < 2) {
      setResult(null);
      setError(`Pick at least two values of ${prettyObsLabel(conditionKey)} to compare.`);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    testComposition(datasetId, {
      condition_key: conditionKey,
      group_key: groupKey,
      ...(sampleKey ? { sample_key: sampleKey } : {}),
      ...(excluded.length ? { conditions: included } : {}),
      ...(reference && included.includes(reference) ? { reference } : {}),
      transform,
    })
      .then((r) => !cancelled && setResult(r))
      .catch((e) => {
        if (cancelled) return;
        setResult(null);
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, conditionKey, groupKey, sampleKey, excluded, included, reference, transform, conditionValues]);

  const rows = useMemo(() => (result ? sortedGroups(result) : []), [result]);

  const exportCsv = useCallback(() => {
    if (!result) return;
    const { headers, rows: csvRows } = compositionCsv(result);
    downloadCsv(`composition_test_${result.group_key}_by_${result.condition_key}_${datasetId}.csv`, headers, csvRows);
  }, [result, datasetId]);

  const ask = useCallback(() => {
    if (!result) return;
    const top = rows
      .slice(0, 5)
      .map((g) => `${g.group}: ${result.conditions.map((k) => `${pct(g.props[k])}%`).join(" -> ")}` +
        (g.fdr != null ? ` (FDR ${formatPValue(g.fdr)})` : ""))
      .join("; ");
    askCopilot(
      `Interpret the composition test of ${result.group_key} between ${result.conditions.join(" and ")}` +
        (result.sample_key ? ` with ${result.sample_key} as replicates` : "") +
        `: ${top}. Which shifts are credible, and what could explain them?`,
    );
  }, [result, rows, askCopilot]);

  if (!dataset) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading columns…
      </div>
    );
  }

  const color = (i: number) => `rgb(${palette[(i * 3) % palette.length]!.join(",")})`;
  const tested = result?.tested ?? false;
  const two = (result?.conditions.length ?? 0) === 2;

  return (
    <div className="space-y-4">
      <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Conditions</span>
            <select
              aria-label="Condition column"
              value={conditionKey}
              onChange={(e) => setConditionKey(e.target.value)}
              className={selectCls}
            >
              <option value="">Select a column…</option>
              {conditionOptions.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Groups</span>
            <select
              aria-label="Group column"
              value={groupKey}
              onChange={(e) => setGroupKey(e.target.value)}
              className={selectCls}
            >
              {groupOptions.length === 0 && <option value="">(no grouping column)</option>}
              {groupOptions.map((c) => (
                <option key={c} value={c}>
                  {prettyObsLabel(c)}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Samples (replicates)</span>
            <select
              aria-label="Sample column"
              value={sampleKey}
              onChange={(e) => setSampleKey(e.target.value)}
              className={selectCls}
            >
              <option value="">
                {!sampleKey && result?.sample_key_detected
                  ? `Auto (${prettyObsLabel(result.sample_key)})`
                  : "Auto"}
              </option>
              {sampleOptions.map((c) => (
                <option key={c.name} value={c.name}>
                  {prettyObsLabel(c.name)}
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Scale</span>
            <select
              aria-label="Transform"
              value={transform}
              onChange={(e) => setTransform(e.target.value as "logit" | "asin")}
              className={selectCls}
            >
              <option value="logit">Logit (default)</option>
              <option value="asin">Arcsin square root</option>
            </select>
          </label>
        </div>
        {conditionValues.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-slate-500">Compare:</span>
            {conditionValues.map((v) => {
              const on = !excluded.includes(v);
              return (
                <button
                  key={v}
                  aria-pressed={on}
                  onClick={() => setExcluded((ex) => (on ? [...ex, v] : ex.filter((x) => x !== v)))}
                  className={`rounded-full border px-2.5 py-0.5 ${
                    on ? "border-primary/40 bg-primary/10 text-primary" : "border-slate-200 text-slate-400"
                  }`}
                >
                  {v}
                </button>
              );
            })}
            {included.length >= 2 && (
              <label className="ml-2 flex items-center gap-1.5 text-slate-500">
                Reference
                <select
                  aria-label="Reference condition"
                  value={reference && included.includes(reference) ? reference : (result?.reference ?? included[0])}
                  onChange={(e) => setReference(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700"
                >
                  {included.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {included.length > MAX_CONDITIONS && (
              <span className="text-amber-700">At most {MAX_CONDITIONS} at once.</span>
            )}
          </div>
        )}
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          {error}
        </div>
      )}
      {!conditionKey && !error && (
        <p className="text-sm text-slate-500">
          Pick the column that holds the conditions (e.g. control vs stimulated) to test whether the{" "}
          {groupKey ? prettyObsLabel(groupKey) : "group"} proportions differ between them.
        </p>
      )}
      {loading && (
        <div className="flex items-center justify-center gap-2 py-8 text-sm text-slate-400">
          <Loader2 className="h-5 w-5 animate-spin" /> Testing composition…
        </div>
      )}

      {!loading && result && (
        <>
          {result.warnings.map((w) => (
            <div
              key={w}
              role="alert"
              className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-800"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
              <span>{w}</span>
            </div>
          ))}
          <CellLevelCaution result={result} />

          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-700">
                {tested
                  ? `${result.n_significant} of ${result.groups.length} groups shift (FDR < ${result.fdr_threshold})`
                  : `Proportions by ${prettyObsLabel(result.group_key)} (not tested)`}
              </h4>
              <div className="ml-auto flex gap-1.5">
                <button
                  onClick={ask}
                  title="Ask the co-pilot what these results suggest"
                  className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 hover:text-primary"
                >
                  <Sparkles className="h-3 w-3" /> Ask
                </button>
                <button
                  onClick={exportCsv}
                  title="Export the table as CSV"
                  className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  <Download className="h-3 w-3" /> CSV
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
                    <th className="px-3 py-2 font-medium">{prettyObsLabel(result.group_key)}</th>
                    {result.conditions.map((k) => (
                      <th key={k} className="px-3 py-2 text-right font-medium" title="Mean over its samples">
                        {k} %
                      </th>
                    ))}
                    {two && (
                      <th className="px-3 py-2 text-right font-medium">
                        {result.conditions[1]} / {result.conditions[0]}
                      </th>
                    )}
                    {tested && (
                      <>
                        <th className="px-3 py-2 text-right font-medium">{two ? "t" : "F"}</th>
                        <th className="px-3 py-2 text-right font-medium">p</th>
                        <th className="px-3 py-2 text-right font-medium">FDR</th>
                      </>
                    )}
                    <th className="px-3 py-2 text-right font-medium">Cells</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((g) => {
                    const hit = isShifted(g, result);
                    return (
                      <tr key={g.group} className={`border-b border-slate-50 ${hit ? "bg-emerald-50/60" : ""}`}>
                        <td className={`px-3 py-1.5 ${hit ? "font-semibold text-slate-900" : "text-slate-800"}`}>
                          {g.group}
                        </td>
                        {result.conditions.map((k) => (
                          <td key={k} className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                            {pct(g.props[k])}
                          </td>
                        ))}
                        {two && (
                          <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                            {g.fold_change == null ? "–" : `×${g.fold_change.toFixed(2)}`}
                          </td>
                        )}
                        {tested && (
                          <>
                            <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                              {g.stat == null ? "–" : g.stat.toFixed(2)}
                            </td>
                            <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                              {g.pval == null ? "–" : formatPValue(g.pval)}
                            </td>
                            <td className={`px-3 py-1.5 text-right tabular-nums ${hit ? "font-semibold text-emerald-700" : "text-slate-600"}`}>
                              {g.fdr == null ? "–" : formatPValue(g.fdr)}
                            </td>
                          </>
                        )}
                        <td className="px-3 py-1.5 text-right font-mono text-xs text-slate-500">
                          {result.conditions.map((k) => (g.n_cells[k] ?? 0).toLocaleString()).join(" / ")}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-slate-500">
              {describeTest(result)} Shares are the mean over each condition&apos;s samples.
              {result.notes.map((n) => ` ${n}`)}
              {result.dropped_samples.length > 0 &&
                ` Left out (under 10 cells): ${result.dropped_samples.join(", ")}.`}
            </p>
          </div>

          {result.samples.length > 0 && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <Plot
                data={result.conditions.map((k, ci) => {
                  const samples = result.samples.filter((s) => s.condition === k);
                  const order = rows.map((g) => result.groups.indexOf(g));
                  return {
                    type: "box" as const,
                    name: k,
                    x: samples.flatMap(() => order.map((gi) => result.groups[gi]!.group)),
                    y: samples.flatMap((s) => order.map((gi) => (s.props[gi] ?? 0) * 100)),
                    text: samples.flatMap((s) => order.map(() => `${s.sample} (${s.n_cells.toLocaleString()} cells)`)),
                    boxpoints: "all" as const,
                    jitter: 0.4,
                    pointpos: 0,
                    marker: { color: color(ci), size: 5 },
                    line: { width: 1 },
                    hovertemplate: "%{x}: %{y:.1f}%<br>%{text}<extra>" + k + "</extra>",
                  };
                })}
                layout={{
                  boxmode: "group",
                  autosize: true,
                  height: 380,
                  margin: { t: 30, r: 16, b: 110, l: 56 },
                  title: {
                    text: `${prettyObsLabel(result.group_key)}: share of each sample's cells`,
                    font: { size: 14, color: "#334155" },
                  },
                  xaxis: { tickangle: -45, tickfont: { size: 10, color: "#64748b" } },
                  yaxis: {
                    title: { text: "% of the sample's cells", font: { size: 11, color: "#64748b" } },
                    gridcolor: "#f1f5f9",
                    tickfont: { size: 10 },
                  },
                  legend: { orientation: "h", y: 1.12, font: { size: 11 } },
                  paper_bgcolor: "white",
                  plot_bgcolor: "white",
                }}
                config={plotConfig(`composition_samples_${result.group_key}`, dataset.name)}
                useResizeHandler
                style={{ width: "100%" }}
              />
              <p className="mt-1 text-xs text-slate-500">
                One point per sample. The test asks whether the conditions differ by more than the
                samples within a condition do.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** Why the pooled cell counts are not a test, with this result's own numbers. */
function CellLevelCaution({ result }: { result: CompositionResult }) {
  const text = result.tested
    ? `A chi-square or Fisher test on the pooled cell counts treats every cell as an independent ` +
      `replicate. Here it would call ${result.n_significant_cell_level ?? "?"} of ` +
      `${result.groups.length} groups significant at FDR < ${result.fdr_threshold}; with the samples ` +
      `as replicates, ${result.n_significant} are. Cells from one sample share its make-up, so at ` +
      `the cell level ordinary sample-to-sample variation looks like a condition effect.`
    : `Without replicates, nothing can tell a condition effect from one sample's own make-up. A ` +
      `chi-square or Fisher test on the cell counts would still give small p-values, because it ` +
      `treats every cell as an independent replicate: read these differences as descriptive.`;
  return (
    <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs text-slate-600">
      {text}
    </p>
  );
}
