import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, Download, Loader2, MousePointerClick, Play, Sparkles } from "lucide-react";
import { getDataset } from "@/api/datasets";
import { miloCells, planMilo, runMilo, type MiloPlan, type MiloRequest, type MiloResult } from "@/api/milo";
import type { DatasetInfo } from "@/api/types";
import { MiloBeeswarm, MiloNhoodGraph } from "@/components/compare/MiloPlots";
import { conditionColumns, suggestConditionColumn } from "@/lib/composition";
import { downloadCsv } from "@/lib/csv";
import { replicateColumns } from "@/lib/de";
import { prettyObsLabel } from "@/lib/formatting";
import { describeMilo, miloCsv, miloQuestion, nhoodRows, pickLayout, shifted } from "@/lib/milo";
import { isCategoricalDtype, rankGroupingColumns, rankOptionsFor } from "@/lib/obsColumns";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { errorMessage } from "@/lib/errorMessage";

interface Props {
  datasetId: string;
  info?: DatasetInfo | null;
}

const selectCls = "rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700";
const btnCls =
  "flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50 hover:text-primary";
const PROPS = [0.05, 0.1, 0.2] as const;

/**
 * Milo (Dann et al. 2022): differential abundance in overlapping neighbourhoods
 * of the cell graph, with the samples as replicates. Plan (design + time
 * estimate), a streamed run, a beeswarm by label and the neighbourhood graph,
 * CSV, and selecting a neighbourhood's cells on the scatter.
 */
export function MiloTest({ datasetId, info }: Props) {
  const askCopilot = useViewStore((s) => s.askCopilot);
  const setPanel = useViewStore((s) => s.setPanel);
  const setSelection = useSelectionStore((s) => s.setSelection);
  const scatterEmbedding = useSettingsStore((s) => s.embedding);

  const [fetched, setFetched] = useState<DatasetInfo | null>(null);
  const dataset = info ?? fetched;
  const [conditionKey, setConditionKey] = useState("");
  const [reference, setReference] = useState("");
  const [tested, setTested] = useState("");
  const [sampleKey, setSampleKey] = useState(""); // "" = detected by the backend
  const [labelKey, setLabelKey] = useState("");
  const [covariate, setCovariate] = useState("");
  const [pairing, setPairing] = useState<"auto" | "paired" | "unpaired">("auto");
  const [prop, setProp] = useState<number | null>(null); // null = the backend's default
  const [plan, setPlan] = useState<MiloPlan | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<MiloResult | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [layout, setLayout] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

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
  useEffect(() => () => abort.current?.abort(), []);

  const cols = useMemo(() => dataset?.obs_columns ?? [], [dataset]);
  const labelOptions = useMemo(
    () =>
      rankGroupingColumns(cols, rankOptionsFor(dataset)).map((g) => g.name),
    [cols, dataset],
  );
  const conditionOptions = useMemo(() => conditionColumns(cols, [labelKey]), [cols, labelKey]);
  const sampleOptions = useMemo(
    () => replicateColumns(cols, [conditionKey, labelKey]),
    [cols, conditionKey, labelKey],
  );
  const covariateOptions = useMemo(
    () =>
      cols.filter(
        (c) =>
          isCategoricalDtype(c.dtype) &&
          c.n_unique >= 2 &&
          c.n_unique <= 50 &&
          ![conditionKey, sampleKey, plan?.sample_key, labelKey].includes(c.name),
      ),
    [cols, conditionKey, sampleKey, plan?.sample_key, labelKey],
  );
  const conditionValues = useMemo(
    () => cols.find((c) => c.name === conditionKey)?.values ?? [],
    [cols, conditionKey],
  );

  // Defaults once the columns are known.
  useEffect(() => {
    if (!cols.length) return;
    const label = labelOptions[0] ?? "";
    setLabelKey(label);
    setConditionKey(suggestConditionColumn(conditionColumns(cols, [label])));
  }, [cols, labelOptions]);

  // A new condition column: its first two values, the first as reference.
  useEffect(() => {
    setReference(conditionValues[0] ?? "");
    setTested(conditionValues[1] ?? "");
    setSampleKey("");
    setCovariate("");
    setPairing("auto");
  }, [conditionKey, conditionValues]);

  const request = useMemo<MiloRequest | null>(() => {
    if (!conditionKey) return null;
    return {
      condition_key: conditionKey,
      ...(sampleKey ? { sample_key: sampleKey } : {}),
      ...(reference && tested ? { conditions: [reference, tested] } : {}),
      ...(covariate ? { covariates: [covariate] } : {}),
      ...(pairing !== "auto" ? { paired: pairing === "paired" } : {}),
      ...(labelKey ? { label_key: labelKey } : {}),
      ...(prop != null ? { prop } : {}),
    };
  }, [conditionKey, sampleKey, reference, tested, covariate, pairing, labelKey, prop]);

  // The plan (design, samples, time estimate) follows every change; cheap.
  useEffect(() => {
    if (!request) {
      setPlan(null);
      setPlanError(null);
      return;
    }
    if (conditionValues.length >= 2 && (!reference || !tested)) return; // defaults on their way
    if (reference && reference === tested) {
      setPlan(null);
      setPlanError("Pick two different values to compare.");
      return;
    }
    let cancelled = false;
    planMilo(datasetId, request)
      .then((p) => {
        if (cancelled) return;
        setPlan(p);
        setPlanError(null);
      })
      .catch((e) => {
        if (cancelled) return;
        setPlan(null);
        setPlanError(errorMessage(e));
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId, request, reference, tested, conditionValues]);

  const run = useCallback(() => {
    if (!request) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setRunError(null);
    setPicked(null);
    setProgress("Starting…");
    runMilo(datasetId, request, { onProgress: (p) => setProgress(p.message), signal: ctrl.signal })
      .then((r) => {
        if (ctrl.signal.aborted) return;
        setResult(r);
        setLayout((l) => (l && r.layouts[l] ? l : pickLayout(r, scatterEmbedding)));
      })
      .catch((e) => {
        if (ctrl.signal.aborted) return;
        setRunError(errorMessage(e));
      })
      .finally(() => {
        if (abort.current === ctrl) {
          setRunning(false);
          setProgress(null);
        }
      });
  }, [datasetId, request, scatterEmbedding]);

  const rows = useMemo(() => (result ? nhoodRows(result) : []), [result]);

  const select = useCallback(
    (ids: number[], what: string) => {
      if (!ids.length) return;
      miloCells(datasetId, ids)
        .then(({ cells }) => {
          setSelection(new Set(cells));
          setPicked(`Selected ${cells.length.toLocaleString()} cells (${what}).`);
        })
        .catch((e) => setPicked(errorMessage(e)));
    },
    [datasetId, setSelection],
  );
  const pickNhood = useCallback(
    (id: number) => {
      const n = rows[id];
      if (n) select([id], `neighbourhood ${id}${result?.label_key ? `, ${n.label}` : ""}`);
    },
    [rows, result, select],
  );

  const exportCsv = useCallback(() => {
    if (!result) return;
    const { headers, rows: csvRows } = miloCsv(result);
    downloadCsv(`milo_${result.condition_key}_${result.conditions[1]}_vs_${result.conditions[0]}_${datasetId}.csv`, headers, csvRows);
  }, [result, datasetId]);

  if (!dataset) {
    return (
      <div className="flex items-center justify-center gap-2 py-8 text-sm text-slate-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Loading columns…
      </div>
    );
  }

  const up = result ? shifted(rows, "up") : [];
  const down = result ? shifted(rows, "down") : [];
  const stale = result && plan && (result.condition_key !== plan.condition_key ||
    result.conditions.join() !== plan.conditions.join() || result.sample_key !== plan.sample_key ||
    result.formula !== plan.formula || result.label_key !== (labelKey || null));

  return (
    <div className="space-y-4">
      <div className="space-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Conditions</span>
            <select aria-label="Milo condition column" value={conditionKey}
              onChange={(e) => setConditionKey(e.target.value)} className={selectCls}>
              <option value="">Select a column…</option>
              {conditionOptions.map((c) => (
                <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>
              ))}
            </select>
          </label>
          {conditionValues.length > 0 && (
            <>
              <label className="text-xs font-medium text-slate-500">
                <span className="mb-1 block">Reference</span>
                <select aria-label="Milo reference" value={reference}
                  onChange={(e) => setReference(e.target.value)} className={selectCls}>
                  {conditionValues.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
              <label className="text-xs font-medium text-slate-500">
                <span className="mb-1 block">Tested</span>
                <select aria-label="Milo tested condition" value={tested}
                  onChange={(e) => setTested(e.target.value)} className={selectCls}>
                  {conditionValues.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            </>
          )}
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Samples (replicates)</span>
            <select aria-label="Milo sample column" value={sampleKey}
              onChange={(e) => setSampleKey(e.target.value)} className={selectCls}>
              <option value="">
                {!sampleKey && plan?.sample_key_detected ? `Auto (${prettyObsLabel(plan.sample_key)})` : "Auto"}
              </option>
              {sampleOptions.map((c) => (
                <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Design</span>
            <select aria-label="Milo design" value={pairing}
              title="Paired adds the donor to the model: each donor is compared with itself"
              onChange={(e) => setPairing(e.target.value as "auto" | "paired" | "unpaired")} className={selectCls}>
              <option value="auto">{pairing === "auto" && plan ? `Auto (${plan.design})` : "Auto"}</option>
              <option value="paired">Paired by sample</option>
              <option value="unpaired">Unpaired</option>
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Covariate</span>
            <select aria-label="Milo covariate" value={covariate}
              title="A per-sample column (batch, sex, site) to adjust for"
              onChange={(e) => setCovariate(e.target.value)} className={selectCls}>
              <option value="">None</option>
              {covariateOptions.map((c) => (
                <option key={c.name} value={c.name}>{prettyObsLabel(c.name)}</option>
              ))}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Label by</span>
            <select aria-label="Milo label column" value={labelKey}
              onChange={(e) => setLabelKey(e.target.value)} className={selectCls}>
              <option value="">(none)</option>
              {labelOptions.map((c) => <option key={c} value={c}>{prettyObsLabel(c)}</option>)}
            </select>
          </label>
          <label className="text-xs font-medium text-slate-500">
            <span className="mb-1 block">Index cells</span>
            <select aria-label="Milo sampled share" value={prop == null ? "" : String(prop)}
              title="Share of cells sampled as neighbourhood centres (Milo: 10%, 5% above 30k cells)"
              onChange={(e) => setProp(e.target.value ? Number(e.target.value) : null)} className={selectCls}>
              <option value="">{plan && prop == null ? `Auto (${Math.round(plan.prop * 100)}%)` : "Auto"}</option>
              {PROPS.map((p) => <option key={p} value={p}>{Math.round(p * 100)}%</option>)}
            </select>
          </label>
          <button
            onClick={run}
            disabled={!plan || running}
            className="ml-auto flex items-center gap-1.5 rounded-lg bg-primary px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Milo
          </button>
        </div>
        {plan && (
          <p className="text-xs text-slate-500" data-testid="milo-plan">
            {plan.n_cells.toLocaleString()} cells in{" "}
            {plan.conditions.map((k) => `${plan.n_samples[k]} ${k}`).join(" and ")} samples of{" "}
            {prettyObsLabel(plan.sample_key)}; {plan.formula} ({plan.design}). Up to{" "}
            {plan.n_nhoods_estimate.toLocaleString()} neighbourhoods on the {plan.representation} graph,
            about {Math.max(1, Math.round(plan.estimate_seconds))} s.
          </p>
        )}
      </div>

      {planError && <ErrorBox text={planError} />}
      {runError && <ErrorBox text={runError} />}
      {plan?.warnings.map((w) => !result && <WarnBox key={w} text={w} />)}
      {running && (
        <div className="flex items-center justify-center gap-2 py-6 text-sm text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" /> {progress}
        </div>
      )}

      {result && !running && (
        <>
          {result.warnings.map((w) => <WarnBox key={w} text={w} />)}
          {stale && (
            <p className="text-xs text-amber-700">The settings changed since this run: run Milo again to update.</p>
          )}
          <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <h4 className="text-sm font-semibold text-slate-700" data-testid="milo-headline">
                {result.n_significant} of {result.n_nhoods.toLocaleString()} neighbourhoods shift
                (spatial FDR &lt; {result.alpha}): {result.n_up} more abundant in {result.conditions[1]},{" "}
                {result.n_down} less
              </h4>
              <div className="ml-auto flex flex-wrap gap-1.5">
                <button className={btnCls} disabled={!up.length}
                  title={`Select the cells of neighbourhoods more abundant in ${result.conditions[1]}`}
                  onClick={() => select(up.map((r) => r.id), `${up.length} neighbourhoods up`)}>
                  <MousePointerClick className="h-3 w-3" /> Select up
                </button>
                <button className={btnCls} disabled={!down.length}
                  title={`Select the cells of neighbourhoods less abundant in ${result.conditions[1]}`}
                  onClick={() => select(down.map((r) => r.id), `${down.length} neighbourhoods down`)}>
                  <MousePointerClick className="h-3 w-3" /> Select down
                </button>
                <button className={btnCls} onClick={() => askCopilot(miloQuestion(result))}
                  title="Ask the co-pilot what these results suggest">
                  <Sparkles className="h-3 w-3" /> Ask
                </button>
                <button className={btnCls} onClick={exportCsv} title="Export the neighbourhood table as CSV">
                  <Download className="h-3 w-3" /> CSV
                </button>
              </div>
            </div>
            {picked && (
              <p className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-600" data-testid="milo-picked">
                {picked}
                <button className="text-primary underline" onClick={() => setPanel("unified")}>
                  Show on the scatter
                </button>
              </p>
            )}
            <MiloBeeswarm result={result} rows={rows} datasetName={dataset.name} onPick={pickNhood} />
            <p className="mt-1 text-xs text-slate-500">
              One point per neighbourhood; click one to select its cells. {describeMilo(result)}
              {result.notes.map((n) => ` ${n}`)}
              {result.dropped_samples.length > 0 &&
                ` Left out (under 10 cells): ${result.dropped_samples.join(", ")}.`}
            </p>
          </div>

          {layout && (
            <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="mb-1 flex items-center gap-2 text-xs text-slate-500">
                <span>Layout</span>
                <select aria-label="Milo graph layout" value={layout} onChange={(e) => setLayout(e.target.value)}
                  className="rounded border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-700">
                  {Object.keys(result.layouts).map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
                {result.n_edges_total > result.edges.length && (
                  <span>
                    {result.edges.length.toLocaleString()} of {result.n_edges_total.toLocaleString()} edges
                    shown (most shared cells).
                  </span>
                )}
              </div>
              <MiloNhoodGraph result={result} rows={rows} datasetName={dataset.name} onPick={pickNhood} layout={layout} />
            </div>
          )}

          {result.label_key && result.labels.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50 text-left text-slate-600">
                    <th className="px-3 py-2 font-medium" title="Each neighbourhood's commonest label among its cells (Mixed when under 70 % of them)">
                      {prettyObsLabel(result.label_key)}
                    </th>
                    <th className="px-3 py-2 text-right font-medium" title="Neighbourhoods with this label">Neighbourhoods</th>
                    <th className="px-3 py-2 text-right font-medium" title={`Of those, significant (spatial FDR < ${result.alpha}) with log2FC > 0: more abundant in ${result.conditions[1]}`}>
                      Up in {result.conditions[1]}
                    </th>
                    <th className="px-3 py-2 text-right font-medium" title={`Of those, significant (spatial FDR < ${result.alpha}) with log2FC < 0: less abundant in ${result.conditions[1]}`}>
                      Down
                    </th>
                    <th className="px-3 py-2 text-right font-medium" title={`Median DESeq2 log2 fold change (${result.conditions[1]} vs ${result.conditions[0]}) over all of the label's neighbourhoods, significant or not`}>
                      Median log2FC
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {[...result.labels].reverse().map((l) => (
                    <tr key={l.label} className={`border-b border-slate-50 ${l.n_up + l.n_down ? "bg-emerald-50/60" : ""}`}>
                      <td className="px-3 py-1.5 text-slate-800">{l.label}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">{l.n_nhoods}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">{l.n_up}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">{l.n_down}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums text-slate-600">
                        {l.median_logfc == null ? "–" : l.median_logfc.toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function ErrorBox({ text }: { text: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      <AlertCircle className="h-4 w-4 flex-shrink-0" />
      {text}
    </div>
  );
}

function WarnBox({ text }: { text: string }) {
  return (
    <div role="alert" className="flex gap-2 rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-sm text-amber-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}
