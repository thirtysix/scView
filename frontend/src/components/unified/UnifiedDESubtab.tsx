import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Download, Lasso, FlaskConical, ArrowLeftRight, Sparkles } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { useDESettingsStore, type DEMode, type DETest } from "@/stores/deSettingsStore";
import { useOnScreenStore } from "@/stores/onScreenStore";
import {
  computeDE,
  computeDECompare,
  computeDECompareMethods,
  computeDEMethods,
  type DEMethodsResponse,
  type DEResponse,
} from "@/api/de";
import type { OnScoreGeneSet } from "@/hooks/useGeneSetScoring";
import { VolcanoPlot } from "@/components/plots/VolcanoPlot";
import { DetectionHistogram } from "@/components/plots/DetectionHistogram";
import { DEEnrichment } from "@/components/unified/DEEnrichment";
import { DEActivity } from "@/components/unified/DEActivity";
import { ProteinNetwork } from "@/components/unified/ProteinNetwork";
import { DETranscriptomeSize } from "@/components/unified/DETranscriptomeSize";
import { DEMethodsAgreement } from "@/components/unified/DEMethodsAgreement";
import { DeCompositionNote } from "@/components/unified/DeCompositionNote";
import { DeOverlap } from "@/components/unified/DeOverlap";
import { PseudobulkDiagnostics, PseudobulkModelRows } from "@/components/unified/PseudobulkDiagnostics";
import { describePbModel, pbMethodLabel } from "@/lib/pseudobulk";
import { AddToGeneListMenu } from "@/components/common/AddToGeneListMenu";
import { CapNote } from "@/components/common/Caption";
import { downloadCsv } from "@/lib/csv";
import { describePrefilter, formatPct, genesPassing, minPctChoices } from "@/lib/deDetection";
import { describeSize } from "@/lib/transcriptomeSize";
import { compositionRule } from "@/lib/deComposition";
import {
  CELL_METHODS,
  MAST_NOTE,
  cellMethodInfo,
  hasPValues,
  listedGenes,
  methodLabel,
  type CellMethod,
} from "@/lib/deMethods";
import {
  deGeneQuestion,
  deResultQuestion,
  isSignificant,
  replicateColumns,
  suggestReplicateColumn,
  type DEComparison,
} from "@/lib/de";
import { formatPValue, prettyObsLabel } from "@/lib/formatting";
import type { ObsColumnInfo } from "@/api/types";
import { RequestError } from "@/components/common/RequestError";

type Mode = DEMode;
type Test = DETest;

/** Rows of the significant-genes table; the CSV export has every gene. */
const DE_TABLE_ROWS = 200;

/** Categorical columns with listed values, usable as a group picker. */
function groupColumns(cols: ObsColumnInfo[] | undefined, maxLevels: number): ObsColumnInfo[] {
  return (cols ?? []).filter(
    (c) => (c.values?.length ?? 0) >= 2 && (c.values?.length ?? 0) <= maxLevels,
  );
}

/** Default "compare" column: the joint dataset's origin, else a condition-like column. */
function defaultSplit(cols: ObsColumnInfo[]): string {
  const names = cols.map((c) => c.name);
  return (
    names.find((n) => n === "dataset") ??
    names.find((n) => /stim|condition|treat|group|disease|status|sample|batch/i.test(n)) ??
    names[0] ??
    ""
  );
}

interface Props {
  onGeneClick: (gene: string) => void;
  /** Scores an enriched gene set on the scatter (from the DE enrichment). */
  onScoreGeneSet: OnScoreGeneSet;
}

/**
 * Differential expression for the current cell selection (lasso or a clicked
 * cluster) vs the rest, shown as a volcano plot + a significant-gene table.
 */
export function UnifiedDESubtab({ onGeneClick, onScoreGeneSet }: Props) {
  const datasetId = useDatasetStore((s) => s.currentDatasetId);
  const dataset = useDatasetStore((s) => s.currentDataset);
  const selected = useSelectionStore((s) => s.selectedCellIndices);
  const highlighted = useSelectionStore((s) => s.highlightedGroup);
  const askCopilot = useViewStore((s) => s.askCopilot);

  const [result, setResult] = useState<DEResponse | null>(null);
  // Display name of the comparison group ("rest" for selection-vs-rest).
  const [refLabel, setRefLabel] = useState("rest");
  // What the result compares, for questions to the co-pilot.
  const [comparison, setComparison] = useState<DEComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pThreshold] = useState(0.05);
  // The co-pilot's "this result" (stores/onScreenStore.ts)
  const setOnScreenDE = useOnScreenStore((s) => s.setDE);
  useEffect(() => {
    setOnScreenDE(result?.result_id ? { id: result.result_id, label: result.label } : null);
  }, [result, setOnScreenDE]);
  useEffect(() => () => setOnScreenDE(null), [setOnScreenDE]);
  // "Compare methods": the same comparison under every cell-level test.
  const [methods, setMethods] = useState<{ res: DEMethodsResponse; a: string; b: string } | null>(null);
  const [methodsLoading, setMethodsLoading] = useState(false);
  const [methodsError, setMethodsError] = useState<string | null>(null);
  // The composition check's label column ("" = the active clustering); not a DE setting.
  const [compositionBy, setCompositionBy] = useState("");

  // The settings live in a store (they survive a tab switch; a session restores them).
  const mode = useDESettingsStore((s) => s.mode);
  const fcThreshold = useDESettingsStore((s) => s.fcThreshold);
  const splitBy = useDESettingsStore((s) => s.splitBy);
  const groupA = useDESettingsStore((s) => s.groupA);
  const groupB = useDESettingsStore((s) => s.groupB);
  const withinCol = useDESettingsStore((s) => s.withinCol);
  const withinVal = useDESettingsStore((s) => s.withinVal);
  const test = useDESettingsStore((s) => s.test);
  const replicateKey = useDESettingsStore((s) => s.replicateKey);
  const minPct = useDESettingsStore((s) => s.minPct);
  const cellMethod = useDESettingsStore((s) => s.cellMethod);
  const pbMethod = useDESettingsStore((s) => s.pbMethod);
  const ruvK = useDESettingsStore((s) => s.ruvK);
  const update = useDESettingsStore((s) => s.update);
  const setMode = (m: Mode) => update({ mode: m });
  const setFcThreshold = (v: number) => update({ fcThreshold: v });
  const setSplitBy = (v: string) => update({ splitBy: v });
  const setGroupA = (v: string) => update({ groupA: v });
  const setGroupB = (v: string) => update({ groupB: v });
  const setWithinVal = (v: string) => update({ withinVal: v });
  const setTest = (t: Test) => update({ test: t });
  const setReplicateKey = useCallback((v: string) => update({ replicateKey: v }), [update]);

  // Between-groups pickers.
  const splitCols = useMemo(() => groupColumns(dataset?.obs_columns, 50), [dataset]);
  const withinCols = useMemo(() => groupColumns(dataset?.obs_columns, 100), [dataset]);

  const donorCols = useMemo(
    () => replicateColumns(dataset?.obs_columns, [splitBy, withinCol]),
    [dataset, splitBy, withinCol],
  );
  // Keep a valid donor column picked, pre-selecting the most donor-like name.
  useEffect(() => {
    if (!donorCols.some((c) => c.name === replicateKey)) {
      setReplicateKey(suggestReplicateColumn(donorCols));
    }
  }, [donorCols, replicateKey, setReplicateKey]);

  const splitValues = splitCols.find((c) => c.name === splitBy)?.values ?? [];
  const withinValues = withinCols.find((c) => c.name === withinCol)?.values ?? [];

  useEffect(() => {
    setResult(null);
    setMethods(null);
    setCompositionBy("");
  }, [datasetId]);

  // Pickers this dataset can't use fall back to the defaults (the joint
  // `dataset` column first); a new dataset starts from DE_DEFAULTS.
  useEffect(() => {
    const s = useDESettingsStore.getState();
    if (!splitCols.some((c) => c.name === s.splitBy)) update({ splitBy: defaultSplit(splitCols) });
    if (s.withinCol && !withinCols.some((c) => c.name === s.withinCol && (c.values ?? []).includes(s.withinVal))) {
      update({ withinCol: "", withinVal: "" });
    }
  }, [splitCols, withinCols, update]);

  useEffect(() => {
    const vals = splitCols.find((c) => c.name === splitBy)?.values ?? [];
    const s = useDESettingsStore.getState();
    if (!vals.includes(s.groupA) || !vals.includes(s.groupB)) {
      update({ groupA: vals[1] ?? vals[0] ?? "", groupB: vals[0] ?? "" });
    }
  }, [splitBy, splitCols, update]);

  // A cluster / cell type clicked in the legend becomes the "within" scope,
  // once per highlight (a scope picked by hand since stays).
  useEffect(() => {
    if (!highlighted || highlighted.column === splitBy) return;
    const done = useDESettingsStore.getState().scopedHighlight;
    if (done && done.column === highlighted.column && done.value === highlighted.value) return;
    if (withinCols.some((c) => c.name === highlighted.column)) {
      update({ withinCol: highlighted.column, withinVal: highlighted.value, scopedHighlight: highlighted });
    }
  }, [highlighted, splitBy, withinCols, update]);

  const nSelected = selected?.size ?? 0;

  const runDE = useCallback(async () => {
    if (!datasetId || !selected || selected.size < 3) return;
    setLoading(true);
    setError(null);
    try {
      const label = highlighted?.value ?? "selection";
      const res = await computeDE(datasetId, Array.from(selected), label, minPct, cellMethod);
      setResult(res);
      setRefLabel("rest");
      setComparison({ a: label, b: "rest", betweenGroups: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, selected, highlighted, minPct, cellMethod]);

  const runCompare = useCallback(async () => {
    if (!datasetId || !splitBy || !groupA || !groupB) return;
    setLoading(true);
    setError(null);
    try {
      const scoped = withinCol && withinVal;
      const res = await computeDECompare(datasetId, {
        split_by: splitBy,
        group_a: groupA,
        group_b: groupB,
        ...(scoped ? { within_column: withinCol, within_value: withinVal } : {}),
        ...(test === "pseudobulk"
          ? {
              method: "pseudobulk",
              replicate_key: replicateKey,
              ...(pbMethod !== "deseq2" ? { pb_method: pbMethod } : {}),
              ...(ruvK ? { ruv_k: ruvK } : {}),
            }
          : { min_pct: minPct, ...(cellMethod !== "wilcoxon" ? { method: cellMethod } : {}) }),
        ...(compositionBy ? { composition_by: compositionBy } : {}),
      });
      setResult(res);
      setRefLabel(groupB);
      setComparison({
        a: groupA,
        b: groupB,
        scope: scoped ? `${withinCol} = ${withinVal}` : undefined,
        betweenGroups: true,
        column: splitBy,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, splitBy, groupA, groupB, withinCol, withinVal, test, replicateKey, pbMethod, ruvK, minPct,
      cellMethod, compositionBy]);

  // The current comparison under every cell-level test, and their agreement.
  const runMethods = useCallback(async () => {
    if (!datasetId) return;
    setMethodsLoading(true);
    setMethodsError(null);
    try {
      if (mode === "groups") {
        const scoped = withinCol && withinVal;
        const res = await computeDECompareMethods(datasetId, {
          split_by: splitBy,
          group_a: groupA,
          group_b: groupB,
          ...(scoped ? { within_column: withinCol, within_value: withinVal } : {}),
          min_pct: minPct,
        });
        setMethods({ res, a: res.label, b: groupB });
      } else {
        if (!selected || selected.size < 3) return;
        const label = highlighted?.value ?? "selection";
        const res = await computeDEMethods(datasetId, { indices: Array.from(selected), label, min_pct: minPct });
        setMethods({ res, a: label, b: "rest" });
      }
    } catch (e) {
      setMethodsError(e instanceof Error ? e.message : String(e));
    } finally {
      setMethodsLoading(false);
    }
  }, [datasetId, mode, splitBy, groupA, groupB, withinCol, withinVal, minPct, selected, highlighted]);

  // The genes the table lists: significant ones, most significant first; for a
  // test without p-values (logistic regression), the largest coefficients.
  const sigGenes = useMemo(
    () => (result ? listedGenes(result, pThreshold, fcThreshold, isSignificant) : []),
    [result, pThreshold, fcThreshold],
  );
  const withP = result ? hasPValues(result) : true;

  const askAboutGene = useCallback(
    (gene: DEResponse["genes"][number]) => {
      if (result && comparison) askCopilot(deGeneQuestion(comparison, result, gene));
    },
    [askCopilot, result, comparison],
  );

  const askAboutResult = useCallback(() => {
    if (result && comparison) askCopilot(deResultQuestion(comparison, result, sigGenes, fcThreshold));
  }, [askCopilot, result, comparison, sigGenes, fcThreshold]);

  const exportCSV = useCallback(() => {
    if (!result) return;
    downloadCsv(
      `de_${result.method && result.method !== "wilcoxon" ? `${result.method}_` : ""}${result.label}_${datasetId}.csv`,
      ["gene", "log2_fold_change", "p_value", "p_value_adjusted", "statistic"],
      // No p-value (logistic regression) stays an empty cell, never a number.
      result.genes.map((g) => [
        g.gene,
        g.logfoldchange.toFixed(4),
        g.pval == null ? null : g.pval.toExponential(4),
        g.pval_adj == null ? null : g.pval_adj.toExponential(4),
        g.stat == null ? null : g.stat.toPrecision(5),
      ]),
    );
  }, [result, datasetId]);

  if (!datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  const selectCls =
    "min-w-0 flex-1 rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700";

  // The cell-level test's gene prefilter (pseudobulk filters on summed counts).
  const minPctRow = (
    <div className="flex items-center gap-1.5">
      <span
        className="w-12 font-medium text-slate-500"
        title="Test only genes detected (> 0) in at least this share of the cells of either group: fewer tests, so a fairer multiple-testing correction, and no near-absent genes at the edges of the volcano."
      >
        Genes
      </span>
      <select
        aria-label="Genes tested"
        value={minPct}
        onChange={(e) => update({ minPct: Number(e.target.value) })}
        className={selectCls}
      >
        {minPctChoices(minPct).map((c) => (
          <option key={c.value} value={c.value}>
            {c.label}
          </option>
        ))}
      </select>
    </div>
  );
  const methodInfo = cellMethodInfo(cellMethod);
  const methodRow = (
    <div className="flex items-center gap-1.5">
      <span
        className="w-12 font-medium text-slate-500"
        title={`${CELL_METHODS.map((m) => `${m.label}: ${m.help}`).join("\n\n")}\n\n${MAST_NOTE}`}
      >
        Method
      </span>
      <select
        aria-label="Cell-level test"
        value={cellMethod}
        onChange={(e) => update({ cellMethod: e.target.value as CellMethod })}
        title={`${methodInfo.help} ${MAST_NOTE}`}
        className={selectCls}
      >
        {CELL_METHODS.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label}
            {m.hasPValues ? "" : " (no p-values)"}
          </option>
        ))}
        <option value="mast" disabled>
          MAST (R only, not available)
        </option>
      </select>
      <button
        onClick={runMethods}
        disabled={
          methodsLoading ||
          (mode === "groups" ? !groupA || !groupB || groupA === groupB : nSelected < 3)
        }
        title="Run this comparison under every cell-level test (Wilcoxon, Welch t-test, logistic regression) and show how far they agree"
        className="flex items-center gap-1 whitespace-nowrap rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
      >
        {methodsLoading && <Loader2 className="h-3 w-3 animate-spin" />}
        Compare methods
      </button>
    </div>
  );
  // Display names of the result's two sides.
  const labelA = comparison?.a ?? result?.label ?? "";
  const labelB = comparison?.b ?? refLabel;
  const shownMinPct = result?.min_pct ?? 0;
  const pendingMinPct =
    result?.detection && result.method !== "pseudobulk" && minPct !== shownMinPct ? minPct : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Mode */}
      <div className="flex rounded-md border border-slate-200 p-0.5 text-xs">
        {(
          [
            ["selection", "Selection vs rest"],
            ["groups", "Between groups"],
          ] as const
        ).map(([m, label]) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 rounded px-2 py-1 font-medium transition ${
              mode === m ? "bg-primary text-white" : "text-slate-500 hover:text-slate-700"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {mode === "groups" ? (
        <div className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs">
          {splitCols.length === 0 ? (
            <span className="text-slate-500">
              This dataset has no categorical column with 2–50 groups to compare.
            </span>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-12 font-medium text-slate-500">Compare</span>
                <select value={splitBy} onChange={(e) => setSplitBy(e.target.value)} className={selectCls}>
                  {splitCols.map((c) => (
                    <option key={c.name} value={c.name}>
                      {prettyObsLabel(c.name)}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-12 font-medium text-slate-500">Groups</span>
                <select value={groupA} onChange={(e) => setGroupA(e.target.value)} className={selectCls}>
                  {splitValues.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => update({ groupA: groupB, groupB: groupA })}
                  title="Swap groups"
                  className="rounded border border-slate-300 bg-white p-1 text-slate-500 hover:bg-slate-100"
                >
                  <ArrowLeftRight className="h-3 w-3" />
                </button>
                <select value={groupB} onChange={(e) => setGroupB(e.target.value)} className={selectCls}>
                  {splitValues.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-12 font-medium text-slate-500">Within</span>
                <select
                  value={withinCol}
                  onChange={(e) =>
                    update({
                      withinCol: e.target.value,
                      withinVal: withinCols.find((c) => c.name === e.target.value)?.values?.[0] ?? "",
                    })
                  }
                  className={selectCls}
                >
                  <option value="">All cells</option>
                  {withinCols
                    .filter((c) => c.name !== splitBy)
                    .map((c) => (
                      <option key={c.name} value={c.name}>
                        {prettyObsLabel(c.name)}
                      </option>
                    ))}
                </select>
                {withinCol && (
                  <select value={withinVal} onChange={(e) => setWithinVal(e.target.value)} className={selectCls}>
                    {withinValues.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                )}
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-12 font-medium text-slate-500">Test</span>
                <div className="flex min-w-0 flex-1 rounded border border-slate-300 bg-white p-0.5">
                  {(
                    [
                      ["wilcoxon", "Cell-level"],
                      ["pseudobulk", "Pseudobulk (donor-aware)"],
                    ] as const
                  ).map(([t, label]) => (
                    <button
                      key={t}
                      onClick={() => setTest(t)}
                      className={`flex-1 truncate rounded px-1.5 py-0.5 transition ${
                        test === t ? "bg-slate-200 font-medium text-slate-800" : "text-slate-500 hover:text-slate-700"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
              {test !== "pseudobulk" && methodRow}
              {test !== "pseudobulk" && minPctRow}
              {test === "pseudobulk" && (
                <>
                  <div className="flex items-center gap-1.5">
                    <span className="w-12 font-medium text-slate-500">Donor</span>
                    <select
                      value={replicateKey}
                      onChange={(e) => setReplicateKey(e.target.value)}
                      className={selectCls}
                    >
                      {!replicateKey && <option value="">Pick the donor / sample column</option>}
                      {donorCols.map((c) => (
                        <option key={c.name} value={c.name}>
                          {prettyObsLabel(c.name)} ({c.n_unique})
                        </option>
                      ))}
                    </select>
                  </div>
                  <PseudobulkModelRows
                    pbMethod={pbMethod}
                    ruvK={ruvK}
                    onChange={update}
                    selectCls={selectCls}
                  />
                  <p className="text-[10px] leading-snug text-slate-500">
                    Sums each donor&apos;s raw counts per group and tests donors, not cells, with{" "}
                    {pbMethodLabel(pbMethod)}, so p-values allow for donor-to-donor variation. Needs raw
                    counts and at least 2 donors per group with 10+ cells each.
                    {loading && pbMethod === "deseq2" && " Fitting one model per gene can take up to a minute."}
                  </p>
                  {result?.method === "pseudobulk" && (
                    <PseudobulkDiagnostics hetero={result.heteroscedasticity} ruv={result.ruv} />
                  )}
                </>
              )}
              <div className="flex items-center justify-between gap-2">
                <span className="text-slate-500">
                  Up = higher in <span className="font-semibold text-slate-700">{groupA}</span>
                </span>
                <button
                  onClick={runCompare}
                  disabled={
                    loading ||
                    !groupA ||
                    !groupB ||
                    groupA === groupB ||
                    (test === "pseudobulk" && !replicateKey)
                  }
                  className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 font-medium text-white transition hover:bg-primary/90 disabled:opacity-50"
                >
                  {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <FlaskConical className="h-3 w-3" />}
                  Compute DE
                </button>
              </div>
            </>
          )}
        </div>
      ) : (
      <div className="flex flex-col gap-2 rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs">
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
        {methodRow}
        {minPctRow}
      </div>
      )}

      {/* Each side's cluster/state mix, before the DE is run */}
      {mode === "groups" && splitCols.length > 0 && (
        <DeCompositionNote
          datasetId={datasetId}
          params={{
            split_by: splitBy,
            group_a: groupA,
            group_b: groupB,
            ...(withinCol && withinVal ? { within_column: withinCol, within_value: withinVal } : {}),
          }}
          columns={withinCols.map((c) => c.name)}
          column={compositionBy}
          onColumnChange={setCompositionBy}
          scopeColumns={withinCols.map((c) => c.name)}
          onCompareWithin={(col, group) => update({ withinCol: col, withinVal: group })}
        />
      )}

      <RequestError message={error} />
      {methodsError && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">
          Compare methods: {methodsError}
        </div>
      )}
      {methodsLoading && !methods && (
        <p className="text-[10px] text-slate-400">
          Running Wilcoxon, Welch t-test and logistic regression: Wilcoxon takes longest (about 10 s on 14k cells).
        </p>
      )}
      {methods && (
        <DEMethodsAgreement
          result={methods.res}
          labelA={methods.a}
          labelB={methods.b}
          csvName={`de_methods_${methods.res.label}_${datasetId}`}
          onGeneClick={onGeneClick}
        />
      )}

      {result && (
        <>
          {/* Threshold + export */}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <label className="whitespace-nowrap font-medium text-slate-500">|log2FC| &ge;</label>
            <input
              aria-label="Minimum |log2FC|"
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
              onClick={askAboutResult}
              title="Ask the co-pilot what these results suggest"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50 hover:text-primary"
            >
              <Sparkles className="h-3 w-3" />
              Ask
            </button>
            <button
              onClick={exportCSV}
              title="Export DE table as CSV"
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-slate-600 hover:bg-slate-50"
            >
              <Download className="h-3 w-3" />
              CSV
            </button>
            {(["up", "down"] as const).map((side) => {
              const higher = side === "up" ? labelA : labelB;
              const genes = sigGenes.filter((g) => (side === "up" ? g.logfoldchange > 0 : g.logfoldchange < 0));
              return (
                <AddToGeneListMenu
                  key={side}
                  label={side === "up" ? "Up" : "Down"}
                  genes={() => genes.map((g) => g.gene)}
                  defaultName={`DE: higher in ${higher}`}
                  source={`DE ${labelA} vs ${labelB}, higher in ${higher}`}
                  disabled={genes.length === 0}
                  title={`Add the ${genes.length} ${withP ? "significant" : "listed"} genes higher in ${higher} (table order) to a gene list`}
                />
              );
            })}
          </div>

          <div className="text-[10px] text-slate-400">
            {result.label} (n={result.n_selected.toLocaleString()}) vs {refLabel} (n=
            {result.n_rest.toLocaleString()}) · {methodLabel(result.method, result)} ·{" "}
            {sigGenes.length.toLocaleString()} {withP ? "significant" : "listed"} of{" "}
            {result.genes.length.toLocaleString()} genes · click a point or row to overlay it
            {!withP && (
              <span className="mt-0.5 block rounded border border-slate-200 bg-slate-50 px-1.5 py-1 text-slate-600">
                Logistic regression gives a coefficient per gene and no p-values, so nothing here is
                &ldquo;significant&rdquo;: the table lists genes past the |log2FC| cut-off whose
                coefficient has the same sign, largest first. Coefficients are joint (each given the
                other genes) and penalised; over-representation needs significant genes, GSEA ranks
                by the coefficient.
              </span>
            )}
            {result.method !== "pseudobulk" && describePrefilter(result) && (
              <span className="mt-0.5 block">{describePrefilter(result)}</span>
            )}
            {result.transcriptome_size && !result.transcriptome_size.caution && (
              <span className="mt-0.5 block">
                {describeSize(result.transcriptome_size, labelA, labelB)}
              </span>
            )}
            {result.method === "pseudobulk" ? (
              <span className="mt-0.5 block">
                {describePbModel(result)}: donors (
                <code>{result.replicate_key}</code>) {result.label.split(" (")[0]}{" "}
                {result.n_samples_a}, {refLabel} {result.n_samples_b}; counts summed from{" "}
                <code>{result.counts_source}</code>; {result.n_genes_tested?.toLocaleString()} genes
                tested.
                {(result.dropped_samples?.length ?? 0) > 0 &&
                  ` Left out (under 10 cells): ${result.dropped_samples!.join(", ")}.`}
              </span>
            ) : refLabel !== "rest" && (
              <span
                className={`mt-0.5 block ${
                  result.underreplicated
                    ? "rounded border border-amber-200 bg-amber-50 px-1.5 py-1 text-amber-800"
                    : ""
                }`}
              >
                Cell-level {methodLabel(result.method)}: cells are treated as independent
                {withP
                  ? ", so p-values are optimistic when the groups also differ by donor or sample"
                  : ", so donor-to-donor differences count as group differences"}{" "}
                (the pseudobulk test allows for donors).
                {result.replicate_key ? (
                  <>
                    {" "}
                    Donors (<code>{result.replicate_key}</code>): {result.label.split(" (")[0]}{" "}
                    {result.n_donors_a}, top donor{" "}
                    {Math.round((result.top_donor_share_a ?? 0) * 100)}%; {refLabel}{" "}
                    {result.n_donors_b}, top donor{" "}
                    {Math.round((result.top_donor_share_b ?? 0) * 100)}%.
                    {result.underreplicated &&
                      " Under-replicated: fewer than 2 donors or one donor dominates a side."}
                  </>
                ) : (
                  " No donor/sample column detected."
                )}
              </span>
            )}
            {result.confounding && (
              <span
                role="alert"
                className="mt-1 block rounded border border-red-200 bg-red-50 px-1.5 py-1 text-red-800"
              >
                Batch and comparison coincide: {result.confounding.message} Read these genes as
                &ldquo;{result.label.split(" (")[0]} vs {refLabel}, batch included&rdquo;, not as a
                pure {result.confounding.condition === result.confounding.batch
                  ? "biological" : `'${result.confounding.condition}'`} effect.
              </span>
            )}
            {result.composition?.warn && (
              <span
                role="alert"
                data-testid="de-composition-caution"
                className="mt-1 block rounded border border-red-200 bg-red-50 px-1.5 py-1 text-red-800"
                title={compositionRule(result.composition)}
              >
                Composition differs (by <code>{result.composition.column}</code>): {result.composition.message}
              </span>
            )}
          </div>

          {/* The two sides differ a lot in RNA content */}
          {result.transcriptome_size?.caution && (
            <DETranscriptomeSize
              size={result.transcriptome_size}
              labelA={labelA}
              labelB={labelB}
              groupColumn={comparison?.column}
            />
          )}

          {/* Gene detection, with the prefilter's cut-off */}
          {result.detection && result.detection.length > 1 && (
            <div className="rounded-lg border border-slate-200 px-2 pt-1.5 text-[10px] text-slate-500">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-medium text-slate-600">Gene detection</span>
                <span className="truncate">
                  genes by the larger share of cells they are detected in, of the two groups
                </span>
              </div>
              <DetectionHistogram
                detection={result.detection}
                minPct={shownMinPct}
                pending={pendingMinPct}
              />
              <p className="pb-1.5">
                {shownMinPct > 0
                  ? `Grey: left out (under ${formatPct(shownMinPct)}, dashed line). `
                  : "Every gene was tested. "}
                {pendingMinPct != null &&
                  `${pendingMinPct > 0 ? `At ${formatPct(pendingMinPct)} (dotted line)` : "With no cut-off"}, ${genesPassing(
                    result.detection,
                    pendingMinPct,
                  ).toLocaleString()} genes would be tested: compute again to apply.`}
              </p>
            </div>
          )}

          {/* Volcano */}
          <div className="rounded-lg border border-slate-200">
            <VolcanoPlot
              genes={result.genes}
              fcThreshold={fcThreshold}
              pThreshold={pThreshold}
              onGeneClick={onGeneClick}
              yMode={withP ? "p" : "coefficient"}
            />
          </div>

          {/* Top significant genes */}
          <CapNote
            shown={DE_TABLE_ROWS}
            total={sigGenes.length}
            by={withP ? "adjusted p, then |log2FC|" : "|coefficient|"}
            rest="the CSV has all"
          />
          <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="px-2 py-1.5 text-left font-medium text-slate-600" title="Gene symbol; click a row to colour the scatter by its expression">
                    Gene
                  </th>
                  <th
                    className="px-2 py-1.5 text-right font-medium text-slate-600"
                    title={
                      result.method === "pseudobulk"
                        ? `DESeq2 log2 fold change of the per-donor summed counts, ${result.label} over ${refLabel}`
                        : `log2 of the mean expression in ${result.label} over ${refLabel} (expm1 of the mean log-normalised values on each side, as scanpy's rank_genes_groups takes it)`
                    }
                  >
                    log2FC
                  </th>
                  <th
                    className="px-2 py-1.5 text-right font-medium text-slate-600"
                    title={
                      withP
                        ? `${methodLabel(result.method)} p-value, Benjamini-Hochberg adjusted over the ${result.genes.length.toLocaleString()} tested genes`
                        : `Logistic regression coefficient (penalised, fitted jointly over the tested genes): positive when the gene is higher in ${result.label}; no p-value`
                    }
                  >
                    {withP ? "Adj.P" : "Coef."}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sigGenes.slice(0, DE_TABLE_ROWS).map((g) => (
                  <tr
                    key={g.gene}
                    onClick={() => onGeneClick(g.gene)}
                    className="group cursor-pointer border-b border-slate-50 transition-colors hover:bg-blue-50"
                  >
                    <td className="px-2 py-1.5">
                      <div className="flex items-center justify-between gap-1">
                        <span className="font-mono font-medium text-slate-800">{g.gene}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            askAboutGene(g);
                          }}
                          title="Ask the co-pilot about this gene"
                          className="flex-shrink-0 text-slate-300 opacity-0 transition-opacity hover:text-primary group-hover:opacity-100"
                        >
                          <Sparkles className="h-3 w-3" />
                        </button>
                      </div>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      <span className={g.logfoldchange > 0 ? "text-red-600" : "text-blue-600"}>
                        {g.logfoldchange.toFixed(2)}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-slate-600">
                      {g.pval_adj != null ? formatPValue(g.pval_adj) : (g.stat ?? 0).toFixed(3)}
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

          {/* Significant up / down genes -> pathway enrichment */}
          <DEEnrichment
            datasetId={datasetId}
            result={result}
            labelA={comparison?.a ?? result.label}
            labelB={comparison?.b ?? refLabel}
            pThreshold={pThreshold}
            fcThreshold={fcThreshold}
            onScoreGeneSet={onScoreGeneSet}
            onGeneClick={onGeneClick}
          />

          {/* Every tested gene's statistic -> TF / pathway activity */}
          <DEActivity
            datasetId={datasetId}
            result={result}
            labelA={comparison?.a ?? result.label}
            labelB={comparison?.b ?? refLabel}
            onScore={onScoreGeneSet}
          />

          {/* Significant genes -> STRING protein interaction subnetwork */}
          {datasetId && result.result_id && (
            <ProteinNetwork
              datasetId={datasetId}
              sources={[{ value: `de:${result.result_id}`, label: result.label }]}
              showDirection
              onGeneClick={onGeneClick}
            />
          )}
        </>
      )}

      {/* The significant genes of several kept DE results, compared (UpSet) */}
      <DeOverlap
        datasetId={datasetId}
        latestResultId={result?.result_id}
        pThreshold={pThreshold}
        fcThreshold={fcThreshold}
        onGeneClick={onGeneClick}
      />
    </div>
  );
}
