import { useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, Download, Lasso, FlaskConical, ArrowLeftRight, Sparkles } from "lucide-react";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useViewStore } from "@/stores/viewStore";
import { computeDE, computeDECompare, type DEResponse } from "@/api/de";
import { VolcanoPlot } from "@/components/plots/VolcanoPlot";
import { DEEnrichment } from "@/components/unified/DEEnrichment";
import { downloadCsv } from "@/lib/csv";
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

type Mode = "selection" | "groups";
type Test = "wilcoxon" | "pseudobulk";

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
  onScoreGeneSet: (scores: Float32Array, name: string) => void;
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

  const [mode, setMode] = useState<Mode>("selection");
  const [result, setResult] = useState<DEResponse | null>(null);
  // Display name of the comparison group ("rest" for selection-vs-rest).
  const [refLabel, setRefLabel] = useState("rest");
  // What the result compares, for questions to the co-pilot.
  const [comparison, setComparison] = useState<DEComparison | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fcThreshold, setFcThreshold] = useState(1);
  const [pThreshold] = useState(0.05);

  // Between-groups pickers.
  const splitCols = useMemo(() => groupColumns(dataset?.obs_columns, 50), [dataset]);
  const withinCols = useMemo(() => groupColumns(dataset?.obs_columns, 100), [dataset]);
  const [splitBy, setSplitBy] = useState("");
  const [groupA, setGroupA] = useState("");
  const [groupB, setGroupB] = useState("");
  const [withinCol, setWithinCol] = useState("");
  const [withinVal, setWithinVal] = useState("");
  const [test, setTest] = useState<Test>("wilcoxon");
  const [replicateKey, setReplicateKey] = useState("");

  const donorCols = useMemo(
    () => replicateColumns(dataset?.obs_columns, [splitBy, withinCol]),
    [dataset, splitBy, withinCol],
  );
  // Keep a valid donor column picked, pre-selecting the most donor-like name.
  useEffect(() => {
    if (!donorCols.some((c) => c.name === replicateKey)) {
      setReplicateKey(suggestReplicateColumn(donorCols));
    }
  }, [donorCols, replicateKey]);

  const splitValues = splitCols.find((c) => c.name === splitBy)?.values ?? [];
  const withinValues = withinCols.find((c) => c.name === withinCol)?.values ?? [];

  // Reset the pickers when the dataset changes; default to the joint `dataset` column.
  useEffect(() => {
    setResult(null);
    setSplitBy(defaultSplit(splitCols));
    setWithinCol("");
    setWithinVal("");
  }, [datasetId, splitCols]);

  useEffect(() => {
    const vals = splitCols.find((c) => c.name === splitBy)?.values ?? [];
    setGroupA(vals[1] ?? vals[0] ?? "");
    setGroupB(vals[0] ?? "");
  }, [splitBy, splitCols]);

  // A cluster / cell type clicked in the legend becomes the "within" scope.
  useEffect(() => {
    if (!highlighted || highlighted.column === splitBy) return;
    if (withinCols.some((c) => c.name === highlighted.column)) {
      setWithinCol(highlighted.column);
      setWithinVal(highlighted.value);
    }
  }, [highlighted, splitBy, withinCols]);

  const nSelected = selected?.size ?? 0;

  const runDE = useCallback(async () => {
    if (!datasetId || !selected || selected.size < 3) return;
    setLoading(true);
    setError(null);
    try {
      const label = highlighted?.value ?? "selection";
      const res = await computeDE(datasetId, Array.from(selected), label);
      setResult(res);
      setRefLabel("rest");
      setComparison({ a: label, b: "rest", betweenGroups: false });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, selected, highlighted]);

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
        ...(test === "pseudobulk" ? { method: "pseudobulk", replicate_key: replicateKey } : {}),
      });
      setResult(res);
      setRefLabel(groupB);
      setComparison({
        a: groupA,
        b: groupB,
        scope: scoped ? `${withinCol} = ${withinVal}` : undefined,
        betweenGroups: true,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [datasetId, splitBy, groupA, groupB, withinCol, withinVal, test, replicateKey]);

  // Significant genes, most significant first, for the table.
  const sigGenes = useMemo(() => {
    if (!result) return [];
    return result.genes
      .filter((g) => isSignificant(g, pThreshold, fcThreshold))
      .sort((a, b) => a.pval_adj - b.pval_adj || Math.abs(b.logfoldchange) - Math.abs(a.logfoldchange));
  }, [result, pThreshold, fcThreshold]);

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
      `de_${result.method === "pseudobulk" ? "pseudobulk_" : ""}${result.label}_${datasetId}.csv`,
      ["gene", "log2_fold_change", "p_value", "p_value_adjusted"],
      result.genes.map((g) => [
        g.gene,
        g.logfoldchange.toFixed(4),
        g.pval.toExponential(4),
        g.pval_adj.toExponential(4),
      ]),
    );
  }, [result, datasetId]);

  if (!datasetId) {
    return <div className="text-xs text-slate-400">No dataset loaded.</div>;
  }

  const selectCls =
    "min-w-0 flex-1 rounded border border-slate-300 bg-white px-1.5 py-1 text-xs text-slate-700";

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
                  onClick={() => {
                    setGroupA(groupB);
                    setGroupB(groupA);
                  }}
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
                  onChange={(e) => {
                    setWithinCol(e.target.value);
                    setWithinVal(withinCols.find((c) => c.name === e.target.value)?.values?.[0] ?? "");
                  }}
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
                      ["wilcoxon", "Cell-level (Wilcoxon)"],
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
                  <p className="text-[10px] leading-snug text-slate-500">
                    Sums each donor&apos;s raw counts per group and tests donors, not cells, with
                    DESeq2, so p-values allow for donor-to-donor variation. Needs raw counts and at
                    least 2 donors per group with 10+ cells each.
                    {loading && " Fitting one model per gene can take up to a minute."}
                  </p>
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
      <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs">
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
      </div>
      )}

      {error && (
        <div className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">{error}</div>
      )}

      {result && (
        <>
          {/* Threshold + export */}
          <div className="flex items-center gap-2 text-xs">
            <label className="whitespace-nowrap font-medium text-slate-500">|log2FC| &ge;</label>
            <input
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
          </div>

          <div className="text-[10px] text-slate-400">
            {result.label} (n={result.n_selected.toLocaleString()}) vs {refLabel} (n=
            {result.n_rest.toLocaleString()}) · {sigGenes.length.toLocaleString()} significant of{" "}
            {result.genes.length.toLocaleString()} genes · click a point or row to overlay it
            {result.method === "pseudobulk" ? (
              <span className="mt-0.5 block">
                Pseudobulk DESeq2 (Wald test, design <code>{result.design}</code>): donors (
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
                Cell-level Wilcoxon: cells are treated as independent, so p-values are optimistic
                when the groups also differ by donor or sample (the pseudobulk test allows for
                donors).
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
          </div>

          {/* Volcano */}
          <div className="rounded-lg border border-slate-200">
            <VolcanoPlot
              genes={result.genes}
              fcThreshold={fcThreshold}
              pThreshold={pThreshold}
              onGeneClick={onGeneClick}
            />
          </div>

          {/* Top significant genes */}
          <div className="max-h-72 overflow-auto rounded-lg border border-slate-200">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-slate-200 bg-slate-50">
                  <th className="px-2 py-1.5 text-left font-medium text-slate-600">Gene</th>
                  <th className="px-2 py-1.5 text-right font-medium text-slate-600">log2FC</th>
                  <th className="px-2 py-1.5 text-right font-medium text-slate-600">Adj.P</th>
                </tr>
              </thead>
              <tbody>
                {sigGenes.slice(0, 200).map((g) => (
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
                      {formatPValue(g.pval_adj)}
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
        </>
      )}
    </div>
  );
}
