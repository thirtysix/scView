import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, AlertTriangle, Loader2 } from "lucide-react";
import { getPerDonorExpression, type DonorMethod, type PerDonorExpression as Result } from "@/api/donors";
import { DonorBoxPlot } from "@/components/plots/DonorBoxPlot";
import { Help } from "@/components/common/Help";
import { helpText } from "@/lib/help";
import { useCategoryColors } from "@/hooks/useCategoryColors";
import { rgbCss } from "@/lib/categoryColors";
import { conditionColumns, suggestConditionColumn } from "@/lib/composition";
import { replicateColumns } from "@/lib/de";
import { prettyObsLabel } from "@/lib/formatting";
import { useDatasetStore } from "@/stores/datasetStore";
import { errorMessage } from "@/lib/errorMessage";

interface Props {
  datasetId: string;
  /** The gene on the scatter, or null (a score, a protein, nothing). */
  gene: string | null;
  /** Why there is no gene, shown instead of the plot. */
  noGeneReason?: string;
  groupColumn: string;
  height?: number;
}

const selCls = "rounded border border-slate-300 bg-white px-1 py-0.5 text-[11px] text-slate-700";
const fmt = (v: number) => v.toFixed(2);

const METHOD_HELP: Record<DonorMethod, string> = {
  mean:
    "Mean of the log-normalised expression over each donor's cells in the group: the violin's own " +
    "scale, every cell one vote, no counts needed.",
  cpm:
    "log2(CPM + 1) of each donor's summed raw counts in the group: what pseudobulk DE (limma-voom or " +
    "DESeq2 on donor sums) works from. Deeper cells weigh more.",
};

/**
 * The violin's "Per donor" mode: a gene's pseudobulk value per donor within
 * each group, box + strip, split by condition (GET /expression/per-donor).
 */
export function PerDonorExpression({ datasetId, gene, noGeneReason, groupColumn, height = 260 }: Props) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const cols = useMemo(() => dataset?.obs_columns ?? [], [dataset]);
  const conditionOptions = useMemo(() => conditionColumns(cols, [groupColumn]), [cols, groupColumn]);
  const [condition, setCondition] = useState<string | null>(null); // null: suggested
  const [sample, setSample] = useState(""); // "": detected by the backend
  const [method, setMethod] = useState<DonorMethod>("mean");
  // The two levels to test when the condition has more than two; "all": every level, no test;
  // null: the backend's default (its two levels besides "Unassigned", if that leaves two).
  const [pick, setPick] = useState<{ a: string; ref: string } | "all" | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const conditionKey = condition ?? suggestConditionColumn(conditionOptions);
  const sampleOptions = useMemo(
    () => replicateColumns(cols, [groupColumn, conditionKey]),
    [cols, groupColumn, conditionKey],
  );

  useEffect(() => {
    if (!gene || !groupColumn) {
      setResult(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    getPerDonorExpression(datasetId, {
      gene,
      groupby: groupColumn,
      ...(sample ? { sample } : {}),
      ...(conditionKey ? { condition: conditionKey } : {}),
      ...(conditionKey && pick && pick !== "all" ? { conditions: [pick.a, pick.ref], reference: pick.ref } : {}),
      method,
    })
      .then((r) => !cancelled && setResult(r))
      .catch((e) => {
        if (cancelled) return;
        setResult(null);
        setError(errorMessage(e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, gene, groupColumn, conditionKey, sample, method, pick]);

  const conditionColor = useCategoryColors(result?.condition_key ?? conditionKey);
  const groupColor = useCategoryColors(groupColumn);
  const colorOf = useCallback(
    (k: string, kind: "condition" | "group") => rgbCss(kind === "condition" ? conditionColor(k) : groupColor(k)),
    [conditionColor, groupColor],
  );

  // More than two levels: the plot shows the two tested ones (with their test), or every level.
  const levels = useMemo(() => (result?.conditions ?? []).filter((c) => c !== ""), [result]);
  const tested = pick === "all" ? null : (result?.tested_conditions ?? null);
  const shown = useMemo(() => {
    if (!result || levels.length <= 2 || !tested) return { conditions: result?.conditions ?? [], points: result?.points ?? [] };
    return { conditions: tested, points: result.points.filter((p) => tested.includes(p.condition)) };
  }, [result, levels, tested]);
  const pickA = pick === "all" ? "" : pick ? pick.a : (tested?.[1] ?? "");
  const pickRef = pick && pick !== "all" ? pick.ref : (tested?.[0] ?? "");
  const choosePair = (a: string, ref: string) => {
    if (!a) return setPick("all");
    setPick({ a, ref: ref && ref !== a ? ref : (levels.find((l) => l !== a) ?? "") });
  };

  if (!gene) {
    return (
      <div className="py-6 text-center text-xs text-slate-400">
        {noGeneReason ?? "Select a gene to see its value per donor."}
      </div>
    );
  }

  const label = method === "mean" ? "mean log-norm expression" : "log2(CPM + 1)";
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-slate-500">
        <label className="flex items-center gap-1">
          <Help id="perdonor.split-by">Split by</Help>
          <select
            aria-label="Condition column"
            value={conditionKey}
            onChange={(e) => {
              setCondition(e.target.value);
              setPick(null);
            }}
            className={selCls}
          >
            <option value="">None</option>
            {conditionOptions.map((c) => (
              <option key={c.name} value={c.name}>
                {prettyObsLabel(c.name)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <Help id="perdonor.donor">Donor</Help>
          <select aria-label="Donor column" value={sample} onChange={(e) => setSample(e.target.value)} className={selCls}>
            <option value="">
              {!sample && result?.sample_key_detected && result.sample_key
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
        <label className="flex items-center gap-1">
          <Help id="perdonor.value">Value</Help>
          <select
            aria-label="Pseudobulk value"
            title={METHOD_HELP[method]}
            value={method}
            onChange={(e) => setMethod(e.target.value as DonorMethod)}
            className={selCls}
          >
            <option value="mean">Mean log-norm (default)</option>
            <option value="cpm">log2 CPM of summed counts</option>
          </select>
        </label>
        {levels.length > 2 && (
          <span className="flex items-center gap-1" title="A donor-level test compares two values of the condition; the others are left out of it.">
            Compare
            <select aria-label="Tested condition" value={pickA} onChange={(e) => choosePair(e.target.value, pickRef)} className={selCls}>
              <option value="">All values (no test)</option>
              {levels.map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
            {pickA && (
              <>
                vs
                <select aria-label="Reference condition" value={pickRef} onChange={(e) => choosePair(pickA, e.target.value)} className={selCls}>
                  {levels
                    .filter((l) => l !== pickA)
                    .map((l) => (
                      <option key={l} value={l}>
                        {l}
                      </option>
                    ))}
                </select>
              </>
            )}
          </span>
        )}
        {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
      </div>

      {error && (
        <div className="flex items-center gap-1.5 rounded border border-red-200 bg-red-50 px-2 py-1.5 text-xs text-red-700">
          <AlertCircle className="h-3.5 w-3.5 flex-shrink-0" />
          {error}
        </div>
      )}
      {result && !result.sample_key && (
        <div data-testid="needs-sample" className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          {result.message ??
            "Per-donor values need a sample or donor column (one value per person or sample)."}{" "}
          Pick one under Donor if the dataset has it under another name.
        </div>
      )}
      {result?.sample_key && result.points.length === 0 && (
        <div className="flex items-start gap-1.5 text-xs text-amber-800">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          No donor has {result.min_cells} or more cells in any group here.
        </div>
      )}
      {result?.sample_key && result.points.length > 0 && (
        <DonorBoxPlot
          points={shown.points}
          groups={result.groups}
          conditions={shown.conditions}
          sampleKey={prettyObsLabel(result.sample_key)}
          colorOf={colorOf}
          valueLabel={`${result.gene} ${label}`}
          format={fmt}
          plotName={`per_donor_${result.gene}_${result.group_key}`}
          datasetName={dataset?.name}
          height={height}
          captionTip={helpText("perdonor.plot")}
          design={result.design}
          pairedTests={result.paired_tests}
          unpairedTests={result.unpaired_tests}
          unit={method === "mean" ? "log-norm units" : "log2 CPM"}
          unitHelp={`Value: ${result.value_label}.`}
          note={
            `Value: ${result.value_label} (${result.source}).` +
            (result.n_dropped > 0
              ? ` ${result.n_dropped} donor-group sample${result.n_dropped === 1 ? "" : "s"} under ${result.min_cells} cells left out.`
              : "") +
            (shown.conditions.length < levels.length
              ? ` Showing ${shown.conditions.join(" and ")} of ${levels.length} values (${shown.conditions[1]} minus ${shown.conditions[0]}); pick All values under Compare for the rest.`
              : "")
          }
        />
      )}
    </div>
  );
}
