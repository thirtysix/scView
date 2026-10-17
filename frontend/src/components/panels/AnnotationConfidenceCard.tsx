import { useEffect, useMemo, useRef, useState } from "react";
import { BadgeCheck, Check, Loader2 } from "lucide-react";
import {
  getAnnotationConfidence,
  makeUnknownColumn,
  runAnnotationConfidence,
  type ConfidenceInfo,
  type ConfidenceResult,
} from "@/api/annotation";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import {
  CONFIDENCE_NOTE,
  confidenceColumns,
  confusedWithText,
  coverageText,
  defaultConfidenceColumn,
  rejectedShare,
  scoredBy,
  shrinkHint,
  sortLabelRows,
} from "@/lib/annotationConfidence";
import { prettyObsLabel } from "@/lib/formatting";
import { pct } from "@/lib/qcThresholds";
import { useDatasetStore } from "@/stores/datasetStore";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const ROWS = 8; // more behind "show all"

const FLAG_STYLES: { key: keyof ConfidenceResult["counts"]; label: string; className: string }[] = [
  { key: "confident", label: "confident", className: "bg-emerald-50 text-emerald-800" },
  { key: "ambiguous", label: "ambiguous", className: "bg-sky-50 text-sky-800" },
  { key: "unsure", label: "unsure", className: "bg-amber-50 text-amber-800" },
  { key: "doesn't fit", label: "doesn't fit", className: "bg-rose-50 text-rose-800" },
];

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
 * How well each cell's label fits (split conformal prediction, C058): pick a
 * label column and an error rate α; every cell gets the set of labels its
 * expression fits, and a flag (confident / ambiguous / unsure / doesn't fit),
 * written to obs columns to colour the embedding by. Shows held-out coverage
 * (what the guarantee is about) and which labels fit worst, and with what.
 */
export function AnnotationConfidenceCard({ datasetId }: { datasetId: string }) {
  const dataset = useDatasetStore((s) => s.currentDataset);
  const columns = useMemo(() => confidenceColumns(dataset), [dataset]);
  const [column, setColumn] = useState("");
  const [alphaText, setAlphaText] = useState("0.1");
  const [info, setInfo] = useState<ConfidenceInfo | null>(null);
  const [result, setResult] = useState<ConfidenceResult | null>(null);
  const [running, setRunning] = useState(false);
  const [copying, setCopying] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [allRows, setAllRows] = useState(false);
  const [scorer, setScorer] = useState<"pca" | "celltypist">("pca");
  const refresh = useDatasetRefresh();
  const request = useRef(0); // answers for an older column or dataset are dropped

  useEffect(() => {
    setColumn("");
    setResult(null);
    setInfo(null);
  }, [datasetId]);

  useEffect(() => {
    if (!columns.some((c) => c.name === column)) setColumn(defaultConfidenceColumn(dataset));
  }, [columns, column, dataset]);

  // The last check of the column, if there is one.
  useEffect(() => {
    const id = ++request.current;
    setResult(null);
    setError(null);
    setCopied(null);
    setAllRows(false);
    getAnnotationConfidence(datasetId, column || undefined)
      .then((i) => {
        if (id !== request.current) return;
        setInfo(i);
        setResult(i.result);
        if (i.result) setAlphaText(String(i.result.alpha));
        // A CellTypist column is scored by CellTypist's own probabilities when they were kept.
        const hasReference = !!(column && i.reference_columns?.[column]);
        const last = i.result?.classifier;
        setScorer(hasReference && (last === undefined || last === "celltypist") ? "celltypist" : "pca");
      })
      .catch((e) => {
        if (id === request.current) setError(errorText(e));
      });
  }, [datasetId, column]);

  const minAlpha = info?.min_alpha ?? 0.01;
  const maxAlpha = info?.max_alpha ?? 0.5;
  const alpha = Number(alphaText);
  const alphaOk = alphaText.trim() !== "" && isFinite(alpha) && alpha >= minAlpha && alpha <= maxAlpha;
  const referenceModel = column ? info?.reference_columns?.[column] : undefined;
  const byReference = scorer === "celltypist" && !!referenceModel;
  const note = result?.note ?? (byReference ? info?.reference_note : info?.note) ?? CONFIDENCE_NOTE;

  const run = async () => {
    const id = ++request.current;
    setRunning(true);
    setError(null);
    setCopied(null);
    try {
      const r = await runAnnotationConfidence(datasetId, column, alpha, byReference ? "celltypist" : undefined);
      await refresh(datasetId, r.columns.flag);
      const bump = useObsRevisionStore.getState().bump;
      for (const c of [r.columns.alt, r.unknown_column]) if (c) bump(datasetId, c);
      if (id === request.current) setResult(r);
    } catch (e) {
      if (id === request.current) setError(errorText(e));
    } finally {
      setRunning(false);
    }
  };

  const copyUnknown = async () => {
    if (!result) return;
    setCopying(true);
    setError(null);
    try {
      const r = await makeUnknownColumn(datasetId, result.column);
      await refresh(datasetId, r.unknown_column);
      setResult((prev) => (prev && prev.column === r.column ? { ...prev, unknown_column: r.unknown_column } : prev));
      setCopied(`${r.n_unknown.toLocaleString()} rejected cells set to Unknown in ${r.unknown_column}.`);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setCopying(false);
    }
  };

  const colourBy = (name: string) => {
    useSettingsStore.getState().setColorBy(name);
    useViewStore.getState().setPanel("unified");
  };

  const rows = result ? sortLabelRows(result.labels) : [];
  const shown = allRows ? rows : rows.slice(0, ROWS);
  const hint = result ? shrinkHint(result) : null;
  const total = result ? result.n_checked || 1 : 1;

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4" data-testid="annotation-confidence">
      <div className="mb-2 flex items-center gap-2">
        <BadgeCheck className="h-5 w-5 text-primary" />
        <h3 className="text-sm font-semibold text-slate-900">Annotation confidence</h3>
      </div>
      <p className="mb-1 text-xs text-slate-500">
        How well each cell&apos;s label fits: a classifier learns the labels from the PCA (or the CellTypist model
        that made the column scores them), and split conformal prediction gives every cell the set of labels that
        fit it at error rate α. A cell&apos;s own label is in its set for at least 1 − α of cells.
      </p>
      <p className="mb-3 text-xs font-medium text-slate-600">{note}</p>

      {info && !info.available ? (
        <p className="text-xs text-slate-500">{info.reason}</p>
      ) : columns.length === 0 ? (
        <p className="text-xs text-slate-500">
          No label column to check yet: annotate cell types or cluster first (a column with 2–100 labels).
        </p>
      ) : (
        <>
          <div className="mb-3 grid grid-cols-1 gap-3 sm:grid-cols-[1fr_auto]">
            <div>
              <label htmlFor="conf-column" className="mb-1 block text-xs font-medium text-slate-600">
                Labels to check
              </label>
              <select
                id="conf-column"
                value={column}
                onChange={(e) => setColumn(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
              >
                {columns.map((c) => (
                  <option key={c.name} value={c.name}>
                    {prettyObsLabel(c.name)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="conf-alpha" className="mb-1 block text-xs font-medium text-slate-600">
                Error rate α
              </label>
              <input
                id="conf-alpha"
                type="number"
                min={minAlpha}
                max={maxAlpha}
                step={0.01}
                value={alphaText}
                onChange={(e) => setAlphaText(e.target.value)}
                className="w-24 rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:border-primary focus:outline-none"
              />
            </div>
          </div>
          {referenceModel && (
            <div className="mb-3 text-xs">
              <span className="mr-2 font-medium text-slate-600">Score with</span>
              <div className="inline-flex rounded-lg border border-slate-200 p-0.5">
                {(
                  [
                    ["celltypist", `CellTypist (${referenceModel.replace(/\.pkl$/, "")})`],
                    ["pca", "Classifier on the PCA"],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setScorer(key)}
                    aria-pressed={scorer === key}
                    className={`rounded-md px-2.5 py-1 font-medium transition-colors ${
                      scorer === key ? "bg-primary text-white" : "text-slate-600 hover:bg-slate-50"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <p className="mt-1 text-[11px] text-slate-400">
                CellTypist&apos;s own call for each cell is independent of the clusters its labels were voted over,
                so it also catches a cluster that holds two cell types; its sets are larger.
              </p>
            </div>
          )}
          <p className="mb-3 text-[11px] text-slate-400">
            About α of all cells are rejected (unsure or doesn&apos;t fit) even when every label is right: read
            each label&apos;s rejected share against α. A smaller α rejects fewer cells and makes sets larger.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void run()}
              disabled={running || !column || !alphaOk}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <BadgeCheck className="h-4 w-4" />}
              Check labels
            </button>
            {!alphaOk && (
              <span className="text-[11px] text-amber-700">
                α must be between {minAlpha} and {maxAlpha}.
              </span>
            )}
          </div>
        </>
      )}

      {result && (
        <div className="mt-3 space-y-2 border-t border-slate-100 pt-3" data-testid="confidence-result">
          {result.stale && (
            <p className="rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
              The labels in {prettyObsLabel(result.column)} changed since this check ran; run it again.
            </p>
          )}
          <p className={`text-xs ${result.coverage.within_sampling_error ? "text-slate-700" : "text-amber-800"}`}>
            {result.coverage.within_sampling_error && <Check className="mr-1 inline h-3.5 w-3.5 text-emerald-600" />}
            <span className="font-medium">Coverage at α = {result.alpha}:</span> {coverageText(result)}
            {!result.coverage.within_sampling_error && " Below the target by more than sampling error."}
          </p>
          <div className="flex flex-wrap gap-1.5 text-[11px]">
            {FLAG_STYLES.map((f) => (
              <span key={f.key} className={`rounded px-1.5 py-0.5 ${f.className}`}>
                {f.label} {result.counts[f.key].toLocaleString()} ({pct(result.counts[f.key] / total)})
              </span>
            ))}
          </div>
          {hint && <p className="text-[11px] text-slate-500">{hint}</p>}
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="font-medium">Label</th>
                <th className="text-right font-medium">Cells</th>
                <th className="text-right font-medium">Confident</th>
                <th className="text-right font-medium">Ambiguous</th>
                <th className="text-right font-medium" title="Unsure (fits no label) or doesn't fit (fits another label)">
                  Rejected
                </th>
                <th className="pl-3 font-medium" title="The labels its flagged cells are nearest to, as a share of its cells">
                  Nearest other labels
                </th>
              </tr>
            </thead>
            <tbody className="text-slate-600">
              {shown.map((r) => (
                <tr key={r.label}>
                  <td className="truncate pr-2">{r.label}</td>
                  <td className="text-right">{r.n_cells.toLocaleString()}</td>
                  <td className="text-right">{pct(r.confident)}</td>
                  <td className="text-right">{pct(r.ambiguous)}</td>
                  <td
                    className={`text-right ${rejectedShare(r) > 2 * result.alpha ? "font-semibold text-amber-700" : ""}`}
                  >
                    {pct(rejectedShare(r))}
                  </td>
                  <td className="pl-3 text-slate-500">{confusedWithText(r)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > ROWS && (
            <button type="button" onClick={() => setAllRows((v) => !v)} className="text-[11px] text-primary hover:underline">
              {allRows ? "Show fewer" : `Show all ${rows.length} labels`}
            </button>
          )}
          {result.labels_not_checked.length > 0 && (
            <p className="text-[11px] text-slate-500">
              Not checked (fewer than {info?.min_label_cells ?? 20} cells):{" "}
              {result.labels_not_checked.map((l) => `${l.label} (${l.n_cells})`).join(", ")}.
            </p>
          )}
          <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 pt-2">
            <button
              type="button"
              onClick={() => colourBy(result.columns.flag)}
              className="text-[11px] text-primary hover:underline"
              title={`Colour the Unified View scatter by ${result.columns.flag}`}
            >
              Show flags on the embedding
            </button>
            <button
              type="button"
              onClick={() => void copyUnknown()}
              disabled={copying || result.stale}
              className="rounded border border-slate-200 px-2.5 py-1 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
              title="A copy of the label column with every rejected cell set to Unknown; the column itself is not changed"
            >
              {copying ? "Saving…" : result.unknown_column ? "Update the Unknown copy" : "Copy with rejected cells as Unknown"}
            </button>
            {result.unknown_column && (
              <button
                type="button"
                onClick={() => colourBy(result.unknown_column ?? "")}
                className="text-[11px] text-primary hover:underline"
              >
                Colour by the copy
              </button>
            )}
          </div>
          {copied && <p className="text-[11px] text-emerald-700">{copied}</p>}
          <p className="text-[11px] text-slate-400">
            {scoredBy(result)}; LAC score; the most probable label is the cell&apos;s own for {pct(result.accuracy)} of
            cells; {result.elapsed_seconds} s. Columns: {result.columns.flag}, {result.columns.fit} (own label&apos;s
            conformal p-value), {result.columns.set_size}, {result.columns.alt} (nearest other label).
          </p>
        </div>
      )}
      {error && (
        <p role="alert" className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
          {error}
        </p>
      )}
    </div>
  );
}
