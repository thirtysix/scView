import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Loader2, Scale, Square } from "lucide-react";
import {
  BATCH_METHOD_LABELS,
  getCompareInfo,
  runCompare,
  type CompareInfo,
  type CompareMethod,
  type CompareResult,
  type CompareRow,
  type CompareStart,
} from "@/api/batchCompare";
import { Caption } from "./Caption";
import { InfoTip } from "./InfoTip";
import { errorMessage } from "@/lib/errorMessage";

function pct(x: number | null | undefined): string {
  return x == null ? "–" : `${Math.round(x * 100)}%`;
}

/**
 * "Compare methods": runs none / Harmony / Scanorama / BBKNN on the current PCA (nothing
 * saved), scores each on batch mixing and label conservation, and recommends one by a
 * stated rule. Applying the choice is the caller's normal run (`onApply`).
 */
export function BatchMethodCompare({
  datasetId,
  batchKey,
  labelColumn,
  onApply,
  applyLabel = "Use this method",
  disabled = false,
}: {
  datasetId: string;
  batchKey: string | null | undefined;
  labelColumn?: string | null;
  onApply?: (method: CompareMethod) => void;
  applyLabel?: string;
  disabled?: boolean;
}) {
  const [info, setInfo] = useState<CompareInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [start, setStart] = useState<CompareStart | null>(null);
  const [rows, setRows] = useState<CompareRow[]>([]);
  const [current, setCurrent] = useState<CompareMethod | null>(null);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    setInfo(null);
    setInfoError(null);
    setResult(null);
    setRows([]);
    setStart(null);
    if (!batchKey) return;
    let cancelled = false;
    getCompareInfo(datasetId, batchKey, labelColumn)
      .then((i) => !cancelled && setInfo(i))
      .catch((e) => !cancelled && setInfoError(errorMessage(e)));
    return () => {
      cancelled = true;
    };
  }, [datasetId, batchKey, labelColumn]);

  useEffect(() => () => abort.current?.abort(), []);

  if (!batchKey) {
    return (
      <p className="text-[11px] text-slate-400" data-testid="batch-compare">
        Pick a batch column to compare correction methods.
      </p>
    );
  }

  const run = async () => {
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setError(null);
    setRows([]);
    setResult(null);
    setStart(null);
    try {
      await runCompare(
        datasetId,
        { batch_key: batchKey, label_column: labelColumn ?? null },
        {
          onStart: setStart,
          onMethodStart: (m) => setCurrent(m),
          onMethod: (row) => setRows((r) => [...r, row]),
          onResult: setResult,
        },
        ctrl.signal,
      );
    } catch (e) {
      if (!ctrl.signal.aborted) setError(errorMessage(e));
    } finally {
      setRunning(false);
      setCurrent(null);
    }
  };

  const cancel = () => abort.current?.abort();
  const shown = result?.rows ?? rows;
  const labelCol = result?.label_column ?? start?.label_column ?? info?.label_column ?? null;

  return (
    <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50/60 p-3" data-testid="batch-compare">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-700">Compare methods</span>
        <InfoTip width={320} label="About the comparison">
          Each method runs on the current PCA, without reprocessing or saving. Mixing: how many
          batches a cell&apos;s {start?.k ?? 15} nearest neighbours come from (iLISI), as a share of
          perfect mixing. Kept apart: how often those neighbours share the cell&apos;s label. BBKNN
          makes a neighbour graph, not an embedding, so its neighbours are each cell&apos;s strongest
          graph links.
        </InfoTip>
        {info?.available && !running && (
          <button
            type="button"
            onClick={run}
            disabled={disabled}
            className="ml-auto inline-flex items-center gap-1.5 rounded bg-primary px-2.5 py-1 text-xs font-medium text-white hover:bg-primary/90 disabled:opacity-50"
          >
            <Scale className="h-3.5 w-3.5" /> Compare
            {info.estimate_seconds != null && (
              <span
                className="text-white/80"
                title={info.first_run ? "The first comparison in a session compiles BBKNN's graph code (about 16 s of this)" : undefined}
              >
                (~{Math.max(1, Math.round(info.estimate_seconds))} s{info.first_run ? ", first run" : ""})
              </span>
            )}
          </button>
        )}
        {running && (
          <button
            type="button"
            onClick={cancel}
            className="ml-auto inline-flex items-center gap-1.5 rounded border border-slate-300 bg-white px-2.5 py-1 text-xs text-slate-600 hover:bg-slate-100"
          >
            <Square className="h-3 w-3" /> Cancel
          </button>
        )}
      </div>

      {infoError && <p className="text-xs text-red-600">{infoError}</p>}
      {info && !info.available && <p className="text-xs text-slate-500">{info.reason}</p>}
      {info?.available && (
        <p className="text-[11px] leading-snug text-slate-500">
          {info.background}{" "}
          <InfoTip width={320} label="About the benchmark">
            {info.background_tip}
          </InfoTip>
        </p>
      )}
      {info?.available && !labelCol && (
        <p className="text-[11px] text-slate-500">
          No cell-type label column was found, so only mixing is scored.
        </p>
      )}

      {running && current && (
        <p className="flex items-center gap-1.5 text-xs text-slate-500">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Running {BATCH_METHOD_LABELS[current]}
          {start ? ` (${rows.length + 1} of ${start.methods.length})` : ""}…
        </p>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}

      {shown.length > 0 && (
        <table className="w-full text-xs" data-testid="batch-compare-table">
          <thead>
            <tr className="text-left text-[11px] text-slate-500">
              <th className="py-1 pr-2 font-medium">Method</th>
              <th className="py-1 pr-2 font-medium" title="iLISI as a share of perfect mixing">
                Mixing
              </th>
              <th className="py-1 pr-2 font-medium" title="Share of neighbours with the cell's label">
                Kept apart
              </th>
              <th className="py-1 font-medium" title="Seconds this method took here">
                Time
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr
                key={r.method}
                className={`border-t border-slate-200 ${result?.recommended === r.method ? "bg-emerald-50" : ""}`}
              >
                <td className="py-1 pr-2 font-medium text-slate-700">
                  {r.label}
                  {result?.recommended === r.method && (
                    <span className="ml-1 rounded bg-emerald-100 px-1 text-[10px] text-emerald-700">recommended</span>
                  )}
                </td>
                {r.error ? (
                  <td colSpan={2} className="py-1 pr-2 text-red-600">
                    {r.error}
                  </td>
                ) : (
                  <>
                    <td className="py-1 pr-2 text-slate-600">{pct(r.mixing_ratio)}</td>
                    <td className="py-1 pr-2 text-slate-600">
                      {pct(r.conservation)}
                      {r.overcorrection && <AlertTriangle className="ml-1 inline h-3 w-3 text-amber-500" />}
                    </td>
                  </>
                )}
                <td className="py-1 text-slate-500">{r.seconds.toFixed(1)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {start && (
        <Caption n={[{ value: start.n_scored, unit: "cells scored", of: start.n_cells }]}>
          {start.n_batches} batches of &lsquo;{start.batch_key}&rsquo;
          {labelCol ? `, labels from '${labelCol}'` : ""}; {start.k} neighbours per cell.
        </Caption>
      )}

      {result && (
        <div className="space-y-1">
          <p className="text-xs text-slate-700" data-testid="batch-compare-reason">
            {result.reason}
          </p>
          {result.warnings.map((w) => (
            <p key={w} className="flex items-start gap-1.5 text-xs text-amber-700">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" /> {w}
            </p>
          ))}
          {result.notes.map((n) => (
            <p key={n} className="text-[11px] text-slate-500">
              {n}
            </p>
          ))}
          {result.recommended && onApply && (
            <button
              type="button"
              onClick={() => onApply(result.recommended as CompareMethod)}
              disabled={disabled}
              className="rounded border border-emerald-300 bg-white px-2.5 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50 disabled:opacity-50"
            >
              {applyLabel}: {BATCH_METHOD_LABELS[result.recommended]}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

