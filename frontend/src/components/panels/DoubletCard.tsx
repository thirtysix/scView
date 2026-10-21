import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { applyDoubletThreshold, getDoublets, type DoubletSummary } from "@/api/qc";
import { Panel } from "@/components/common/Panel";
import { ThresholdHistogram, type RefLine } from "@/components/plots/ThresholdHistogram";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { pct, roundTo } from "@/lib/qcThresholds";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const STEP = 0.005; // threshold resolution on the 0-1 score
const DEBOUNCE_MS = 120;
const SAMPLE_ROWS = 8; // more behind "show all"

/** A score for display: Scrublet's own threshold has 17 digits. */
export function fmtScore(v: number): string {
  return String(Number(v.toFixed(4)));
}

/** The rule-of-thumb sentence under the histogram. */
export function expectedRateText(s: DoubletSummary): string {
  const sug = s.suggestion;
  const where = sug.column
    ? `from the cells in each of ${sug.samples.length} ${sug.column} samples`
    : "taking all cells as one channel (no sample column found)";
  return `Expected about ${pct(sug.rate)} doublets ${where}: a rule of thumb from 10x Genomics, ~0.8 % per 1,000 cells captured in one channel.`;
}

/**
 * The doublet-score histogram with a threshold to drag or type, how many cells it
 * calls doublets overall and per sample beside the rate their size suggests, and
 * Apply: the calls go into `predicted_doublet` (what Cell/Gene Filtering drops when
 * "Drop predicted doublets" is on), Scrublet's own call is kept and can be put back.
 */
export function DoubletCard({
  datasetId,
  refreshKey = 0,
  onApplied,
}: {
  datasetId: string;
  refreshKey?: number;
  onApplied?: () => void;
}) {
  const [summary, setSummary] = useState<DoubletSummary | null>(null);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [logY, setLogY] = useState(true);
  const [allSamples, setAllSamples] = useState(false);
  const refresh = useDatasetRefresh();
  const request = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchAt = useCallback(
    (t: number | null) => {
      const id = ++request.current;
      setLoading(true);
      getDoublets(datasetId, t)
        .then((s) => {
          if (id !== request.current) return; // a newer threshold was asked for
          setSummary(s);
          setError(null);
          if (t === null) {
            setThreshold(s.threshold ?? null);
            setText(s.threshold != null ? fmtScore(s.threshold) : "");
          }
        })
        .catch((e) => {
          if (id === request.current) setError(String(e?.message ?? e));
        })
        .finally(() => {
          if (id === request.current) setLoading(false);
        });
    },
    [datasetId],
  );

  useEffect(() => {
    setSummary(null);
    fetchAt(null);
    const pending = request;
    const debounce = timer;
    return () => {
      pending.current++; // answers still on their way are for the old dataset
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, [fetchAt, refreshKey]);

  const ask = useCallback(
    (t: number) => {
      if (timer.current) clearTimeout(timer.current);
      timer.current = setTimeout(() => fetchAt(t), DEBOUNCE_MS);
    },
    [fetchAt],
  );

  const commit = useCallback(
    (raw: number) => {
      const t = roundTo(raw, STEP);
      setThreshold(t);
      setText(fmtScore(t));
      ask(t);
    },
    [ask],
  );

  const scrublet = summary?.scrublet_threshold ?? null;
  const refLines = useMemo<RefLine[]>(
    () => (scrublet !== null ? [{ value: scrublet, label: "Scrublet", color: "#64748b", dash: "dot", showLabel: true }] : []),
    [scrublet],
  );
  const lines = useMemo(
    () => (threshold !== null ? [{ id: "t", value: threshold, label: "threshold" }] : []),
    [threshold],
  );

  if (!summary) {
    return error ? (
      <div className="rounded-xl border border-slate-200 bg-white p-3 text-xs text-slate-400">
        Doublet scores unavailable: {error}
      </div>
    ) : null;
  }
  if (!summary.available || !summary.hist) return null;

  const apply = async (t: number | null) => {
    setApplying(true);
    setError(null);
    try {
      const s = await applyDoubletThreshold(datasetId, t);
      setSummary(s);
      setThreshold(s.threshold ?? null);
      setText(s.threshold != null ? fmtScore(s.threshold) : "");
      await refresh(datasetId, "predicted_doublet");
      onApplied?.();
    } catch (e) {
      setError(String((e as Error)?.message ?? e));
    } finally {
      setApplying(false);
    }
  };

  const showOnEmbedding = () => {
    useSettingsStore.getState().setColorBy("predicted_doublet");
    useViewStore.getState().setPanel("unified");
  };

  const applied = summary.applied ?? null;
  const unchanged = applied !== null && threshold !== null && Math.abs(applied.threshold - threshold) < 1e-9;
  const samples = summary.samples ?? [];
  const shown = allSamples ? samples : samples.slice(0, SAMPLE_ROWS);

  return (
    <Panel
      title="Doublet score"
      actions={
        <label className="flex items-center gap-1 text-[10px] font-normal text-slate-500">
          <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} className="h-3 w-3" />
          log scale
        </label>
      }
    >
      <div className="space-y-2">
        <div className="h-[200px]">
          <ThresholdHistogram
            edges={summary.hist.bin_edges}
            counts={summary.hist.counts}
            lines={lines}
            refLines={refLines}
            logY={logY}
            barColor="#a78bfa"
            plotName="qc_doublet_score"
            onDrag={(_, v) => {
              const t = roundTo(v, STEP);
              setText(fmtScore(t));
              ask(t);
            }}
            onCommit={(_, v) => commit(v)}
          />
        </div>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
          <label htmlFor="doublet-threshold" className="font-medium">
            Threshold
          </label>
          <input
            id="doublet-threshold"
            type="number"
            min={0}
            max={1}
            step={STEP}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              const v = Number(e.target.value);
              if (e.target.value.trim() !== "" && isFinite(v) && v >= 0) {
                setThreshold(v);
                ask(v);
              }
            }}
            className="w-20 rounded border border-slate-200 px-1.5 py-0.5 text-xs"
          />
          {scrublet !== null && (
            <button
              type="button"
              onClick={() => {
                // Exactly Scrublet's, not rounded to the step: its own calls.
                setThreshold(scrublet);
                setText(fmtScore(scrublet));
                ask(scrublet);
              }}
              className="text-primary hover:underline"
              title="Set the threshold Scrublet picked from the score distribution"
            >
              Scrublet&apos;s: {fmtScore(scrublet)}
            </button>
          )}
          {loading && <Loader2 className="h-3 w-3 animate-spin text-slate-400" />}
        </div>
        {summary.n_called != null && summary.threshold != null && (
          <p className="text-[11px] text-slate-700" data-testid="doublet-calls">
            Above {fmtScore(summary.threshold)}:{" "}
            <span className="font-semibold">
              {summary.n_called.toLocaleString()} cells ({pct(summary.fraction_called)})
            </span>{" "}
            would be called doublets, of {summary.n_cells.toLocaleString()}.
          </p>
        )}
        <p className="text-[11px] text-slate-500">
          {expectedRateText(summary)}
          {summary.expected_rate_used != null
            ? ` The doublet step ran with an expected rate of ${pct(summary.expected_rate_used)}.`
            : ""}
        </p>
        {samples.length > 0 && (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="font-medium">{summary.suggestion.column}</th>
                <th className="text-right font-medium" title="Cells in the sample">Cells</th>
                <th className="text-right font-medium" title="Cells called doublets: Scrublet score above the threshold">Called</th>
                <th className="text-right font-medium" title="Called doublets as a share of the sample's cells">Share</th>
                <th className="text-right font-medium" title="Rule of thumb: ~0.8 % per 1,000 cells">
                  Expected
                </th>
              </tr>
            </thead>
            <tbody className="text-slate-600">
              {shown.map((s) => (
                <tr key={s.name}>
                  <td className="truncate pr-2">{s.name}</td>
                  <td className="text-right">{s.n_cells.toLocaleString()}</td>
                  <td className="text-right">{s.n_called.toLocaleString()}</td>
                  <td className={`text-right ${s.fraction > 2 * s.expected_rate ? "font-semibold text-amber-700" : ""}`}>
                    {pct(s.fraction)}
                  </td>
                  <td className="text-right text-slate-400">~{pct(s.expected_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {samples.length > SAMPLE_ROWS && (
          <button type="button" onClick={() => setAllSamples((v) => !v)} className="text-[11px] text-primary hover:underline">
            {allSamples ? "Show fewer" : `Show all ${samples.length}`}
          </button>
        )}
        <div className="flex flex-wrap items-center gap-2 border-t border-slate-100 pt-2">
          <button
            type="button"
            onClick={() => void apply(threshold)}
            disabled={applying || threshold === null || unchanged}
            className="rounded bg-primary px-2.5 py-1 text-[11px] font-medium text-white hover:bg-primary/90 disabled:opacity-50"
          >
            {applying ? "Saving…" : "Apply threshold"}
          </button>
          {applied && (
            <button
              type="button"
              onClick={() => void apply(null)}
              disabled={applying}
              className="rounded border border-slate-200 px-2.5 py-1 text-[11px] text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              Back to Scrublet&apos;s call
            </button>
          )}
          <button
            type="button"
            onClick={showOnEmbedding}
            className="text-[11px] text-primary hover:underline"
            title="Colour the Unified View scatter by predicted_doublet (doublet_score is there too)"
          >
            Show calls on the embedding
          </button>
        </div>
        <p className="text-[11px] text-slate-500">
          {applied
            ? `Calls at ${fmtScore(applied.threshold)} are in predicted_doublet (${applied.n_called.toLocaleString()} cells); Scrublet's own call is kept as predicted_doublet_scrublet.`
            : `predicted_doublet holds Scrublet's call${
                summary.n_called_now != null ? ` (${summary.n_called_now.toLocaleString()} cells)` : ""
              }.`}{" "}
          Cell/Gene Filtering drops them when &quot;Drop predicted doublets&quot; is on.
        </p>
        {error && (
          <p role="alert" className="rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
            {error}
          </p>
        )}
      </div>
    </Panel>
  );
}
