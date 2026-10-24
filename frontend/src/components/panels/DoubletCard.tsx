import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { applyDoubletThreshold, getDoublets, type DoubletChannelCalls, type DoubletSummary } from "@/api/qc";
import { CapNote, Caption } from "@/components/common/Caption";
import { Panel } from "@/components/common/Panel";
import { DoubletAgreement } from "@/components/qc/DoubletAgreement";
import { DoubletMethods } from "@/components/qc/DoubletMethods";
import { ThresholdHistogram, type RefLine } from "@/components/plots/ThresholdHistogram";
import { useDatasetRefresh } from "@/hooks/useDatasetRefresh";
import { InfoTip } from "@/components/common/InfoTip";
import {
  fewCalledHelp,
  fewCalledText,
  perChannelText,
  rateRange,
  statusText,
  unscoredText,
} from "@/lib/doubletChannels";
import { sourceText } from "@/lib/doubletSource";
import { pct, roundTo } from "@/lib/qcThresholds";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";

const STEPS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01];
const DEBOUNCE_MS = 120;
const CHANNEL_ROWS = 8; // more behind "show all"
const FEW_CALLED_SHOWN = 3; // channels named in the warning; the rest in the table
const GUESS =
  "No lane or GEM well column was found, so Scrublet ran in each sample: donors pooled in one lane are not " +
  "channels, so its expected rates are a guess. Pick the capture channel in the Doublet Detection step if the " +
  "data has one.";

/** Threshold resolution: about a 200th of the histogram's range (0.005 on 0-1; finer when
 *  low per-channel rates put every score under 0.1). */
export function thresholdStep(top: number | undefined): number {
  const raw = (top && top > 0 ? top : 1) / 200;
  return STEPS.find((s) => s >= raw) ?? STEPS[STEPS.length - 1]!;
}

/** A score for display: Scrublet's own threshold has 17 digits. */
export function fmtScore(v: number): string {
  return String(Number(v.toFixed(4)));
}

/** The rule-of-thumb sentence under the histogram: the rates the run used per channel,
 *  or (one channel of every cell) the one it used. */
export function expectedRateText(s: DoubletSummary): string {
  const rule = "a rule of thumb from 10x Genomics, ~0.8 % per 1,000 cells captured in one channel.";
  const rows = (s.channels ?? []).filter((r) => r.status === "scored");
  if (s.per_channel && rows.length) {
    return `Expected ${rateRange(rows.map((r) => r.expected_rate))} doublets per channel, from each channel's cells: ${rule}`;
  }
  const rate = s.expected_rate_used ?? s.suggestion.rate;
  return `Expected about ${pct(rate)} doublets, taking all cells as one channel: ${rule}`;
}

/** The channel table's rows: by share called, the most doublet-heavy first, when there
 *  are more than fit; in their own order otherwise. */
export function channelRows(rows: DoubletChannelCalls[], all: boolean): DoubletChannelCalls[] {
  if (rows.length <= CHANNEL_ROWS) return rows;
  const sorted = [...rows].sort((a, b) => b.fraction - a.fraction);
  return all ? sorted : sorted.slice(0, CHANNEL_ROWS);
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
  const [allChannels, setAllChannels] = useState(false);
  const [openChannel, setOpenChannel] = useState<string | null>(null); // its scores' histogram
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

  const edges = summary?.hist?.bin_edges;
  const step = thresholdStep(edges?.length ? edges[edges.length - 1] : undefined);
  const commit = useCallback(
    (raw: number) => {
      const t = roundTo(raw, step);
      setThreshold(t);
      setText(fmtScore(t));
      ask(t);
    },
    [ask, step],
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
  if (!summary.available || !summary.hist) {
    // The classifier ran alone: no Scrublet scores to threshold, its own calls
    if (!summary.classifier) return null;
    return (
      <Panel title="Doublet calls">
        <div className="space-y-2">
          <DoubletMethods classifier={summary.classifier} scrubletAt="Scrublet" showHistogram />
          {summary.source && (
            <Caption testId="doublet-source">
              Doublet views use: {sourceText(summary.source)}
            </Caption>
          )}
        </div>
      </Panel>
    );
  }

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
  const hashingFirst = summary.source?.source === "hashing"; // auto picks hashing once it was run
  const unchanged = applied !== null && threshold !== null && Math.abs(applied.threshold - threshold) < 1e-9;
  const channels = summary.channels ?? [];
  const shown = channelRows(channels, allChannels);
  const minCells = summary.min_channel_cells ?? 100;
  const perChannel = perChannelText(summary);
  const unscored = unscoredText(summary.n_unscored ?? 0, summary.n_unscored_small_channels ?? 0, minCells);
  const guess = summary.channel_reason === "donor" || summary.channel_reason === "batch";
  // Channels whose automatic threshold called almost nothing (backend doublets.FEW_CALLED_*)
  const fewCalled = channels.filter((r) => r.few_called);

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
              const t = roundTo(v, step);
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
            step={step}
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
        {perChannel && (
          <p className="text-[11px] text-slate-600" data-testid="doublet-per-channel">
            {perChannel}
            {summary.threshold == null && summary.n_called_now != null && (
              <>
                {" "}
                Scrublet&apos;s calls:{" "}
                <span className="font-semibold">
                  {summary.n_called_now.toLocaleString()} cells ({pct(summary.n_called_now / Math.max(1, summary.n_cells))})
                </span>
                . Type or drag a threshold to see one over every channel.
              </>
            )}
          </p>
        )}
        {fewCalled.length > 0 && (
          <div className="space-y-1 rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800" data-testid="doublet-few-called">
            {fewCalled.slice(0, FEW_CALLED_SHOWN).map((r) => (
              <div key={r.name}>
                <p className="flex flex-wrap items-center gap-x-1.5">
                  <span>{fewCalledText(r)}</span>
                  <InfoTip width={320} label={`Why ${r.name} is flagged`}>
                    {fewCalledHelp(summary.few_called_share ?? 0.002, summary.few_called_min_cells ?? 2000)}
                  </InfoTip>
                  {r.hist && (
                    <button
                      type="button"
                      onClick={() => setOpenChannel((c) => (c === r.name ? null : r.name))}
                      className="text-primary hover:underline"
                    >
                      {openChannel === r.name ? "Hide its scores" : "Show its scores"}
                    </button>
                  )}
                </p>
                {openChannel === r.name && r.hist && (
                  <div className="mt-1 h-[140px] rounded bg-white" data-testid="doublet-channel-hist">
                    <ThresholdHistogram
                      edges={summary.hist!.bin_edges}
                      counts={r.hist}
                      // Draggable from the channel's own threshold when none is set yet
                      lines={
                        lines.length || r.scrublet_threshold == null
                          ? lines
                          : [{ id: "t", value: r.scrublet_threshold, label: "threshold" }]
                      }
                      refLines={
                        r.scrublet_threshold != null
                          ? [{ value: r.scrublet_threshold, label: `Scrublet (${r.name})`, color: "#b45309", dash: "dot", showLabel: true }]
                          : []
                      }
                      logY={logY}
                      barColor="#f59e0b"
                      plotName={`qc_doublet_score_${r.name}`}
                      onDrag={(_, v) => {
                        const t = roundTo(v, step);
                        setText(fmtScore(t));
                        ask(t);
                      }}
                      onCommit={(_, v) => commit(v)}
                    />
                  </div>
                )}
              </div>
            ))}
            {fewCalled.length > FEW_CALLED_SHOWN && (
              <p>And {(fewCalled.length - FEW_CALLED_SHOWN).toLocaleString()} more channels (in the table).</p>
            )}
          </div>
        )}
        {guess && (
          <p className="rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800" data-testid="doublet-guess">
            {summary.suggestion.note && summary.suggestion.reason === summary.channel_reason ? summary.suggestion.note : GUESS}
          </p>
        )}
        {unscored && (
          <p className="text-[11px] text-slate-600" data-testid="doublet-unscored">
            {unscored}
          </p>
        )}
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
        {channels.length > 0 && (
          <table className="w-full text-[11px]" aria-label="Doublet calls per capture channel">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="font-medium" title={`Capture channel (${summary.channels_column ?? ""})`}>
                  {summary.channels_column === "capture_channel" ? "Channel" : summary.channels_column}
                </th>
                <th className="text-right font-medium" title="Cells in the channel">Cells</th>
                <th
                  className="text-right font-medium"
                  title={
                    summary.threshold != null
                      ? "Cells called doublets: score above the threshold shown"
                      : "Cells called doublets now: Scrublet's own call in this channel"
                  }
                >
                  Called
                </th>
                <th className="text-right font-medium" title="Called doublets as a share of the channel's cells">Share</th>
                <th className="text-right font-medium" title="The expected rate Scrublet used here: ~0.8 % per 1,000 cells">
                  Expected
                </th>
                <th className="text-right font-medium" title="The threshold Scrublet picked in this channel">
                  Scrublet&apos;s
                </th>
              </tr>
            </thead>
            <tbody className="text-slate-600">
              {shown.map((r) => {
                const note = statusText(r, minCells);
                return (
                  <tr key={r.name} className={r.status === "scored" ? "" : "text-slate-400"}>
                    <td className="truncate pr-2">{r.name}</td>
                    <td className="text-right tabular-nums">{r.n_cells.toLocaleString()}</td>
                    {r.status === "scored" || r.n_scored > 0 ? (
                      <>
                        <td className="text-right tabular-nums">{(r.n_called ?? 0).toLocaleString()}</td>
                        <td className={`text-right tabular-nums ${r.fraction > 2 * r.expected_rate ? "font-semibold text-amber-700" : ""}`}>
                          {pct(r.fraction)}
                        </td>
                      </>
                    ) : (
                      <td colSpan={2} className="text-right">{note ?? "not scored"}</td>
                    )}
                    <td className="text-right tabular-nums text-slate-400">~{pct(r.expected_rate)}</td>
                    <td
                      className={`text-right tabular-nums ${r.few_called ? "font-semibold text-amber-700" : "text-slate-400"}`}
                      title={r.few_called ? fewCalledText(r) : (note ?? undefined)}
                    >
                      {r.scrublet_threshold != null ? fmtScore(r.scrublet_threshold) : "—"}
                      {r.few_called && (
                        <AlertTriangle className="ml-0.5 inline h-3 w-3 align-[-2px]" aria-label="called almost nothing" />
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {channels.length > CHANNEL_ROWS && (
          <div className="flex flex-wrap items-baseline gap-2">
            {!allChannels && <CapNote shown={CHANNEL_ROWS} total={channels.length} by="share called" />}
            <button type="button" onClick={() => setAllChannels((v) => !v)} className="text-[11px] text-primary hover:underline">
              {allChannels ? "Show fewer" : `Show all ${channels.length}`}
            </button>
          </div>
        )}
        {summary.agreement && (
          <DoubletAgreement
            agreement={summary.agreement}
            scrubletAt={
              summary.agreement.threshold != null
                ? `Scrublet at ${fmtScore(summary.agreement.threshold)}`
                : "Scrublet's saved call"
            }
          />
        )}
        {summary.classifier && (
          <DoubletMethods
            classifier={summary.classifier}
            methods={summary.methods}
            scrubletAt={summary.threshold != null ? `Scrublet at ${fmtScore(summary.threshold)}` : "Scrublet"}
          />
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
          {hashingFirst
            ? "Cell hashing is the primary call here: cluster QC, the QC report and Cell/Gene Filtering's " +
              "\"Drop predicted doublets\" (source auto) use its doublets; pick Scrublet or Either there to drop these."
            : "Cell/Gene Filtering drops them when \"Drop predicted doublets\" is on."}
        </p>
        {summary.source && (
          <Caption testId="doublet-source">
            Doublet views use: {sourceText(summary.source)}
          </Caption>
        )}
        {error && (
          <p role="alert" className="rounded-md bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
            {error}
          </p>
        )}
      </div>
    </Panel>
  );
}
