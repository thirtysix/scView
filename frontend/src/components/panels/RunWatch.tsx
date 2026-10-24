import { AlertTriangle, Clock, Loader2, Square } from "lucide-react";
import {
  peakMemoryText,
  runErrorLines,
  runOutcome,
  watchdogMessage,
  type RunOutcomeFields,
  type WatchdogEvent,
} from "@/api/pipelineRun";

type Label = (step: string) => string;

/** Cancel for the run in progress. The run stops before its next step; a step that
 *  is working finishes first (a Leiden clustering is ended at once). */
export function CancelRunButton({ onCancel, cancelling }: { onCancel: () => void; cancelling: boolean }) {
  return (
    <button
      type="button"
      onClick={onCancel}
      disabled={cancelling}
      className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
      title="Stop the run before its next step. Nothing is saved: the dataset stays as it was before the run."
    >
      {cancelling ? <Loader2 className="h-3 w-3 animate-spin" /> : <Square className="h-3 w-3" />}
      {cancelling ? "Cancelling..." : "Cancel"}
    </button>
  );
}

/** The watchdog's warnings while a run works, and what a requested cancel waits for. */
export function RunWatchNotes({
  memory,
  slow,
  cancelling,
  runningStep,
  label,
}: {
  memory: WatchdogEvent | null;
  slow: WatchdogEvent | null;
  cancelling: boolean;
  runningStep: string | null;
  label: Label;
}) {
  if (!memory && !slow && !cancelling) return null;
  return (
    <div className="mt-2 space-y-1 text-xs">
      {memory && (
        <p
          className={`flex items-start gap-1.5 ${memory.kind === "memory" && memory.pct >= 90 ? "text-red-700" : "text-amber-800"}`}
          title="Memory in use (without the file cache the system can drop) against what scView may use, sampled every 2 s while the run works."
        >
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span>{watchdogMessage(memory, label)}</span>
        </p>
      )}
      {slow && (
        <p className="flex items-start gap-1.5 text-blue-700" title="Sent every minute one step keeps running.">
          <Clock className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
          <span>{watchdogMessage(slow, label)}</span>
        </p>
      )}
      {cancelling && (
        <p className="text-slate-600">
          Cancelling: {runningStep ? `${label(runningStep)} finishes first, then ` : ""}the run stops without
          saving.
        </p>
      )}
    </div>
  );
}

/** A run that was cancelled or stopped before a step: why, and which steps finished. */
export function RunOutcome({ result, label }: { result: RunOutcomeFields; label: Label }) {
  const outcome = runOutcome(result, label);
  if (!outcome) return null;
  const peak = peakMemoryText(result.resources);
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
      <p className="font-medium">{outcome.title}</p>
      <p className="mt-1">{outcome.reason}</p>
      <p className="mt-1">{outcome.steps}</p>
      {peak && <p className="mt-1 text-xs text-amber-700">{peak}</p>}
    </div>
  );
}

/** A run that finished with errors: each failed step and its error, the steps skipped
 *  because they need one, and the stale results of an earlier run that were removed. */
export function RunErrors({ result, label }: { result: RunOutcomeFields; label: Label }) {
  const lines = runErrorLines(result, label);
  if (!lines.failed.length && !lines.skipped && !lines.cleared) return null;
  return (
    <div className="mt-2 space-y-1">
      {lines.failed.map((f) => (
        <p key={f} className="text-red-700">
          {f}
        </p>
      ))}
      {lines.skipped && <p>{lines.skipped}</p>}
      {lines.cleared && <p className="text-xs text-amber-800">{lines.cleared}</p>}
    </div>
  );
}

/** The peak memory line under a completed run, from the watchdog's samples. */
export function RunPeakMemory({ result }: { result: RunOutcomeFields }) {
  const peak = peakMemoryText(result.resources);
  if (!peak) return null;
  return (
    <p
      className="mt-1 text-xs text-emerald-700"
      title="The most memory the run was seen using (without the file cache the system can drop). Samples are seconds apart, so a shorter spike can be missed."
    >
      {peak}
    </p>
  );
}
