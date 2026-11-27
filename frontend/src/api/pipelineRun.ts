import { apiFetch, readError } from "@/api/client";
import { errorMessage } from "@/lib/errorMessage";
import { isHosted } from "@/lib/memoryAdvice";
import { clearedText } from "@/lib/runErrors";

/** A warning the run watchdog sends into a run's stream (backend core/run_watchdog.py). */
export type WatchdogEvent =
  | {
      type: "watchdog";
      kind: "memory";
      /** The threshold crossed: 80 or 90 (% of the limit). */
      threshold_pct: number;
      used_gb: number;
      limit_gb: number;
      pct: number;
      /** A container limit (SCVIEW_MEMORY_GB); false: all the memory this computer gives scView. */
      limited: boolean;
      step: string | null;
    }
  | {
      type: "watchdog";
      kind: "slow_step";
      step: string;
      elapsed_s: number;
      /** The step's preflight time range in seconds, and in words ("roughly 4 to 16 min"):
       *  missing when the run had no estimate (an older backend, or no plan). */
      expected_s?: [number, number];
      expected_text?: string;
      /** The step's point estimate in seconds; the first note comes at 1.5 times it. */
      estimate_s?: number;
    }
  | {
      type: "watchdog";
      /** Before a step that fits only once other work (another run, a read) has finished:
       *  the run waits for it (backend run_watchdog.room_for_step); Cancel still works. */
      kind: "waiting";
      step: string;
      /** What it waits for, in words ("the run on Kang"). */
      waiting_for: string[];
      waited_s: number;
      max_wait_s: number;
      need_gb: number | null;
      others_gb: number | null;
      available_gb: number | null;
      /** The backend's sentence. */
      message: string;
    };

/** How a run ended: every step ran; a step failed (the steps that need it were skipped, the
 *  others ran, and it was saved); stopped by the pre-step check; or cancelled. */
export type RunStatus = "complete" | "finished_with_errors" | "stopped" | "cancelled";

/** The status of a run that was saved with failed steps (backend core/step_failures.py). */
export const FINISHED_WITH_ERRORS = "finished_with_errors";

/** What a watched run used: memory sampled every `sample_seconds`, each step's seconds. */
export interface RunResources {
  peak_gb: number | null;
  start_gb: number | null;
  limit_gb: number | null;
  limited: boolean;
  sample_seconds: number;
  steps: Record<string, { elapsed_s: number; peak_gb: number | null }>;
  write_s: number | null;
}

/** The run result fields this module reads (the rest is in DataAssessmentPanel's PipelineResult). */
export interface RunOutcomeFields {
  steps_run: string[];
  /** Missing from an older backend: read as "complete". */
  status?: RunStatus;
  stop_reason?: string | null;
  resources?: RunResources | null;
  /** Failed steps and their errors */
  errors?: Record<string, string>;
  /** Steps not run because a step they need failed -> that step */
  skipped?: Record<string, string>;
  /** Results of an earlier run removed as stale: where (obsm, obsp, uns, var, obs, raw) -> keys */
  cleared?: Record<string, string[]>;
}

/** Ask the backend to cancel the pipeline run working on a dataset (404: none is). */
export async function cancelPipelineRun(datasetId: string): Promise<{ cancelling: boolean; step: string | null }> {
  return apiFetch(`/datasets/${datasetId}/assessment/cancel`, { method: "POST" });
}

/** A refused run request in words: the backend's own sentence (a 422 names the valid
 *  steps and the one probably meant), not its JSON. */
export async function runErrorMessage(res: Response): Promise<string> {
  const text = await readError(res);
  try {
    const detail = JSON.parse(text).detail;
    if (typeof detail === "string") return detail;
  } catch {
    /* plain text, or nginx's page already turned into a sentence */
  }
  return `API error ${res.status}: ${text}`;
}

/** The words of the backend's refusal of a run that would convert the dataset to
 *  large-dataset layout without the user's yes (core/processing.py, allow_large_mode false). */
export const CONVERSION_REFUSAL_PHRASE = "run it again allowing large-dataset mode";

export function isConversionRefusal(message: string): boolean {
  return message.includes(CONVERSION_REFUSAL_PHRASE);
}

/** The question before a run converts the dataset: the backend's reason, and what it turns off. */
export function conversionQuestion(message: string): string {
  const reason = (message.split(" Nothing was run:")[0] ?? message).trim();
  return `${reason}\n\nRun it in large-dataset mode anyway? The dataset stays in that layout.`;
}

/** The words of the backend's refusal of a run that would go to large-dataset mode on a dataset
 *  stored by columns (core/store_by_rows.py REFUSAL_PHRASE): the UI offers the rewrite there. */
export const COLUMN_STORED_PHRASE = "store the dataset by rows first";

export function isColumnStoredRefusal(message: string): boolean {
  return message.includes(COLUMN_STORED_PHRASE);
}

/** That refusal where the offer itself cannot be shown (a run started from another panel). */
export function columnStoredText(): string {
  return (
    "Not run: this dataset is stored by columns, and the run does not fit in memory, so it would use " +
    "large-dataset mode, which reads cells in blocks and cannot read that layout. Open Data Assessment: beside " +
    "Run it offers to store the dataset by rows (no value changes). Then run this again."
  );
}

/** Run `call` without the yes to a conversion (`allow_large_mode` false, the backend's
 *  default); when the backend refuses because the run would convert the dataset, ask the
 *  user (`ask`, a confirm by default) and, on yes only, run it again allowing it. Every UI
 *  caller of a run route goes through here or asks the same way (DataAssessmentPanel). */
export async function withConversionConsent<T>(
  call: (allowLarge: boolean) => Promise<T>,
  ask: (question: string) => boolean = (q) => window.confirm(q),
): Promise<T> {
  try {
    return await call(false);
  } catch (e) {
    const msg = errorMessage(e);
    if (isConversionRefusal(msg) && ask(conversionQuestion(msg))) return call(true);
    throw e;
  }
}

/** What a run's failure says in the UI: a declined conversion in plain words (the refusal's own
 *  text names the API flag), anything else as the server's reason. */
export function declinedConversion(): string {
  return (
    "Not run: it would have switched the dataset to large-dataset mode, and you chose not to. Close other " +
    `datasets or ${isHosted() ? "ask the site admin for more memory" : "raise scView's memory limit"} to run ` +
    "it in memory."
  );
}

export function runErrorText(e: unknown): string {
  const msg = errorMessage(e);
  if (isColumnStoredRefusal(msg)) return columnStoredText();
  return isConversionRefusal(msg) ? declinedConversion() : msg;
}

const gb = (n: number) => (n >= 10 ? n.toFixed(0) : n.toFixed(1));

/** One watchdog event as a plain sentence. */
export function watchdogMessage(e: WatchdogEvent, label: (step: string) => string): string {
  if (e.kind === "memory") {
    const of = e.limited
      ? `of the ${gb(e.limit_gb)} GB limit`
      : `of the ${gb(e.limit_gb)} GB this computer gives scView`;
    return (
      `Memory at ${e.pct}% ${of}. If it runs out, the system stops the backend; ` +
      "Cancel keeps the dataset as it was."
    );
  }
  if (e.kind === "waiting") {
    return e.waited_s >= 60 ? `${e.message} Waiting for ${elapsedText(e.waited_s)} so far.` : e.message;
  }
  const sofar = `${label(e.step)}: ${elapsedText(e.elapsed_s)} so far`;
  if (!e.expected_s || !e.expected_text) return `${sofar}.`;
  if (e.elapsed_s <= e.expected_s[1]) return `${sofar} (estimate: ${e.expected_text}).`;
  return `${sofar}, longer than its estimate (${e.expected_text}). It is still working; Cancel stops the run without saving.`;
}

/** Whole minutes, as hours and minutes from an hour on: "3 min", "1 h 5 min". */
function elapsedText(seconds: number): string {
  const minutes = Math.max(1, Math.floor(seconds / 60));
  if (minutes < 60) return `${minutes} min`;
  const m = minutes % 60;
  return `${Math.floor(minutes / 60)} h${m ? ` ${m} min` : ""}`;
}

export { clearedText };

/** The lines of a run that finished with errors: each failed step and its error, the steps
 *  skipped because of it, and the stale results removed. */
export function runErrorLines(
  r: RunOutcomeFields,
  label: (step: string) => string,
): { failed: string[]; skipped: string | null; cleared: string | null } {
  const failed = Object.entries(r.errors ?? {}).map(([s, e]) => `${label(s)} failed: ${e}`);
  const entries = Object.entries(r.skipped ?? {});
  const skipped = entries.length
    ? "Skipped, as they need a step that failed: " +
      entries.map(([s, because]) => `${label(s)} (needs ${label(because)})`).join(", ") + "."
    : null;
  const what = clearedText(r.cleared);
  const cleared = what
    ? `Results of an earlier run that these steps would have replaced were removed, as they no longer match the data: ${what}.`
    : null;
  return { failed, skipped, cleared };
}

/** The sentences for a run that did not run to its end, or null when it did (every step
 *  ran). A run that finished with errors was saved: what failed, what was skipped. */
export function runOutcome(
  r: RunOutcomeFields,
  label: (step: string) => string,
): { title: string; reason: string; steps: string } | null {
  const status = r.status ?? "complete";
  if (status === "complete") return null;
  const done = r.steps_run.map(label).join(", ");
  if (status === FINISHED_WITH_ERRORS) {
    const lines = runErrorLines(r, label);
    return {
      title: "Finished with errors",
      reason: lines.failed.join(" ") || "A step failed.",
      steps: [done ? `Saved: ${done}.` : "No step finished.", lines.skipped, lines.cleared]
        .filter(Boolean)
        .join(" "),
    };
  }
  if (status === "cancelled")
    return {
      title: "Run cancelled",
      reason: r.stop_reason ?? "Cancelled: nothing was saved, so the dataset is as it was before the run.",
      steps: done ? `Finished before the cancel, not saved: ${done}.` : "No step had finished.",
    };
  return {
    title: "Run stopped",
    reason: r.stop_reason ?? "Stopped before a step.",
    steps: done ? `Finished before the stop, not saved: ${done}.` : "No step had run, so nothing changed.",
  };
}

/** "Peak memory 4.2 of 6.0 GB (sampled every 2 s).", from a watched run, or null without one. */
export function peakMemoryText(res: RunResources | null | undefined): string | null {
  if (!res || res.peak_gb == null) return null;
  const of = res.limit_gb != null ? ` of ${gb(res.limit_gb)} GB` : " GB";
  return `Peak memory ${gb(res.peak_gb)}${of} (sampled every ${res.sample_seconds} s).`;
}
