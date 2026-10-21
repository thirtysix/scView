import { apiFetch, readError } from "@/api/client";

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
  | { type: "watchdog"; kind: "slow_step"; step: string; elapsed_s: number };

/** How a run ended: ran to the end (errors or not), stopped by the pre-step check, or cancelled. */
export type RunStatus = "complete" | "stopped" | "cancelled";

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
  const minutes = Math.max(1, Math.floor(e.elapsed_s / 60));
  return `${label(e.step)}: ${minutes} min so far.`;
}

/** The sentences for a run that did not run to its end, or null when it did. */
export function runOutcome(
  r: RunOutcomeFields,
  label: (step: string) => string,
): { title: string; reason: string; steps: string } | null {
  const status = r.status ?? "complete";
  if (status === "complete") return null;
  const done = r.steps_run.map(label).join(", ");
  if (status === "cancelled")
    return {
      title: "Run cancelled",
      reason: r.stop_reason ?? "Cancelled: nothing was saved, so the dataset is as it was before the run.",
      steps: done ? `Finished before the cancel, not saved: ${done}.` : "No step had finished.",
    };
  return {
    title: "Run stopped",
    reason: r.stop_reason ?? "Stopped before a step.",
    steps: done ? `Completed and saved: ${done}.` : "No step had run, so nothing changed.",
  };
}

/** "Peak memory 4.2 of 6.0 GB (sampled every 2 s).", from a watched run, or null without one. */
export function peakMemoryText(res: RunResources | null | undefined): string | null {
  if (!res || res.peak_gb == null) return null;
  const of = res.limit_gb != null ? ` of ${gb(res.limit_gb)} GB` : " GB";
  return `Peak memory ${gb(res.peak_gb)}${of} (sampled every ${res.sample_seconds} s).`;
}
