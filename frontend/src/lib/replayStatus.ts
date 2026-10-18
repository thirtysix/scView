import type { RecipeStep, ReplayStatus } from "@/api/provenance";

/** What each replay status means (backend core/replay_script.py), for its badge. */
export const REPLAY_LABELS: Record<ReplayStatus, { label: string; help: string; cls: string }> = {
  exact: {
    label: "exact",
    help: "The script runs the same computation scView ran.",
    cls: "bg-emerald-50 text-emerald-700",
  },
  approximate: {
    label: "approximate",
    help: "The script runs the closest scanpy code; a NOTE in it says what differs.",
    cls: "bg-amber-50 text-amber-700",
  },
  recorded: {
    label: "from record",
    help: "The script writes in the result scView stored (e.g. an LLM's labels) rather than computing it again.",
    cls: "bg-sky-50 text-sky-700",
  },
  not_replayable: {
    label: "not replayable",
    help: "No code can repeat this step (e.g. hand-made labels); the script has a comment with the recorded values.",
    cls: "bg-slate-100 text-slate-600",
  },
  unknown: {
    label: "unknown step",
    help: "This scView version has no code for this step type; the script has a placeholder comment.",
    cls: "bg-red-50 text-red-700",
  },
};

/** A status the backend sent, or "unknown" for a missing or newer one. */
export function replayStatus(step: RecipeStep): ReplayStatus {
  const st = step.replay?.status;
  return st && st in REPLAY_LABELS ? st : "unknown";
}

/** Steps per replay status, in the order the statuses are listed above. */
export function replayCounts(steps: readonly RecipeStep[]): [ReplayStatus, number][] {
  const counts = new Map<ReplayStatus, number>();
  for (const s of steps) {
    const st = replayStatus(s);
    counts.set(st, (counts.get(st) ?? 0) + 1);
  }
  return (Object.keys(REPLAY_LABELS) as ReplayStatus[])
    .filter((k) => counts.has(k))
    .map((k) => [k, counts.get(k)!]);
}
