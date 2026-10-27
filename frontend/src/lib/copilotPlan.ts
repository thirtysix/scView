/** The co-pilot's multi-step plans (backend core/assistant.py `_as_plan`): run the
 *  steps one at a time, wait for each run to finish, and check a rule after it
 *  (the column, markers or enrichment it should have made exist) before the next.
 *  The checks are deterministic; no model judges the result. */
import type { AssistantAction } from "@/api/assistant";
import { apiFetch } from "@/api/client";
import { streamEvents } from "@/api/clustering";
import type { RunOutcomeFields } from "@/api/pipelineRun";
import type { DatasetInfo, ObsColumnInfo } from "@/api/types";

export type StepStatus = "pending" | "running" | "done" | "failed" | "skipped";
export type PlanStatus = "ready" | "running" | "done" | "failed" | "cancelled";

/** A plan as a chat turn keeps it (saved with the thread). */
export interface PlanTurn {
  steps: AssistantAction[];
  advisory?: string;
  status: PlanStatus;
  stepStatus: StepStatus[];
  message?: string;
}

export function newPlanTurn(a: AssistantAction): PlanTurn {
  const steps = a.steps ?? [];
  return {
    steps,
    advisory: a.advisory || undefined,
    status: "ready",
    stepStatus: steps.map(() => "pending"),
  };
}

/** A plan saved while running (the page was reloaded mid-plan) is not running any more. */
export function settleInterrupted(plan: PlanTurn): PlanTurn {
  if (plan.status !== "running") return plan;
  return {
    ...plan,
    status: "cancelled",
    stepStatus: plan.stepStatus.map((s) => (s === "running" || s === "pending" ? "skipped" : s)),
    message: "Interrupted: the page was reloaded while the plan ran. Check the last step's result.",
  };
}

/** Which groups of a column have markers, and which have enrichment (GET /enrichment/groups). */
export interface EnrichmentGroups {
  groups: string[];
  enrichment_computed: Record<string, boolean>;
}

export interface PlanDeps {
  /** Run one pipeline step to the end; resolves with the run's result. */
  runStep: (step: string, params: Record<string, unknown>) => Promise<RunOutcomeFields>;
  getDataset: () => Promise<DatasetInfo>;
  /** null when the column has no markers (a 404). */
  enrichmentGroups: (column: string) => Promise<EnrichmentGroups | null>;
  applyView: (a: AssistantAction) => void;
  /** The dataset as it is after a run (to refresh the app's copy). */
  onDataset?: (d: DatasetInfo) => void;
}

export interface PlanHooks {
  onStep: (index: number, status: StepStatus) => void;
  /** Checked before each step: Cancel stops after the current one. */
  isCancelled: () => boolean;
}

export interface PlanResult {
  status: "done" | "failed" | "cancelled";
  failedAt?: number;
  message: string;
}

class StepFailed extends Error {}

const fail = (why: string): never => {
  throw new StepFailed(why);
};

const column = (d: DatasetInfo, name: string): ObsColumnInfo | undefined =>
  d.obs_columns.find((c) => c.name === name);

/** A step bound to the columns earlier steps made: `from_step` -> that step's column. */
export function bindStep(step: AssistantAction, made: Record<number, string>): AssistantAction {
  if (step.from_step == null) return step;
  const col = made[step.from_step];
  if (!col) fail(`it needs the column step ${step.from_step + 1} makes, and that step made none`);
  const params =
    step.step === "marker_genes"
      ? { ...step.params, marker_columns: [col] }
      : step.step === "enrichment"
        ? { ...step.params, enrichment_columns: [col] }
        : step.params;
  return { ...step, column: col, params };
}

function needGroup(d: DatasetInfo, col: string, value: string | null | undefined) {
  const info = column(d, col);
  if (!info) fail(`there is no column ${col}`);
  const values = info!.values;
  if (value && values && !values.includes(value)) {
    const some = values.slice(0, 8).join(", ") + (values.length > 8 ? ", …" : "");
    fail(`${col} has no group "${value}" (it has ${values.length}: ${some})`);
  }
}

/** What must hold before a step runs (the columns and groups it names exist). */
export function precondition(step: AssistantAction, d: DatasetInfo): void {
  if (step.type === "cluster") return; // its column is the one it makes
  if (step.type === "open_de_compare" && step.column) {
    needGroup(d, step.column, step.group_a);
    needGroup(d, step.column, step.group_b);
  } else if (step.column) {
    needGroup(d, step.column, step.step ? null : step.value);
  }
}

/** Why a finished run did not do its step, or null when it did. */
export function runFailure(out: RunOutcomeFields | null): string | null {
  if (!out) return "the run sent no result";
  const err = Object.values(out.errors ?? {})[0];
  if (err) return err;
  const status = out.status ?? "complete";
  if (status === "complete") return null;
  if (status === "cancelled") return "the run was cancelled";
  if (status === "stopped") return out.stop_reason || "the run was stopped";
  return "a step failed";
}

/** The rule checked after a run step; returns the column a re-cluster made. */
export async function postcondition(
  step: AssistantAction,
  d: DatasetInfo,
  deps: Pick<PlanDeps, "enrichmentGroups">,
): Promise<string | undefined> {
  const has = (n: string | null | undefined) => !!n && !!column(d, n);
  if (step.step === "clustering") {
    // The run names the column itself (resolution 1 may be r1 or r1.0): it is the active one.
    const made = has(d.active_clustering) ? d.active_clustering! : has(step.column) ? step.column! : "";
    if (!made) fail("the new clustering column is missing");
    return made;
  }
  if (step.step === "marker_genes" || step.step === "enrichment") {
    const col = step.column || d.active_clustering || "";
    if (!col) fail("no clustering column to check");
    const eg = await deps.enrichmentGroups(col);
    if (!eg || !eg.groups.length) {
      fail(`no marker genes were stored for ${col}${step.step === "enrichment" ? " (enrichment needs them)" : ""}`);
    }
    if (step.step === "enrichment") {
      if (step.value) {
        if (!eg!.groups.includes(step.value)) fail(`${col} has no group "${step.value}"`);
        if (!eg!.enrichment_computed[step.value]) fail(`no enrichment result for ${step.value} in ${col}`);
      } else if (!Object.values(eg!.enrichment_computed).some(Boolean)) {
        fail(`no enrichment results were stored for ${col}`);
      }
    }
    return undefined;
  }
  if (step.step === "doublet_detection" && !has("predicted_doublet") && !has("doublet_score")) {
    fail("no doublet columns were added");
  }
  if (step.step === "cell_type_annotation") {
    const target = step.params?.annotation_target as string | undefined;
    if (target && !has(target)) fail(`the ${target} column is missing`);
  }
  return undefined;
}

const stepName = (s: AssistantAction) => (s.label || s.type).replace(/\.$/, "");

/** Run a plan's steps in order. A failed check or an error stops it and says which step
 *  and why; a cancel stops it before the next step. */
export async function runPlan(steps: AssistantAction[], deps: PlanDeps, hooks: PlanHooks): Promise<PlanResult> {
  const made: Record<number, string> = {};
  let ds: DatasetInfo | null = null;
  const skipFrom = (i: number) => {
    for (let j = i; j < steps.length; j++) hooks.onStep(j, "skipped");
  };
  for (let i = 0; i < steps.length; i++) {
    if (hooks.isCancelled()) {
      skipFrom(i);
      return { status: "cancelled", message: `Stopped after step ${i} of ${steps.length}.` };
    }
    hooks.onStep(i, "running");
    try {
      const step = bindStep(steps[i]!, made);
      ds ??= await deps.getDataset();
      precondition(step, ds);
      if (step.step) {
        const out = await deps.runStep(step.step, step.params ?? {});
        const why = runFailure(out);
        if (why) fail(why);
        ds = await deps.getDataset();
        deps.onDataset?.(ds);
        const col = await postcondition(step, ds, deps);
        if (col) made[i] = col;
        if (step.step === "enrichment" && step.value && step.column) {
          // "enrichment for cluster 3": show that group's result
          deps.applyView({ type: "highlight_cluster", column: step.column, value: step.value });
          deps.applyView({ type: "set_subtab", subtab: "enrichment" });
        }
      } else {
        deps.applyView(step);
      }
      hooks.onStep(i, "done");
    } catch (e) {
      hooks.onStep(i, "failed");
      skipFrom(i + 1);
      const why = e instanceof Error ? e.message : String(e);
      return { status: "failed", failedAt: i, message: `Step ${i + 1} (${stepName(steps[i]!)}) failed: ${why}.` };
    }
  }
  return { status: "done", message: `All ${steps.length} steps done.` };
}

/** One pipeline step over the streamed run endpoint (kept alive while it runs, so a
 *  long step is not cut off), resolving with the run's result. */
export async function runStepStreamed(
  datasetId: string,
  step: string,
  params: Record<string, unknown>,
): Promise<RunOutcomeFields> {
  let result: RunOutcomeFields | null = null;
  await streamEvents(
    `/datasets/${datasetId}/assessment/run-stream`,
    { steps: [step], params },
    () => undefined,
    "The run",
    undefined,
    (data) => {
      result = data as RunOutcomeFields;
    },
  );
  return result ?? { steps_run: [] };
}

/** Markers and enrichment per group of a column; null when the column has no markers. */
export async function fetchEnrichmentGroups(datasetId: string, col: string): Promise<EnrichmentGroups | null> {
  try {
    return await apiFetch<EnrichmentGroups>(
      `/datasets/${datasetId}/enrichment/groups?column=${encodeURIComponent(col)}`,
    );
  } catch {
    return null;
  }
}

/** Plans running now, by thread and turn (`planId`): shared by the two mounted chats
 *  (drawer and panel), and outliving either, so both show it and either can stop it. */
const active = new Map<string, { cancel: boolean }>();
export const planId = (threadKey: string, turn: number) => `${threadKey}#${turn}`;
export const planRuns = {
  start(id: string): { cancel: boolean } {
    const run = { cancel: false };
    active.set(id, run);
    return run;
  },
  stop(id: string): void {
    const run = active.get(id);
    if (run) run.cancel = true;
  },
  end(id: string): void {
    active.delete(id);
  },
  isActive: (id: string): boolean => active.has(id),
  any: (): boolean => active.size > 0,
};
