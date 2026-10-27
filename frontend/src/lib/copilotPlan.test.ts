import { describe, expect, it, vi } from "vitest";
import type { AssistantAction } from "@/api/assistant";
import type { DatasetInfo, ObsColumnInfo } from "@/api/types";
import { runPlan, settleInterrupted, newPlanTurn, type PlanDeps, type StepStatus } from "@/lib/copilotPlan";
import { applyViewAction } from "@/lib/copilotActions";
import { useDESettingsStore } from "@/stores/deSettingsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

const NEW = "scview_leiden_r1.0";
const cat = (name: string, values: string[]): ObsColumnInfo => ({
  name, dtype: "category", n_unique: values.length, values,
});
const ds = (cols: ObsColumnInfo[], active = "leiden"): DatasetInfo => ({
  id: "kang", name: "Kang", filename: "k.h5ad", n_cells: 100, n_genes: 10, status: "ready",
  available_embeddings: [], obs_columns: cols, active_clustering: active, created_at: "",
});
const BEFORE = ds([cat("leiden", ["0", "1"]), cat("stim", ["CTRL", "STIM"])]);
// The run names the column itself: resolution 1 as r1 (JSON drops the .0)
const AFTER = ds([...BEFORE.obs_columns, cat("scview_leiden_r1", ["0", "1", "2", "3"])], "scview_leiden_r1");

// The plan the backend makes of "re-cluster at resolution 1, then compute markers and
// run enrichment for cluster 3" (core/assistant.py _as_plan)
const PLAN: AssistantAction[] = [
  { type: "cluster", step: "clustering", column: NEW, requires_confirm: true, label: "Re-cluster (Leiden, resolution 1.0).",
    params: { clustering_method: "leiden", clustering_resolution: 1 } },
  { type: "compute_markers", step: "marker_genes", column: NEW, from_step: 0, requires_confirm: true,
    params: { marker_columns: [NEW] }, label: `Compute marker genes for ${NEW}.` },
  { type: "run_enrichment", step: "enrichment", column: NEW, from_step: 0, value: "3", requires_confirm: true,
    params: { enrichment_columns: [NEW] }, label: `Run pathway enrichment for ${NEW}, showing 3.` },
];

function harness(over: Partial<PlanDeps> = {}) {
  let current = BEFORE;
  const deps: PlanDeps = {
    runStep: vi.fn(async (step: string) => {
      if (step === "clustering") current = AFTER;
      return { steps_run: [step], status: "complete" as const, errors: {} };
    }),
    getDataset: vi.fn(async () => current),
    enrichmentGroups: vi.fn(async () => ({
      groups: ["0", "1", "2", "3"], enrichment_computed: { "0": true, "1": true, "2": true, "3": true },
    })),
    applyView: vi.fn(),
    onDataset: vi.fn(),
    ...over,
  };
  const log: [number, StepStatus][] = [];
  let cancel = false;
  const hooks = { onStep: (i: number, s: StepStatus) => log.push([i, s]), isCancelled: () => cancel };
  return { deps, hooks, log, cancelNow: () => (cancel = true) };
}

describe("runPlan", () => {
  it("runs the steps in order, binding later steps to the column the re-cluster made", async () => {
    const { deps, hooks, log } = harness();
    const res = await runPlan(PLAN, deps, hooks);
    expect(res).toEqual({ status: "done", message: "All 3 steps done." });
    // markers and enrichment on the column the run really made (r1, not r1.0)
    expect(vi.mocked(deps.runStep).mock.calls.map((c) => [c[0], c[1]])).toEqual([
      ["clustering", { clustering_method: "leiden", clustering_resolution: 1 }],
      ["marker_genes", { marker_columns: ["scview_leiden_r1"] }],
      ["enrichment", { enrichment_columns: ["scview_leiden_r1"] }],
    ]);
    expect(deps.enrichmentGroups).toHaveBeenCalledWith("scview_leiden_r1");
    expect(log.filter(([, s]) => s === "done").map(([i]) => i)).toEqual([0, 1, 2]);
    // ...and the asked-for group's enrichment is shown
    expect(deps.applyView).toHaveBeenCalledWith({ type: "highlight_cluster", column: "scview_leiden_r1", value: "3" });
    expect(deps.applyView).toHaveBeenCalledWith({ type: "set_subtab", subtab: "enrichment" });
    expect(deps.onDataset).toHaveBeenCalledWith(AFTER);
  });

  it("stops at a failed check and says which step and why", async () => {
    const { deps, hooks, log } = harness({
      enrichmentGroups: vi.fn(async () => ({ groups: ["0", "1", "2", "3"], enrichment_computed: { "3": false } })),
    });
    const res = await runPlan(PLAN, deps, hooks);
    expect(res.status).toBe("failed");
    expect(res.failedAt).toBe(2);
    expect(res.message).toBe(
      `Step 3 (Run pathway enrichment for ${NEW}, showing 3) failed: no enrichment result for 3 in scview_leiden_r1.`,
    );
    expect(log.at(-1)).toEqual([2, "failed"]);
    expect(deps.applyView).not.toHaveBeenCalled();
  });

  it("a group the new clustering does not have fails before the next step runs", async () => {
    const steps: AssistantAction[] = [
      PLAN[0]!,
      { type: "highlight_cluster", column: NEW, from_step: 0, value: "9", label: "Highlighted 9." },
      PLAN[1]!,
    ];
    const { deps, hooks, log } = harness();
    const res = await runPlan(steps, deps, hooks);
    expect(res.status).toBe("failed");
    expect(res.message).toContain('scview_leiden_r1 has no group "9" (it has 4: 0, 1, 2, 3)');
    expect(deps.runStep).toHaveBeenCalledTimes(1); // the markers step never ran
    expect(log.at(-1)).toEqual([2, "skipped"]);
  });

  it("a run that reports an error stops the plan", async () => {
    const { deps, hooks } = harness({
      runStep: vi.fn(async () => ({ steps_run: [], status: "finished_with_errors" as const,
        errors: { clustering: "No neighbour graph: run the pipeline first." } })),
    });
    const res = await runPlan(PLAN, deps, hooks);
    expect(res).toMatchObject({ status: "failed", failedAt: 0 });
    expect(res.message).toContain("No neighbour graph");
    expect(deps.runStep).toHaveBeenCalledTimes(1);
  });

  it("a thrown error (the stream broke) stops the plan too", async () => {
    const { deps, hooks } = harness({ runStep: vi.fn().mockRejectedValue(new Error("The run stopped before it finished")) });
    const res = await runPlan(PLAN, deps, hooks);
    expect(res.message).toBe("Step 1 (Re-cluster (Leiden, resolution 1.0)) failed: The run stopped before it finished.");
  });

  it("Cancel stops after the current step", async () => {
    const h = harness();
    h.deps.runStep = vi.fn(async (step: string) => {
      h.cancelNow(); // pressed while the first step runs
      if (step === "clustering") h.deps.getDataset = vi.fn(async () => AFTER);
      return { steps_run: [step], status: "complete" as const };
    });
    const res = await runPlan(PLAN, h.deps, h.hooks);
    expect(res).toEqual({ status: "cancelled", message: "Stopped after step 1 of 3." });
    expect(h.deps.runStep).toHaveBeenCalledTimes(1);
    expect(h.log).toEqual([[0, "running"], [0, "done"], [1, "skipped"], [2, "skipped"]]);
  });

  it("view steps apply without a run", async () => {
    const { deps, hooks } = harness();
    const steps: AssistantAction[] = [
      { type: "set_color_by", column: "stim" },
      { type: "open_de_compare", column: "stim", group_a: "STIM", group_b: "CTRL", subtab: "de" },
    ];
    expect((await runPlan(steps, deps, hooks)).status).toBe("done");
    expect(deps.runStep).not.toHaveBeenCalled();
    expect(deps.applyView).toHaveBeenCalledTimes(2);
  });
});

describe("plan turns", () => {
  it("a plan saved mid-run reads as interrupted, not running", () => {
    const p = { ...newPlanTurn({ type: "run_plan", steps: PLAN }), status: "running" as const,
      stepStatus: ["done", "running", "pending"] as StepStatus[] };
    const s = settleInterrupted(p);
    expect(s.status).toBe("cancelled");
    expect(s.stepStatus).toEqual(["done", "skipped", "skipped"]);
    expect(s.message).toContain("Interrupted");
  });
});

describe("open_de_compare", () => {
  it("pre-fills DE between groups without running it", () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cell_type", value: "B" } });
    useDESettingsStore.setState({ mode: "selection", withinCol: "cell_type", withinVal: "NK" });
    applyViewAction({ type: "open_de_compare", column: "stim", group_a: "STIM", group_b: "CTRL", subtab: "de" });
    const s = useDESettingsStore.getState();
    expect([s.mode, s.splitBy, s.groupA, s.groupB, s.withinCol, s.withinVal]).toEqual(
      ["groups", "stim", "STIM", "CTRL", "", ""]);
    // the current highlight is marked seen, so opening the tab does not scope to it
    expect(s.scopedHighlight).toEqual({ column: "cell_type", value: "B" });
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("de");
    expect(useViewStore.getState().activePanel).toBe("unified");
  });
});
