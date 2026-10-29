import { describe, it, expect } from "vitest";
import { BACKEND_DOWN_MESSAGE } from "@/api/client";
import {
  clearedText,
  peakMemoryText,
  runErrorLines,
  runErrorMessage,
  runOutcome,
  watchdogMessage,
  type RunResources,
  type WatchdogEvent,
} from "./pipelineRun";

const label = (s: string) => ({ doublet_detection: "Doublet Detection", qc_metrics: "QC Metrics" })[s] ?? s;

type MemoryEvent = Extract<WatchdogEvent, { kind: "memory" }>;

const memory = (pct: number, limited = true): MemoryEvent => ({
  type: "watchdog", kind: "memory", threshold_pct: pct >= 90 ? 90 : 80, used_gb: 7.3, limit_gb: 8,
  pct, limited, step: "pca",
});

describe("watchdogMessage", () => {
  it("says memory against the limit and what Cancel keeps", () => {
    expect(watchdogMessage(memory(91), label)).toBe(
      "Memory at 91% of the 8.0 GB limit. If it runs out, the system stops the backend; " +
        "Cancel keeps the dataset as it was.",
    );
  });

  it("says when the limit is the computer's memory, not a set limit", () => {
    expect(watchdogMessage({ ...memory(82, false), limit_gb: 15.6 }, label)).toMatch(
      /^Memory at 82% of the 16 GB this computer gives scView\./,
    );
  });

  it("gives a slow step's minutes so far, by its label", () => {
    const slow = (s: number): WatchdogEvent => ({ type: "watchdog", kind: "slow_step", step: "doublet_detection", elapsed_s: s });
    expect(watchdogMessage(slow(185), label)).toBe("Doublet Detection: 3 min so far.");
    expect(watchdogMessage(slow(60), label)).toBe("Doublet Detection: 1 min so far.");
    expect(watchdogMessage(slow(3900), label)).toBe("Doublet Detection: 1 h 5 min so far.");
    expect(watchdogMessage(slow(7200), label)).toBe("Doublet Detection: 2 h so far.");
  });

  it("puts the minutes against the step's estimate, and says plainly when it is past it", () => {
    const slow = (s: number): WatchdogEvent => ({
      type: "watchdog", kind: "slow_step", step: "doublet_detection", elapsed_s: s,
      expected_s: [240, 960], expected_text: "roughly 4 to 16 min",
    });
    expect(watchdogMessage(slow(300), label)).toBe("Doublet Detection: 5 min so far (estimate: roughly 4 to 16 min).");
    expect(watchdogMessage(slow(960), label)).toBe("Doublet Detection: 16 min so far (estimate: roughly 4 to 16 min).");
    expect(watchdogMessage(slow(1020), label)).toBe(
      "Doublet Detection: 17 min so far, longer than its estimate (roughly 4 to 16 min). " +
        "It is still working; Cancel stops the run without saving.",
    );
  });
});

describe("runOutcome", () => {
  it("is null for a run that ran to its end, and for an older backend without status", () => {
    expect(runOutcome({ steps_run: ["qc_metrics"], status: "complete" }, label)).toBeNull();
    expect(runOutcome({ steps_run: ["qc_metrics"] }, label)).toBeNull();
  });

  it("says a cancelled run saved nothing and which steps had finished", () => {
    const o = runOutcome(
      {
        steps_run: ["qc_metrics", "doublet_detection"], status: "cancelled",
        stop_reason: "Cancelled before filtering: nothing was saved, so the dataset is as it was before the run.",
      },
      label,
    );
    expect(o).toEqual({
      title: "Run cancelled",
      reason: "Cancelled before filtering: nothing was saved, so the dataset is as it was before the run.",
      steps: "Finished before the cancel, not saved: QC Metrics, Doublet Detection.",
    });
  });

  it("says a stopped run kept the steps it completed", () => {
    const o = runOutcome({ steps_run: ["qc_metrics"], status: "stopped", stop_reason: "PCA needs 9 GB." }, label);
    expect(o).toMatchObject({ title: "Run stopped", reason: "PCA needs 9 GB.", steps: "Finished before the stop, not saved: QC Metrics." });
    expect(runOutcome({ steps_run: [], status: "stopped", stop_reason: "No." }, label)!.steps).toBe(
      "No step had run, so nothing changed.",
    );
  });
});

describe("a run that finished with errors", () => {
  const r = {
    steps_run: ["qc_metrics"],
    status: "finished_with_errors" as const,
    errors: { doublet_detection: "Scrublet failed" },
    skipped: { filtering: "doublet_detection" },
    cleared: { obsm: ["X_pca", "X_umap_3d", "X_umap"], obsp: ["connectivities"], uns: ["rank_genes_groups__leiden", "enrichment__leiden__0"] },
  };
  const lbl = (s: string) => ({ filtering: "Cell/Gene Filtering" })[s] ?? label(s);

  it("runOutcome: saved, with what failed and what was skipped", () => {
    expect(runOutcome(r, lbl)).toEqual({
      title: "Finished with errors",
      reason: "Doublet Detection failed: Scrublet failed",
      steps:
        "Saved: QC Metrics. Skipped, as they need a step that failed: Cell/Gene Filtering (needs Doublet Detection). " +
        "Results of an earlier run that these steps would have replaced were removed, as they no longer match the data: " +
        "PCA, UMAP, neighbour graph, marker genes and enrichment.",
    });
  });

  it("names what was removed once each, and nothing when nothing was", () => {
    expect(clearedText(r.cleared)).toBe("PCA, UMAP, neighbour graph, marker genes and enrichment");
    expect(clearedText({ uns: ["scview_active_clustering"] })).toBe("active clustering");
    expect(clearedText({})).toBeNull();
    expect(runErrorLines({ steps_run: [], errors: { pca: "boom" } }, label)).toEqual({
      failed: ["pca failed: boom"], skipped: null, cleared: null,
    });
  });
});

describe("peakMemoryText", () => {
  const res = { peak_gb: 4.23, limit_gb: 6, sample_seconds: 2 } as RunResources;
  it("gives the sampled peak against the limit, and says how often it was sampled", () => {
    expect(peakMemoryText(res)).toBe("Peak memory 4.2 of 6.0 GB (sampled every 2 s).");
    expect(peakMemoryText({ ...res, limit_gb: null })).toBe("Peak memory 4.2 GB (sampled every 2 s).");
  });
  it("is null for an unwatched run", () => {
    expect(peakMemoryText(null)).toBeNull();
    expect(peakMemoryText({ ...res, peak_gb: null })).toBeNull();
  });
});

describe("runErrorMessage", () => {
  it("shows a 422's sentence, not its JSON", async () => {
    const detail = "Unknown pipeline step 'umap' (did you mean 'embeddings'?). Valid steps: qc_metrics.";
    const res = new Response(JSON.stringify({ detail }), { status: 422 });
    expect(await runErrorMessage(res)).toBe(detail);
  });

  it("turns nginx's page into the backend-down sentence and keeps plain text", async () => {
    expect(await runErrorMessage(new Response("<html>502</html>", { status: 502 }))).toBe(
      `API error 502: ${BACKEND_DOWN_MESSAGE}`,
    );
    expect(await runErrorMessage(new Response("boom", { status: 500 }))).toBe("API error 500: boom");
  });
});
