import { describe, it, expect } from "vitest";
import type { DomainPlan } from "@/api/spatialDomains";
import { labelColumns, planText, repName, runShare, runText, scanText, strongestPairs, zRange } from "./spatialDomains";

const plan = (over: Partial<DomainPlan> = {}): DomainPlan => ({
  n_obs: 1000, technology: "visium", unit: "spot", libraries: ["A"], rep: "X_pca", n_pcs: 10, n_pcs_available: 50, n_hops: 3,
  dims: 40, k: 7, k_min: 2, k_max: 15, repeats: 5, seed: 0, n_fit: 1000, graph: "", notes: [], estimate_seconds: 20,
  estimate_scan_seconds: 180, estimate_bytes: 1, ...over,
});

describe("spatial domains text", () => {
  it("says which embedding, how many components and hops, and the time", () => {
    expect(planText(plan())).toBe("First 10 components of the PCA, 3 hops; about 20 s at k = 7.");
    expect(planText(plan({ rep: "X_pca_harmony", n_hops: 1 }))).toContain("the Harmony-corrected PCs, 1 hop;");
    expect(repName("X_other")).toBe("X_other");
  });

  it("states a scan's size and its sample", () => {
    expect(scanText(plan())).toBe("Suggest k tries k = 2 to 15, 5 fits each; about 3 min.");
    expect(scanText(plan({ n_obs: 400000, n_fit: 50000 }))).toContain("on a sample of 50,000");
  });

  it("reports a run's step, count and elapsed time and its share done", () => {
    const stage = { stage: "scan", label: "Stability", index: 3, total: 5, elapsed_s: 2 };
    expect(runText(null, null)).toBe("Starting…");
    expect(runText(stage, null)).toBe("Step 3 of 5: Stability");
    expect(runText(stage, { done: 10, total: 80, elapsed_s: 60 })).toBe("Step 3 of 5: Stability, 10 of 80 (1 min so far)");
    expect(runShare(null, null)).toBe(0);
    expect(runShare(stage, { done: 40, total: 80, elapsed_s: 1 })).toBeCloseTo(0.5);
  });
});

describe("enrichment heatmap", () => {
  it("range is the largest |z|, with missing values skipped and never 0", () => {
    expect(zRange([[1, -7], [null, 3]])).toBe(7);
    expect(zRange([[null]])).toBe(1);
  });

  it("takes the strongest pairs of different labels and ignores the diagonal and weak ones", () => {
    const labels = ["a", "b", "c"];
    expect(strongestPairs(labels, [[99, 5, -3], [5, 99, 1], [-3, 1, 99]])).toEqual({
      high: ["a", "b", 5],
      low: ["a", "c", -3],
    });
    expect(strongestPairs(labels, [[1, 1, 1], [1, 1, null], [1, null, 1]])).toEqual({ high: null, low: null });
  });
});

describe("label columns", () => {
  it("keeps categorical columns with 2 to 100 labels", () => {
    const cols = labelColumns([
      { name: "ok", dtype: "category", n_unique: 5 },
      { name: "text", dtype: "object", n_unique: 3 },
      { name: "num", dtype: "float32", n_unique: 3 },
      { name: "one", dtype: "category", n_unique: 1 },
      { name: "many", dtype: "category", n_unique: 101 },
    ]);
    expect(cols.map((c) => c.name)).toEqual(["ok", "text"]);
  });
});
