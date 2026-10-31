import { describe, expect, it } from "vitest";
import {
  activityLabel,
  depthChoiceText,
  formatDistance,
  formatP,
  planText,
  sampled,
  splitByGroup,
  stageText,
  wnnPlanText,
} from "./atac";
import { parseSession, type SessionOverlay } from "./session";

describe("ATAC text", () => {
  it("says what a run will read and need before it starts", () => {
    const text = planText({
      n_cells: 11898, n_peaks: 143887, nnz: 104_012_773, genome: "GRCh38", n_lsi: 50, dims_to: 30, drop_first: "auto",
      percentile: 5, resolution: 1, activity: true, estimate_seconds: 240, estimate_bytes: 1.2 * 1024 ** 3,
      estimate_disk_bytes: 1.8 * 1024 ** 3,
    });
    expect(text).toContain("11,898 cells");
    expect(text).toContain("143,887 peaks");
    expect(text).toContain("104M values");
    expect(text).toContain("1.2 GB");
    expect(text).toContain("1.8 GB of disk");
  });

  it("names the WNN inputs", () => {
    const text = wnnPlanText({ n_cells: 10, rep: "X_pca", d_rna: 50, lsi_dims: [2, 3, 50], notes: [], estimate_seconds: 60, estimate_bytes: 1024 ** 3 });
    expect(text).toContain("50 RNA PCs (PCA)");
    expect(text).toContain("LSI 2:50");
  });

  it("explains the depth choice either way", () => {
    const dropped = depthChoiceText({ depth_r: [-0.79], depth_limit: 0.75, dropped_first: true, dims_used: [2, 3, 30] });
    expect(dropped).toContain("r = -0.79");
    expect(dropped).toContain("left out");
    expect(dropped).toContain("components 2:30");
    const kept = depthChoiceText({ depth_r: [0.2], depth_limit: 0.75, dropped_first: false, dims_used: [1, 30] });
    expect(kept).toContain("it is kept");
    expect(kept).toContain("components 1:30");
  });

  it("reports a stage", () => {
    expect(stageText(null)).toBe("Starting…");
    expect(stageText({ stage: "umap", label: "ATAC UMAP", index: 4, total: 6, elapsed_s: 1 })).toBe("Step 4 of 6: ATAC UMAP");
  });

  it("formats p and TSS distance", () => {
    expect(formatP(0)).toBe("< 1e-300");
    expect(formatP(2e-21)).toBe("2e-21");
    expect(formatP(0.0342)).toBe("0.034");
    expect(formatP(null)).toBe("–");
    expect(formatDistance(0)).toBe("at TSS");
    expect(formatDistance(1500)).toBe("+1.5 kb");
    expect(formatDistance(-340)).toBe("-340 b");
    expect(formatDistance(null)).toBe("–");
  });

  it("labels the overlay as activity, never expression", () => {
    expect(activityLabel("CD3D")).toBe("CD3D gene activity");
  });
});

describe("ATAC QC sampling", () => {
  it("keeps everything under the cap and spaces the cells over it", () => {
    expect(sampled([1, 2, 3], 5)).toEqual([1, 2, 3]);
    const out = sampled(Array.from({ length: 100 }, (_, i) => i), 10);
    expect(out).toHaveLength(10);
    expect(out[0]).toBe(0);
    expect(out[9]).toBe(90);
  });

  it("splits by group, leaving out cells with no value or no group", () => {
    const by = splitByGroup([1, 2, null, 4, 5], ["a", "b", "a", null, "a"], ["a", "b"], 100);
    expect(by).toEqual({ a: [1, 5], b: [2] });
  });
});

describe("session round trip of the gene activity overlay", () => {
  const session = (overlay: unknown) => ({
    format: "scview-session",
    version: 1,
    view: { overlay },
  });

  it("parses it, and refuses one with no gene", () => {
    const o: SessionOverlay = { type: "atac_activity", gene: "CD3D" };
    expect(parseSession(session(o)).view?.overlay).toEqual(o);
    expect(() => parseSession(session({ type: "atac_activity" }))).toThrow();
  });
});
