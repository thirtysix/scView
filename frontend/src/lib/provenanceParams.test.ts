import { describe, it, expect } from "vitest";
import { keptRerunParams, lastRunText, paramText } from "./provenanceParams";

describe("fast Leiden in the History", () => {
  it("the chip says which Leiden optimisation ran", () => {
    expect(paramText("leiden_iterations", 2)).toBe("Leiden: fast (2 passes)");
    expect(paramText("leiden_iterations", -1)).toBe("Leiden: full optimisation");
  });

  it("re-clustering from the History keeps the method and Leiden's passes", () => {
    const rec = { step: "clustering", params: { clustering_method: "leiden", clustering_resolution: 0.5, leiden_iterations: 2 } };
    expect(keptRerunParams(rec)).toEqual({ clustering_method: "leiden", leiden_iterations: 2 });
    expect(keptRerunParams({ step: "clustering", params: { clustering_method: "louvain" } })).toEqual({
      clustering_method: "louvain",
    });
    expect(keptRerunParams({ step: "embeddings", params: { umap_min_dist: 0.3 } })).toEqual({});
  });
});

describe("a run that finished with errors in the History", () => {
  it("the current state says how the last run ended", () => {
    expect(lastRunText({ status: "complete", failed: [], skipped: [] })).toBe("last run: complete");
    expect(lastRunText({ status: "finished_with_errors", failed: ["pca"], skipped: ["neighbors", "clustering"] })).toBe(
      "last run: finished with errors (pca failed; 2 skipped)",
    );
  });

  it("says what failed, what was skipped and what was removed, not JSON", () => {
    expect(paramText("status", "finished_with_errors")).toBe("finished with errors");
    expect(paramText("failed", { pca: "boom" })).toBe("failed: pca (boom)");
    expect(paramText("skipped", { neighbors: "pca", clustering: "pca" })).toBe(
      "skipped, needing a failed step: neighbors, clustering",
    );
    expect(paramText("cleared", { obsm: ["X_pca", "X_umap"], uns: ["neighbors"] })).toBe(
      "removed as stale: PCA, UMAP and neighbour graph",
    );
  });
});

describe("paramText", () => {
  it("shows plain values and lists as before", () => {
    expect(paramText("min_cells", 3)).toBe("min_cells=3");
    expect(paramText("marker_columns", ["leiden", "cell_type"])).toBe("marker_columns=leiden, cell_type");
  });

  it("summarises the filtering step's per-sample thresholds instead of [object Object]", () => {
    expect(paramText("qc_thresholds", { column: "Donor", samples: [{}, {}, {}] })).toBe(
      "per-sample thresholds: Donor (3 samples)",
    );
    expect(paramText("qc_thresholds", { column: null, samples: [] })).toBe("thresholds: all cells pooled");
  });

  it("cuts a long list to its first 12 items and the count", () => {
    const genes = Array.from({ length: 2000 }, (_, i) => `G${i}`);
    expect(paramText("genes", genes)).toBe(`genes=${genes.slice(0, 12).join(", ")}, ... (2,000)`);
    expect(paramText("seeds", [0, 1, 2])).toBe("seeds=0, 1, 2");
  });

  it("summarises the doublet step's channels instead of [object Object]", () => {
    const wells = Array.from({ length: 133 }, (_, i) => ({
      name: String(i + 1), n_cells: 640 + i, expected_doublet_rate: 0.00512 + i * 1.6e-5, n_called: 5, status: "scored",
    }));
    expect(paramText("channels", wells)).toBe("Scrublet per channel: 133 channels, 640-772 cells, expected 0.5-0.7%, 665 called");
    expect(paramText("channels", [{ n_cells: 50, expected_doublet_rate: 0.004, n_called: 0, status: "too_small" }])).toBe(
      "Scrublet per channel: 1 channel, 50 cells, expected 0.4%, 0 called, 1 too small to score",
    );
  });

  it("names the doublet detectors and summarises the classifier's channels", () => {
    expect(paramText("doublet_method", "both")).toBe("doublets: Scrublet + classifier");
    expect(paramText("doublet_method", "scdblfinder-style")).toBe("doublets: scDblFinder-style classifier");
    expect(paramText("doublet_method", "scrublet")).toBe("doublets: Scrublet");
    const lanes = [
      { name: "CTRL", n_cells: 6573, expected_doublet_rate: 0.0526, n_called: 490, status: "scored" },
      { name: "STIM", n_cells: 7263, expected_doublet_rate: 0.0581, n_called: 410, status: "scored" },
    ];
    expect(paramText("classifier", { channels: lanes, n_called: 900 })).toBe(
      "Classifier per channel: 2 channels, 6,573-7,263 cells, expected 5.3-5.8%, 900 called",
    );
  });

  it("keeps any other object short", () => {
    const text = paramText("extra", { a: "x".repeat(200) });
    expect(text.startsWith('extra={"a":"xxx')).toBe(true);
    expect(text.length).toBeLessThanOrEqual(86);
  });

  it("summarises the ingest step's tissue positions and R conversions", () => {
    expect(paramText("spatial", {
      source: "spaceranger", spaceranger_version: null, technology: "visium", unit: "spot",
      libraries: { LN: 4035, brain: 2310 }, images: { LN: ["hires", "lowres"], brain: ["lowres"] },
    })).toBe("Visium: 2 sections (LN 4,035 spots, brain 2,310 spots), images hires, lowres");
    expect(paramText("spatial", { technology: "cells", unit: "cell", libraries: { s1: 1 }, images: { s1: [] } }))
      .toBe("tissue positions: 1 section (s1 1 cell), no image");
    expect(paramText("converted", [
      { sample: "a", file: "a.rds", tool: "scView R converter", source_class: "Seurat" },
      { sample: "b", file: "b.rds", tool: "scView R converter" },
    ])).toBe("converted from R: a.rds (Seurat), b.rds");
  });

  it("never shows [object Object] for a list of objects", () => {
    expect(paramText("rows", [{ a: 1 }])).toBe('rows=[{"a":1}]');
  });
});
