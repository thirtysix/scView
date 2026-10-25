import { describe, it, expect } from "vitest";
import { batchParamText } from "./batchMethods";
import { paramText } from "./provenanceParams";
import { applySteps } from "@/components/compare/JointSummary";

describe("batch-correction History chips", () => {
  it("names the method and the BBKNN graph", () => {
    expect(paramText("batch_method", "scanorama")).toBe("batch correction: Scanorama (corrected PCs)");
    expect(paramText("batch_method", "bbknn")).toBe("batch correction: BBKNN (batch-balanced neighbour graph)");
    expect(paramText("kept", true)).toBe("graph: BBKNN's, from batch correction (kept)");
    expect(paramText("neighbors_within_batch", 8)).toBe("8 neighbours from each batch");
    expect(paramText("representation", "X_scanorama")).toBe("corrected PCs: X_scanorama");
    expect(paramText("approx", false)).toBe("exact neighbours");
  });

  it("leaves other params alone", () => {
    expect(batchParamText("n_neighbors", 15)).toBeNull();
    expect(paramText("n_neighbors", 15)).toBe("n_neighbors=15");
  });
});

describe("applying a method to a joint dataset", () => {
  it("re-runs from batch correction, or from PCA for none", () => {
    expect(applySteps("bbknn")).toEqual(["batch_correction", "neighbors", "clustering", "embeddings", "marker_genes"]);
    expect(applySteps("none")[0]).toBe("pca");
  });
});
