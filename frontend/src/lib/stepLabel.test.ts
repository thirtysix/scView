import { describe, expect, it } from "vitest";
import { stepLabel } from "./stepLabel";

describe("stepLabel", () => {
  it("keeps abbreviations in capitals", () => {
    expect(stepLabel("qc_metrics")).toBe("QC metrics");
    expect(stepLabel("pca")).toBe("PCA");
    expect(stepLabel("highly_variable_genes")).toBe("Highly variable genes");
    expect(stepLabel("umap_3d")).toBe("UMAP 3d");
    expect(stepLabel("tsne")).toBe("t-SNE");
  });

  it("names the spatial steps in words", () => {
    expect(stepLabel("spatial_autocorr")).toBe("Spatially variable genes (Moran's I)");
    expect(stepLabel("doublet_detection_skipped")).toBe("Doublet detection skipped");
  });
});
