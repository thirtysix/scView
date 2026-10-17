import { describe, it, expect } from "vitest";
import {
  formatNumber,
  formatPValue,
  formatFoldChange,
  prettyObsLabel,
  prettyEmbeddingLabel,
  isRedundantClusterCol,
} from "./formatting";

describe("formatNumber", () => {
  it("abbreviates thousands and millions", () => {
    expect(formatNumber(950)).toBe("950");
    expect(formatNumber(13836)).toBe("13.8K");
    expect(formatNumber(2_500_000)).toBe("2.5M");
  });
});

describe("formatPValue", () => {
  it("uses scientific notation for tiny values, fixed otherwise", () => {
    expect(formatPValue(0)).toBe("0.00e+0");
    expect(formatPValue(1e-9)).toBe("1.00e-9");
    expect(formatPValue(0.0123)).toBe("0.0123");
  });
});

describe("prettyObsLabel", () => {
  it("prettifies scview clustering columns", () => {
    expect(prettyObsLabel("scview_leiden_r0.5")).toBe("Leiden (res 0.5)");
    expect(prettyObsLabel("scview_louvain_r1.0")).toBe("Louvain (res 1.0)");
  });
  it("prettifies cell-type annotation columns recursively", () => {
    expect(prettyObsLabel("scview_leiden_r0.5_celltypeAnno")).toBe("Leiden (res 0.5) · cell types");
  });
  it("names the annotation confidence check's columns after their label column", () => {
    expect(prettyObsLabel("scview_conf_cell_type_flag")).toBe("cell_type · label fit");
    expect(prettyObsLabel("scview_conf_cell_type_set_size")).toBe("cell_type · label set size");
    expect(prettyObsLabel("scview_conf_cluster_2_alt")).toBe("cluster_2 · nearest other label");
    expect(prettyObsLabel("scview_conf_scview_leiden_r0.5_unknown")).toBe(
      "Leiden (res 0.5) · Unknown where rejected",
    );
    expect(prettyObsLabel("scview_conf_x_fit")).toBe("x · fit p-value");
  });
  it("passes through unknown names and handles nullish input", () => {
    expect(prettyObsLabel("cell_type")).toBe("cell_type");
    expect(prettyObsLabel(null)).toBe("");
    expect(prettyObsLabel(undefined)).toBe("");
  });
});

describe("isRedundantClusterCol", () => {
  it("flags bare leiden/louvain only when a named scview copy exists", () => {
    expect(isRedundantClusterCol("leiden", ["leiden", "scview_leiden_r0.5"])).toBe(true);
    expect(isRedundantClusterCol("leiden", ["leiden"])).toBe(false);
    expect(isRedundantClusterCol("cell_type", ["cell_type", "scview_leiden_r0.5"])).toBe(false);
  });
});

describe("prettyEmbeddingLabel", () => {
  it("labels plain, corrected and uncorrected embeddings", () => {
    expect(prettyEmbeddingLabel("X_umap")).toBe("UMAP");
    expect(prettyEmbeddingLabel("X_umap_3d")).toBe("UMAP");
    expect(prettyEmbeddingLabel("X_umap_uncorrected")).toBe("UMAP (uncorrected)");
    expect(prettyEmbeddingLabel("X_pca_harmony")).toBe("PCA (HARMONY)");
  });
});

describe("formatFoldChange", () => {
  it("formats a number and marks a missing one", () => {
    expect(formatFoldChange(1.23456, 2)).toBe("1.23");
    expect(formatFoldChange(null, 2)).toBe("–");
    expect(formatFoldChange(Number.NaN, 4, "")).toBe("");
  });
});
