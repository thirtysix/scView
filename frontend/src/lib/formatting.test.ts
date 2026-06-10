import { describe, it, expect } from "vitest";
import { formatNumber, formatPValue, prettyObsLabel, isRedundantClusterCol } from "./formatting";

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
