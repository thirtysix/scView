import { describe, it, expect } from "vitest";
import type { MarkerGene } from "@/api/markers";
import { isStrongAndSpecific, specificityPoints } from "./MarkerSpecificityPlot";

const m = (gene: string, lfc: number | null, tau: number | null): MarkerGene => ({
  gene, group: "B", logfoldchange: lfc, pval: 0, pval_adj: 0, tau,
});

describe("specificity plot points", () => {
  it("plots only genes with both a fold change and a tau", () => {
    const pts = specificityPoints([m("A", 2, 0.9), m("B", null, 0.9), m("C", 2, null), m("D", NaN, 0.5)]);
    expect(pts.map((p) => p.gene)).toEqual(["A"]);
  });

  it("calls a gene strong and specific past both guides", () => {
    expect(isStrongAndSpecific({ logfoldchange: 6.9, tau: 0.945 })).toBe(true);
    expect(isStrongAndSpecific({ logfoldchange: 3.7, tau: 0.52 })).toBe(false); // CD74: shared
    expect(isStrongAndSpecific({ logfoldchange: 0.5, tau: 0.95 })).toBe(false);
  });
});
