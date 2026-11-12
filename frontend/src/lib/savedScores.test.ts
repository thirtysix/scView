import { describe, it, expect } from "vitest";
import { savedGeneSetMatches } from "./savedScores";

describe("savedGeneSetMatches", () => {
  const saved = {
    score_IFN: { kind: "geneset", name: "IFN", method: "score_genes", genes: ["ISG15", "MX1"] },
    activity_STAT1: { kind: "activity", name: "IFN", method: "ulm" },
  };

  it("matches the same set, method and genes in any order", () => {
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["MX1", "ISG15"])).toBe(true);
  });

  it("does not match another method, other genes, or no saved entry", () => {
    expect(savedGeneSetMatches(saved, "IFN", "aucell", ["ISG15", "MX1"])).toBe(false);
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["ISG15"])).toBe(false);
    expect(savedGeneSetMatches(saved, "IFN", "score_genes", ["ISG15", "MX1", "IFIT1"])).toBe(false);
    expect(savedGeneSetMatches(saved, "OTHER", "score_genes", ["ISG15"])).toBe(false);
    expect(savedGeneSetMatches({}, "IFN", "score_genes", ["ISG15"])).toBe(false);
  });

  it("reads a record without a method as score_genes (the default)", () => {
    const old = { score_IFN: { kind: "geneset", name: "IFN", genes: ["ISG15"] } };
    expect(savedGeneSetMatches(old, "IFN", "score_genes", ["ISG15"])).toBe(true);
  });
});
