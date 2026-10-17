import { describe, it, expect } from "vitest";
import { spearman, targetGenes } from "./protein";

describe("targetGenes", () => {
  it("maps antibody targets to their genes, the name itself last", () => {
    expect(targetGenes("CD16")[0]).toBe("FCGR3A");
    expect(targetGenes("CD45RO")[0]).toBe("PTPRC");
    expect(targetGenes("PD-1")[0]).toBe("PDCD1");
    expect(targetGenes("CD8a")[0]).toBe("CD8A");
    expect(targetGenes("TIGIT")).toEqual(["TIGIT", "Tigit"]);
    expect(targetGenes("anti-CD3")[0]).toBe("CD3E");
    expect(targetGenes("NovelTarget")).toEqual(["NovelTarget", "NOVELTARGET"]);
  });
});

describe("spearman", () => {
  it("is 1 for any increasing relation, -1 decreasing, and handles ties", () => {
    expect(spearman([1, 2, 3, 4], [1, 4, 9, 16])).toBeCloseTo(1, 12);
    expect(spearman([1, 2, 3, 4], [4, 3, 2, 1])).toBeCloseTo(-1, 12);
    // scipy.stats.spearmanr([0,0,1,2,3],[0,1,1,2,5]) = 0.9210526315789475
    expect(spearman([0, 0, 1, 2, 3], [0, 1, 1, 2, 5])).toBeCloseTo(0.9210526315789475, 10);
    expect(spearman([1, 1, 1], [1, 2, 3])).toBeNull();
    expect(spearman(new Float32Array([1, 2]), new Float32Array([2, 1]))).toBeNull();
  });
});
