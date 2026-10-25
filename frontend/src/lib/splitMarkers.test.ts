import { describe, expect, it } from "vitest";
import type { SplitGene, SplitGroup, SplitMarkersResult } from "@/api/splitMarkers";
import {
  NEG_LOG10_CAP,
  comparisonPoints,
  groupsByConcern,
  keptText,
  negLog10,
  splitPValueMap,
  summaryText,
} from "./splitMarkers";

const gene = (over: Partial<SplitGene>): SplitGene => ({
  gene: "G",
  logfoldchange: 1,
  pval: 1e-5,
  pval_adj: 1e-4,
  split_logfoldchange: 1,
  split_pval: 0.01,
  split_pval_adj: 0.02,
  split_pct_in: 0.5,
  split_pct_out: 0.1,
  marker: true,
  kept: true,
  ...over,
});

const group = (over: Partial<SplitGroup>): SplitGroup => ({
  group: "0",
  n_cells: 100,
  train_cluster: "3",
  jaccard: 0.8,
  shared: false,
  tested: true,
  reason: null,
  n_ordinary: 20,
  n_kept: 15,
  n_lost: 5,
  n_split: 16,
  n_split_only: 1,
  kept: [],
  lost: [],
  split_only: [],
  genes: [],
  ...over,
});

describe("splitMarkers helpers", () => {
  it("takes -log10 with zero at the cap and nothing for a missing p", () => {
    expect(negLog10(0.01)).toBeCloseTo(2);
    expect(negLog10(0)).toBe(NEG_LOG10_CAP);
    expect(negLog10(1e-320)).toBe(NEG_LOG10_CAP);
    expect(negLog10(null)).toBeNull();
    expect(negLog10(Number.NaN)).toBeNull();
  });

  it("marks points kept, lost or other and skips genes without both p-values", () => {
    const pts = comparisonPoints([
      gene({ gene: "A" }),
      gene({ gene: "B", kept: false }),
      gene({ gene: "C", marker: false, kept: false }),
      gene({ gene: "D", split_pval_adj: null }),
    ]);
    expect(pts.map((p) => [p.gene, p.kind])).toEqual([
      ["A", "kept"],
      ["B", "lost"],
      ["C", "other"],
    ]);
    expect(pts[0]!.x).toBeCloseTo(4);
  });

  it("maps genes to their split adjusted p", () => {
    const m = splitPValueMap(group({ genes: [gene({ gene: "A", split_pval_adj: 0.5 })] }));
    expect(m.get("A")).toBe(0.5);
    expect(m.has("B")).toBe(false);
    expect(splitPValueMap(undefined).size).toBe(0);
  });

  it("words a group and the run", () => {
    expect(keptText(group({}))).toBe("keeps 15 of 20 markers");
    expect(keptText(group({ tested: false }))).toBe("not tested");
    expect(keptText(group({ tested: false, reason: "too few cells to test" }))).toBe("not tested: too few cells to test");
    expect(keptText(group({ n_ordinary: 0, n_kept: 0 }))).toBe("no markers to check");
    const r = {
      n_keep_enough: 7,
      n_tested_groups: 9,
      n_kept: 1200,
      n_ordinary: 3400,
      thresholds: { keeps_enough: 10 },
    } as SplitMarkersResult;
    expect(summaryText(r)).toBe(
      "7 of 9 groups keep at least 10 markers on held-out counts; 1,200 of 3,400 ordinary markers kept",
    );
  });

  it("puts groups keeping few markers first, untested last", () => {
    const order = groupsByConcern([
      group({ group: "a", n_kept: 30 }),
      group({ group: "b", tested: false }),
      group({ group: "c", n_kept: 2 }),
    ]).map((g) => g.group);
    expect(order).toEqual(["c", "a", "b"]);
  });
});
