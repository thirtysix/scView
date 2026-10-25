import { describe, expect, it } from "vitest";
import { fakeCompare } from "@/test/communicationCompareFixture";
import {
  compareLabel,
  compareSummary,
  diffColorRange,
  diffMatrix,
  filterCompareRows,
  sortCompareRows,
  toCompareRows,
  DEFAULT_COMPARE_FILTER,
} from "./communicationCompare";

describe("compare rows", () => {
  it("turns the column table into rows that keep their server index", () => {
    const rows = toCompareRows(fakeCompare());
    expect(rows).toHaveLength(3);
    expect(rows[1]).toMatchObject({ index: 1, source: "NK", call: "lost", delta: -0.9, n_donors: 8, composition_note: true });
    expect(rows[0]!.magnitude_rank_a).toBeNull();
    expect(compareLabel(rows[0]!)).toBe("Mono → NK: CXCL10 → CXCR3");
  });

  it("filters by call, groups, name and adjusted p", () => {
    const rows = toCompareRows(fakeCompare());
    expect(filterCompareRows(rows, DEFAULT_COMPARE_FILTER).map((r) => r.call)).toEqual(["gained", "lost"]);
    expect(filterCompareRows(rows, { ...DEFAULT_COMPARE_FILTER, call: "all" })).toHaveLength(3);
    expect(filterCompareRows(rows, { ...DEFAULT_COMPARE_FILTER, call: "lost" })[0]!.ligand_complex).toBe("IFNG");
    expect(filterCompareRows(rows, { ...DEFAULT_COMPARE_FILTER, call: "all", target: "Mono" })).toHaveLength(2);
    expect(filterCompareRows(rows, { ...DEFAULT_COMPARE_FILTER, query: "cxcr" })).toHaveLength(1);
    expect(filterCompareRows(rows, { ...DEFAULT_COMPARE_FILTER, call: "all", maxPadj: 0.05 })).toHaveLength(1);
  });

  it("sorts with nulls last either way and by absolute change", () => {
    const rows = toCompareRows(fakeCompare());
    expect(sortCompareRows(rows, "abs_delta", false).map((r) => r.ligand_complex)).toEqual(["CXCL10", "IFNG", "CCL2"]);
    expect(sortCompareRows(rows, "p_adj", true).map((r) => r.p_adj)).toEqual([0.03, 0.06, null]);
    expect(sortCompareRows(rows, "p_adj", false).map((r) => r.p_adj)).toEqual([0.06, 0.03, null]);
    expect(sortCompareRows(rows, "source", true)[0]!.source).toBe("Mono");
  });
});

describe("difference matrix", () => {
  it("is B minus A, nulls as 0", () => {
    const res = fakeCompare();
    expect(diffMatrix(res, "count")).toEqual([[3, 0], [-2, 0]]);
    expect(diffMatrix(res, "strength")).toEqual([[0.5, -1], [0, 0.5]]);
  });

  it("has a colour range symmetric about 0, also when every change has one sign", () => {
    expect(diffColorRange([[3, 0], [-2, 0]], "RdBu")).toEqual({ zmin: -3, zmax: 3, colormap: "RdBu" });
    expect(diffColorRange([[0, 1], [4, 0]], "PuOr")).toEqual({ zmin: -4, zmax: 4, colormap: "PuOr" });
    expect(diffColorRange([[0, 0]], "RdBu")).toEqual({ zmin: -1, zmax: 1, colormap: "RdBu" });
  });

  it("summarises the calls", () => {
    expect(compareSummary(fakeCompare())).toBe(
      "STIM vs CTRL: 1 gained, 1 lost, 0 changed, 1 unchanged of the 3 interactions specific in either condition",
    );
  });
});
