import { describe, it, expect } from "vitest";
import { barLabel, barValue, crosstabBars, sortBars, waffleSquares, type Crosstab } from "./compositionBars";

// Donors x cell types, split by condition (d2 has no STIM cells).
const CT: Crosstab = {
  row_values: ["d10", "d2", "d1"],
  col_values: ["B", "CD14 Mono"],
  counts: [
    [10, 30],
    [40, 10],
    [5, 5],
  ],
  facet_values: ["CTRL", "STIM"],
  facet_counts: [
    [
      [4, 16],
      [40, 10],
      [5, 0],
    ],
    [
      [6, 14],
      [0, 0],
      [0, 5],
    ],
  ],
};

describe("crosstabBars", () => {
  it("makes one bar per row without a split", () => {
    const bars = crosstabBars(CT, false);
    expect(bars.map((b) => [b.row, b.facet, b.total])).toEqual([
      ["d10", null, 40],
      ["d2", null, 50],
      ["d1", null, 10],
    ]);
  });

  it("makes one bar per (facet, row) with cells when split", () => {
    const bars = crosstabBars(CT, true);
    expect(bars.map(barLabel)).toEqual(["d10 (CTRL)", "d2 (CTRL)", "d1 (CTRL)", "d10 (STIM)", "d1 (STIM)"]);
    expect(bars[0]!.counts).toEqual([4, 16]);
  });

  it("ignores a split the response does not carry", () => {
    const plain: Crosstab = { row_values: CT.row_values, col_values: CT.col_values, counts: CT.counts };
    expect(crosstabBars(plain, true)).toHaveLength(3);
  });
});

describe("sortBars", () => {
  const opts = { rowOrder: ["d1", "d2", "d10"], colValues: CT.col_values, facetOrder: CT.facet_values };

  it("keeps the stored category order by default", () => {
    expect(sortBars(crosstabBars(CT, false), "category", opts).map((b) => b.row)).toEqual(["d1", "d2", "d10"]);
  });

  it("sorts names naturally (d2 before d10)", () => {
    const rev = { ...opts, rowOrder: [] };
    expect(sortBars(crosstabBars(CT, false), "name", rev).map((b) => b.row)).toEqual(["d1", "d2", "d10"]);
  });

  it("sorts by size, largest first", () => {
    expect(sortBars(crosstabBars(CT, false), "size", opts).map((b) => b.row)).toEqual(["d2", "d10", "d1"]);
  });

  it("sorts by the share of a chosen category, largest first", () => {
    const by = { ...opts, shareOf: "CD14 Mono" };
    // d10 75%, d1 50%, d2 20%
    expect(sortBars(crosstabBars(CT, false), "share", by).map((b) => b.row)).toEqual(["d10", "d1", "d2"]);
  });

  it("keeps the facets in order and one row order across them, from the rows' totals", () => {
    const by = { ...opts, shareOf: "CD14 Mono" };
    // Overall: d10 75%, d1 50%, d2 20%, so STIM is d10 then d1 although d1 is 100% there.
    expect(sortBars(crosstabBars(CT, true), "share", by).map(barLabel)).toEqual([
      "d10 (CTRL)", "d1 (CTRL)", "d2 (CTRL)", "d10 (STIM)", "d1 (STIM)",
    ]);
    expect(sortBars(crosstabBars(CT, true), "category", opts).map(barLabel)).toEqual([
      "d1 (CTRL)", "d2 (CTRL)", "d10 (CTRL)", "d1 (STIM)", "d10 (STIM)",
    ]);
  });
});

describe("barValue", () => {
  it("gives counts or percent of the bar", () => {
    const [d10] = crosstabBars(CT, false);
    expect(barValue(d10!, 1, false)).toBe(30);
    expect(barValue(d10!, 1, true)).toBe(75);
    expect(barValue({ row: "x", facet: null, counts: [0], total: 0 }, 0, true)).toBe(0);
  });
});

describe("waffleSquares", () => {
  it("adds up to 100 by the largest remainder", () => {
    const sq = waffleSquares([1, 1, 1]);
    expect(sq.reduce((a, b) => a + b, 0)).toBe(100);
    expect(sq).toEqual([34, 33, 33]);
  });

  it("matches exact percentages", () => {
    expect(waffleSquares([30, 70])).toEqual([30, 70]);
    expect(waffleSquares([1, 199], 10)).toEqual([0, 10]);
  });

  it("is all zeros for an empty bar", () => {
    expect(waffleSquares([0, 0])).toEqual([0, 0]);
  });
});
