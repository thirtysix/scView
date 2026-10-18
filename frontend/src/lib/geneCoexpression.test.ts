import { describe, expect, it } from "vitest";
import {
  BLEND_CORNERS,
  blendColor,
  blendColors,
  blendScale,
  cellsWithCode,
  pearson,
  quadrantCells,
  quadrantCounts,
  quantile,
  scaled,
  sharedRange,
  spearmanOf,
  strideSample,
  valueOrder,
  valuesByGroup,
} from "./geneCoexpression";

describe("correlations", () => {
  const x = [1, 2, 3, 4, 5];
  const y = [2, 4, 5, 4, 5];

  it("gives Pearson's r as numpy does", () => {
    // np.corrcoef([1,2,3,4,5],[2,4,5,4,5])[0,1]
    expect(pearson(x, y)).toBeCloseTo(0.7745966692, 9);
    expect(pearson([1, 2, 3], [3, 2, 1])).toBeCloseTo(-1, 12);
    expect(pearson([1, 1, 1], [1, 2, 3])).toBeNull();
    expect(pearson([1, 2], [1, 2])).toBeNull();
  });

  it("restricts both correlations to a subset of cells", () => {
    const a = [0, 1, 2, 3, 10, 0];
    const b = [5, 1, 2, 3, -10, 0];
    const subset = [1, 2, 3];
    expect(pearson(a, b, subset)).toBeCloseTo(1, 12);
    expect(spearmanOf(a, b, subset)).toBeCloseTo(1, 12);
    expect(pearson(a, b)!).toBeLessThan(0);
    // scipy.stats.spearmanr(x, y) with ties averaged
    expect(spearmanOf(x, y)).toBeCloseTo(0.7378647873, 9);
  });
});

describe("quadrants", () => {
  // cells:      0  1  2  3  4  5
  const a = [0, 1, 2, 0, 0.5, -1];
  const b = [0, 0, 3, 4, 0.1, 2];

  it("counts cells expressing both genes, one, or neither (positive = above 0)", () => {
    expect(quadrantCounts(a, b)).toEqual({ both: 2, aOnly: 1, bOnly: 2, neither: 1 });
    expect(quadrantCounts(a, b, [0, 1, 2])).toEqual({ both: 1, aOnly: 1, bOnly: 0, neither: 1 });
  });

  it("selects the cells of a quadrant", () => {
    expect([...quadrantCells(a, b)]).toEqual([2, 4]);
    expect([...quadrantCells(a, b, "bOnly")]).toEqual([3, 5]);
    expect([...quadrantCells(a, b, "neither")]).toEqual([0]);
  });

  it("finds the cells of one group", () => {
    expect(Array.from(cellsWithCode([0, 1, 1, 2, 1], 1))).toEqual([1, 2, 4]);
  });
});

describe("blend", () => {
  it("takes quantiles as numpy's default (linear) does", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantile([4, 1, 3, 2], 0.99)).toBeCloseTo(3.97, 10);
    expect(quantile([NaN], 0.5)).toBeNaN();
  });

  it("scales a gene from 0 to the clip quantile of its expressing cells", () => {
    // 96 zeros, then 1, 2, 3, 100: the outlier is clipped at the 99th percentile of 1..100.
    const v = [...Array(96).fill(0), 1, 2, 3, 100];
    const s = blendScale(v, 0.99);
    expect(s.lo).toBe(0);
    expect(s.hi).toBeCloseTo(quantile([1, 2, 3, 100], 0.99), 10);
    expect(blendScale(v, 1)).toEqual({ lo: 0, hi: 100 });
    expect(scaled(100, s)).toBe(1);
    expect(scaled(0, s)).toBe(0);
    // A gene nobody expresses still has a usable scale.
    expect(blendScale([0, 0, 0])).toEqual({ lo: 0, hi: 1 });
    // A layer with negative values starts at its minimum.
    expect(blendScale([-2, 0, 2], 1)).toEqual({ lo: -2, hi: 2 });
  });

  it("maps A to red, B to green, both to yellow and neither to grey", () => {
    expect(blendColor(0, 0)).toEqual(BLEND_CORNERS.neither);
    expect(blendColor(1, 0)).toEqual([255, 0, 0]);
    expect(blendColor(0, 1)).toEqual([0, 255, 0]);
    expect(blendColor(1, 1)).toEqual([255, 255, 0]);
    // Half of each: the mean of the four corners.
    expect(blendColor(0.5, 0.5)).toEqual([180, 180, 53]);
    expect(blendColor(2, -1)).toEqual([255, 0, 0]);
  });

  it("colours cells in draw order and fades cells outside a selection", () => {
    const a = [0, 4, 0, 4];
    const b = [0, 0, 4, 4];
    const rgba = blendColors(Uint32Array.from([3, 0, 1, 2]), a, b, { opacity: 1, clip: 1 });
    const px = (s: number) => Array.from(rgba.slice(s * 4, s * 4 + 4));
    expect(px(0)).toEqual([255, 255, 0, 255]);
    expect(px(1)).toEqual([211, 211, 211, 255]);
    expect(px(2)).toEqual([255, 0, 0, 255]);
    expect(px(3)).toEqual([0, 255, 0, 255]);
    const faded = blendColors(Uint32Array.from([0, 1]), a, b, { opacity: 1, selected: new Set([1]) });
    expect(faded[3]).toBe(Math.round(0.15 * 255));
    expect(faded[7]).toBe(255);
  });
});

describe("feature grid helpers", () => {
  it("draws higher values last, ties in their shuffled order", () => {
    const values = [5, 0, 2, 0, 9];
    expect(Array.from(valueOrder(Uint32Array.from([4, 3, 2, 1, 0]), (c) => values[c]!))).toEqual([3, 1, 2, 0, 4]);
    expect(Array.from(valueOrder(Uint32Array.from([0, 1]), (c) => (c === 0 ? NaN : 1)))).toEqual([0, 1]);
  });

  it("shares one range across genes", () => {
    expect(sharedRange([[0, 2], Float32Array.from([1, 7]), [NaN, -1]])).toEqual({ min: -1, max: 7 });
    expect(sharedRange([])).toEqual({ min: 0, max: 0 });
  });

  it("groups values by category name for a violin, optionally for a sample of cells", () => {
    const v = [1, 2, 3, 4];
    const codes = Int32Array.from([1, 0, 1, 2]);
    expect(valuesByGroup(v, codes, ["B", "NK"])).toEqual({ B: [2], NK: [1, 3], "2": [4] });
    expect(valuesByGroup(v, codes, ["B", "NK"], [0, 3])).toEqual({ NK: [1], "2": [4] });
  });

  it("samples every k-th cell above a cap", () => {
    expect(strideSample(5, 10)).toBeNull();
    expect(Array.from(strideSample(10, 4)!)).toEqual([0, 3, 6, 9]);
    expect(strideSample(100_000, 30_000)!.length).toBeLessThanOrEqual(30_000);
  });
});
