import { describe, it, expect } from "vitest";
import {
  bandwidth,
  compareSides,
  densities,
  effectLines,
  fmtValue,
  histograms,
  ksDistance,
  mannWhitneyAuc,
  quantile,
  sidesFromGroups,
  sidesFromSelection,
  splitSides,
  summarise,
  toScale,
} from "./overlayHistogram";
import { pseudoLog } from "./logAxis";

const sorted = (xs: number[]) => Float64Array.from(xs).sort();

describe("sides", () => {
  it("puts the selection on side A and every other cell on side B", () => {
    expect(Array.from(sidesFromSelection(5, new Set([1, 3, 9])))).toEqual([2, 1, 2, 1, 2]);
  });

  it("compares a group with the rest (cells without a group left out) or with another group", () => {
    const codes = [0, 1, 2, -1, 0];
    expect(Array.from(sidesFromGroups(codes, 0, "rest"))).toEqual([1, 2, 2, 0, 1]);
    expect(Array.from(sidesFromGroups(codes, 0, 2))).toEqual([1, 0, 2, 0, 1]);
  });

  it("splits finite values by side, sorted", () => {
    const { a, b } = splitSides([3, NaN, 1, 2, Infinity, 0], Uint8Array.from([1, 1, 1, 2, 2, 0]));
    expect(Array.from(a)).toEqual([1, 3]);
    expect(Array.from(b)).toEqual([2]);
  });
});

describe("statistics", () => {
  it("interpolates quantiles", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
    expect(quantile([], 0.5)).toBeNaN();
  });

  it("KS distance is the largest gap between the cumulative distributions", () => {
    expect(ksDistance(sorted([1, 2, 3]), sorted([1, 2, 3]))).toBe(0);
    expect(ksDistance(sorted([5, 6]), sorted([1, 2]))).toBe(1);
    // A: 1,2,3,4 ; B: 3,4,5,6 -> at 2: F_A 0.5, F_B 0
    expect(ksDistance(sorted([1, 2, 3, 4]), sorted([3, 4, 5, 6]))).toBe(0.5);
  });

  it("AUC counts ties as half and matches the pairwise definition", () => {
    expect(mannWhitneyAuc(sorted([1, 1]), sorted([1, 1]))).toBe(0.5);
    expect(mannWhitneyAuc(sorted([2, 3]), sorted([0, 1]))).toBe(1);
    expect(mannWhitneyAuc(sorted([0, 1]), sorted([2, 3]))).toBe(0);
    const a = [0, 0, 1, 3, 3, 7];
    const b = [0, 1, 1, 2, 5];
    let u = 0;
    for (const x of a) for (const y of b) u += x > y ? 1 : x === y ? 0.5 : 0;
    expect(mannWhitneyAuc(sorted(a), sorted(b))).toBeCloseTo(u / (a.length * b.length), 12);
  });

  it("summarises each side: n, median, mean, share above 0", () => {
    expect(summarise(sorted([0, 0, 2, 4]))).toEqual({ n: 4, median: 1, mean: 1.5, fracPositive: 0.5 });
    const c = compareSides(sorted([3, 4, 5, 6]), sorted([0, 0, 1, 2]));
    expect(c.medianDiff).toBe(4);
    expect(c.ks).toBe(1);
    expect(c.auc).toBe(1);
  });

  it("says the effect in words, not as a test", () => {
    const lines = effectLines(compareSides(sorted([3, 4, 5, 6]), sorted([0, 0, 1, 2])), "selected", "other cells");
    expect(lines[0]).toBe("Difference in medians (selected − other cells): +4");
    expect(lines[1]).toMatch(/^KS distance 1.00:/);
    expect(lines[2]).toMatch(/AUC 1.00: .*the selected one has the higher value 100% of the time/);
    expect(effectLines(compareSides(sorted([]), sorted([1])), "a", "b")).toEqual([]);
  });

  it("formats values to three significant digits", () => {
    expect(fmtValue(0.012345)).toBe("0.0123");
    expect(fmtValue(12345.6)).toBe((12346).toLocaleString());
    expect(fmtValue(NaN)).toBe("–");
  });
});

describe("curves", () => {
  const a = sorted([0, 0, 0, 1, 2, 2, 3, 8]);
  const b = sorted([0, 1, 1, 1, 5]);

  it("normalises each side's histogram to its own cells, on shared bins", () => {
    const h = histograms(a, b, 8);
    expect(h.x).toHaveLength(8);
    expect(h.a.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(h.b.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
    expect(h.a[0]).toBeCloseTo(3 / 8, 12); // the zeros
    expect(h.width).toBe(1);
  });

  it("bins on the log scale when asked", () => {
    const h = histograms(a, b, 4, true);
    expect(h.x[3]! + h.width / 2).toBeCloseTo(pseudoLog(8), 12);
    expect(toScale(8, true)).toBe(pseudoLog(8));
    expect(toScale(8, false)).toBe(8);
  });

  it("densities integrate to about 1", () => {
    const big = sorted(Array.from({ length: 2000 }, (_, i) => Math.sin(i) * 3 + 5));
    const d = densities(big, b, 200);
    const area = (ys: number[]) => ys.reduce((s, v) => s + v, 0) * d.width;
    expect(area(d.a)).toBeGreaterThan(0.97);
    expect(area(d.a)).toBeLessThan(1.03);
    expect(area(d.b)).toBeGreaterThan(0.97); // the grid leaves room for the tails
    expect(area(d.b)).toBeLessThan(1.03);
    // the curve comes down to (about) 0 at both ends
    expect(d.b[0]! / Math.max(...d.b)).toBeLessThan(0.05);
    expect(d.b[d.b.length - 1]! / Math.max(...d.b)).toBeLessThan(0.05);
  });

  it("does not spike when most values are the same", () => {
    const spiky = sorted([...new Array(500).fill(0), 1, 50]);
    const d = densities(spiky, sorted([0, 10, 20]), 100);
    const area = d.a.reduce((s, v) => s + v, 0) * d.width;
    expect(area).toBeGreaterThan(0.9);
    expect(area).toBeLessThan(1.1);
  });

  it("keeps a positive bandwidth when every value is the same", () => {
    expect(bandwidth([2, 2, 2, 2], 1)).toBeGreaterThan(0);
    const d = densities(sorted([2, 2, 2]), sorted([2, 2]), 50);
    expect(d.a.every(Number.isFinite)).toBe(true);
  });
});
