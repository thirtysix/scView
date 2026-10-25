import { describe, it, expect } from "vitest";
import {
  binSelected,
  cellsInPlotlySelection,
  countSelectedByCode,
  countSelectedByPair,
  sampleSelected,
  selectedSorted,
  selectedValues,
  selectionMask,
} from "./linkedSelection";

describe("linkedSelection", () => {
  it("builds one mask per selection and size, ignoring cells out of range", () => {
    const sel = new Set([4, 0, 2, 99]);
    const m = selectionMask(sel, 5);
    expect([...m]).toEqual([1, 0, 1, 0, 1]);
    expect(selectionMask(sel, 5)).toBe(m); // cached on the Set
    expect(selectionMask(new Set([4, 0, 2, 99]), 5)).not.toBe(m);
    expect([...selectedSorted(sel)]).toEqual([0, 2, 4, 99]);
  });

  it("counts the selected cells per group code and per pair of codes", () => {
    const codes = Int32Array.from([0, 1, 1, 2, -1, 0]);
    const mask = selectionMask(new Set([1, 3, 4, 5]), 6);
    const { selected, total } = countSelectedByCode(codes, mask, 3);
    expect([...selected]).toEqual([1, 1, 1]); // cell 4 has no value
    expect([...total]).toEqual([2, 2, 1]);
    const facets = Int32Array.from([0, 0, 1, 1, 1, 1]);
    // [facet * nRows + row]: cell 1 = (0, 1), 3 = (1, 2), 5 = (1, 0)
    expect([...countSelectedByPair(codes, facets, mask, 3, 2)]).toEqual([0, 1, 0, 1, 0, 1]);
  });

  it("takes the selected cells' values and an even sample of them", () => {
    const sel = new Set([3, 1, 7]);
    expect([...selectedValues(Float32Array.from([10, 11, 12, 13, 14]), sel)]).toEqual([11, 13]);
    expect([...sampleSelected(sel, 5, 10)]).toEqual([1, 3]);
    const big = new Set(Array.from({ length: 1000 }, (_, i) => i));
    const s = sampleSelected(big, 1000, 100);
    expect(s.length).toBe(100);
    expect(s[0]).toBe(0);
    expect(s[99]).toBe(990);
  });

  it("maps a Plotly box or lasso to every cell inside it, drawn or not", () => {
    const x = Float32Array.from([0, 1, 2, 3, 4]);
    const y = Float32Array.from([0, 1, 2, 3, 4]);
    expect([...cellsInPlotlySelection(x, y, { range: { x: [3.5, 0.5], y: [0, 10] } })!]).toEqual([1, 2, 3]);
    const lasso = { lassoPoints: { x: [-1, 2.5, 2.5, -1], y: [-1, -1, 2.5, 2.5] } };
    expect([...cellsInPlotlySelection(x, y, lasso)!]).toEqual([0, 1, 2]);
    // On a log axis the outline is in axis units.
    const log = (v: number) => Math.log10(1 + v);
    expect([...cellsInPlotlySelection(x, y, { range: { x: [0.25, 0.5], y: [0, 10] } }, { x: log })!]).toEqual([1, 2]);
    expect(cellsInPlotlySelection(x, y, {})).toBeNull();
    expect(cellsInPlotlySelection(x, y, null)).toBeNull();
  });

  it("bins the selected cells with a histogram's own edges", () => {
    const values = Float32Array.from([0, 0.5, 1, 1.5, 2, 9, -3, NaN]);
    const mask = selectionMask(new Set([0, 1, 2, 3, 4, 5, 6, 7]), 8);
    // [0, 1), [1, 2), [2, 3]: 9 falls in the last bin, -3 in the first, NaN nowhere.
    expect(binSelected(values, mask, [0, 1, 2, 3])).toEqual([3, 2, 2]);
    expect(binSelected(values, selectionMask(new Set([2]), 8), [0, 1, 2, 3])).toEqual([0, 1, 0]);
  });

  it("stays fast at 100k cells", () => {
    const n = 100_000;
    const codes = new Int32Array(n).map((_, i) => i % 13);
    const sel = new Set<number>();
    for (let i = 0; i < n; i += 3) sel.add(i);
    const t0 = performance.now();
    const mask = selectionMask(sel, n);
    countSelectedByCode(codes, mask, 13);
    binSelected(new Float32Array(n).map((_, i) => i), mask, Array.from({ length: 41 }, (_, i) => i * 2500));
    expect(performance.now() - t0).toBeLessThan(300);
  });
});
