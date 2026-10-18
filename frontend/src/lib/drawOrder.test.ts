import { describe, it, expect } from "vitest";
import { gatherByOrder, selectedLast, shuffledOrder } from "./drawOrder";

const sorted = (a: ArrayLike<number>) => Array.from(a).sort((x, y) => x - y);

describe("shuffledOrder", () => {
  it("is a permutation of the cells, the same every time", () => {
    const order = shuffledOrder(1000);
    expect(sorted(order)).toEqual(Array.from({ length: 1000 }, (_, i) => i));
    expect(Array.from(shuffledOrder(1000))).toEqual(Array.from(order));
    // Shuffled: few cells keep their slot, and the last stored ones are not all drawn last.
    expect(Array.from(order).filter((c, s) => c === s).length).toBeLessThan(20);
    expect(Array.from(order.subarray(900)).filter((c) => c >= 900).length).toBeLessThan(40);
  });

  it("shuffles a sample's cells without changing the sample", () => {
    const sample = Uint32Array.from([3, 8, 20, 21, 40, 77]);
    const order = shuffledOrder(sample);
    expect(sorted(order)).toEqual(Array.from(sample));
    expect(Array.from(sample)).toEqual([3, 8, 20, 21, 40, 77]); // input untouched
    expect(Array.from(shuffledOrder(sample, 1))).not.toEqual(Array.from(order)); // the seed decides
  });

  it("handles no cells and one cell", () => {
    expect(shuffledOrder(0).length).toBe(0);
    expect(Array.from(shuffledOrder(1))).toEqual([0]);
  });
});

describe("selectedLast", () => {
  const order = Uint32Array.from([5, 2, 9, 0, 7, 3]);

  it("moves the selected cells to the end, each part in its order", () => {
    expect(Array.from(selectedLast(order, new Set([0, 5])))).toEqual([2, 9, 7, 3, 5, 0]);
  });

  it("ignores selected cells that are not drawn, and returns the order itself without a selection", () => {
    expect(Array.from(selectedLast(order, new Set([9, 100])))).toEqual([5, 2, 0, 7, 3, 9]);
    expect(selectedLast(order, null)).toBe(order);
    expect(selectedLast(order, new Set())).toBe(order);
    expect(selectedLast(order, new Set([100]))).toBe(order);
  });
});

describe("gatherByOrder", () => {
  it("takes each cell's values in draw order, keeping the array type", () => {
    const pos = Float32Array.from([0, 0.5, 1, 1.5, 2, 2.5]);
    const xy = gatherByOrder(pos, Uint32Array.from([2, 0, 1]), 2);
    expect(xy).toBeInstanceOf(Float32Array);
    expect(Array.from(xy)).toEqual([2, 2.5, 0, 0.5, 1, 1.5]);
    const rgba = gatherByOrder(Uint8Array.from([1, 1, 1, 1, 2, 2, 2, 2]), Uint32Array.from([1]), 4);
    expect(rgba).toBeInstanceOf(Uint8Array);
    expect(Array.from(rgba)).toEqual([2, 2, 2, 2]);
  });
});

describe("draw order at 200k cells", () => {
  it("stays fast: shuffle, selection on top and both buffers in well under a frame budget", () => {
    const n = 200_000;
    const positions = new Float32Array(n * 2);
    const colors = new Uint8Array(n * 4);
    const selected = new Set(Array.from({ length: 20_000 }, (_, i) => i * 10));
    const t = performance.now();
    const order = selectedLast(shuffledOrder(n), selected);
    gatherByOrder(positions, order, 2);
    gatherByOrder(colors, order, 4);
    const ms = performance.now() - t;
    expect(order.length).toBe(n);
    expect(ms).toBeLessThan(500); // ~20 ms here; generous for a loaded test machine
  });
});
