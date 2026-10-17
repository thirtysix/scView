import { describe, it, expect } from "vitest";
import { sameValues } from "./useEmbedding";

describe("sameValues", () => {
  it("matches equal coordinates, NaN included, and nothing else", () => {
    const a = new Float32Array([1, 2, NaN, 4]);
    expect(sameValues(a, new Float32Array([1, 2, NaN, 4]))).toBe(true);
    expect(sameValues(a, new Float32Array([1, 2, NaN, 5]))).toBe(false);
    expect(sameValues(a, new Float32Array([1, 2, NaN]))).toBe(false);
    expect(sameValues(null, a)).toBe(false);
  });
});
