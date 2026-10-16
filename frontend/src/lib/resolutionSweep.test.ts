import { describe, it, expect } from "vitest";
import { bestSilhouette, formatResolutions, parseResolutions } from "./resolutionSweep";

const limits = { min: 0.01, max: 5, maxCount: 8 };

describe("parseResolutions", () => {
  it("reads commas or spaces, rounds to 0.01, dedupes and sorts", () => {
    expect(parseResolutions("1.0, 0.2 0.4;0.204", limits)).toEqual({ values: [0.2, 0.4, 1], error: null });
  });

  it("explains what is wrong", () => {
    expect(parseResolutions("  ", limits).error).toMatch(/at least one/);
    expect(parseResolutions("0.5, abc", limits).error).toMatch(/"abc" is not a number/);
    expect(parseResolutions("0, 1", limits).error).toMatch(/between 0.01 and 5/);
    expect(parseResolutions("0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9", limits).error).toMatch(/at most 8/);
  });
});

describe("formatResolutions", () => {
  it("writes whole numbers with one decimal, as the column names do", () => {
    expect(formatResolutions([0.2, 1, 1.5])).toBe("0.2, 1.0, 1.5");
  });
});

describe("bestSilhouette", () => {
  it("finds the highest score, skipping rows without one", () => {
    expect(bestSilhouette([{ silhouette: null }, { silhouette: 0.2 }, { silhouette: 0.31 }, { silhouette: 0.1 }])).toBe(2);
    expect(bestSilhouette([{ silhouette: null }])).toBe(-1);
  });
});
