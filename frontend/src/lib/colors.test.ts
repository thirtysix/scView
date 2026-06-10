import { describe, it, expect } from "vitest";
import { viridisColor, mapCategoryToColor, mapValueToColor, CATEGORICAL_COLORS } from "./colors";

const inByte = (n: number) => n >= 0 && n <= 255 && Number.isInteger(n);

describe("viridisColor", () => {
  it("returns in-range RGB bytes across the domain", () => {
    for (const t of [-1, 0, 0.25, 0.5, 0.75, 1, 2]) {
      const c = viridisColor(t);
      expect(c.every(inByte)).toBe(true);
    }
  });
  it("clamps out-of-range t to the endpoints", () => {
    expect(viridisColor(-5)).toEqual(viridisColor(0));
    expect(viridisColor(5)).toEqual(viridisColor(1));
  });
  it("goes from dark purple (low) to bright yellow (high)", () => {
    const [, , bLow] = viridisColor(0);
    const [rHigh, gHigh, bHigh] = viridisColor(1);
    expect(bLow).toBeGreaterThan(50); // purple has blue
    expect(rHigh).toBeGreaterThan(200); // yellow = high R+G
    expect(gHigh).toBeGreaterThan(200);
    expect(bHigh).toBeLessThan(100);
  });
});

describe("mapCategoryToColor", () => {
  it("wraps the palette and returns opaque RGBA", () => {
    const n = CATEGORICAL_COLORS.length;
    expect(mapCategoryToColor(0)).toEqual(mapCategoryToColor(n)); // wraps
    expect(mapCategoryToColor(0)[3]).toBe(255); // alpha
  });
  it("handles negative and non-finite indices without throwing", () => {
    expect(mapCategoryToColor(-1).every(inByte)).toBe(true);
    expect(mapCategoryToColor(NaN)).toEqual(mapCategoryToColor(0));
  });
});

describe("mapValueToColor", () => {
  it("maps min→low and max→high of the viridis ramp", () => {
    expect(mapValueToColor(0, 0, 10).slice(0, 3)).toEqual(viridisColor(0));
    expect(mapValueToColor(10, 0, 10).slice(0, 3)).toEqual(viridisColor(1));
  });
  it("degenerate range (max==min) maps to t=0", () => {
    expect(mapValueToColor(5, 5, 5).slice(0, 3)).toEqual(viridisColor(0));
  });
});
