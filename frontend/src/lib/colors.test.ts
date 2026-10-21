import { describe, it, expect } from "vitest";
import {
  viridisColor,
  mapCategoryToColor,
  mapValueToColor,
  colormapColor,
  isDivergingColormap,
  plotlyColorscale,
  CATEGORICAL_COLORS,
  COLORMAP_OPTIONS,
  COLORMAP_STOPS,
  DIVERGING_COLORMAP_OPTIONS,
  DIVERGING_COLORMAP_STOPS,
  type DivergingColormapName,
} from "./colors";

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

describe("diverging colormaps", () => {
  const names = Object.keys(DIVERGING_COLORMAP_STOPS) as DivergingColormapName[];
  it("are typed apart from the sequential maps and listed only in their own picker", () => {
    expect(names).toEqual(["RdBu", "PuOr", "BrBG"]);
    for (const n of names) {
      expect(n in COLORMAP_STOPS).toBe(false);
      expect(COLORMAP_OPTIONS.some((o) => (o.value as string) === n)).toBe(false);
      expect(DIVERGING_COLORMAP_OPTIONS.some((o) => o.value === n)).toBe(true);
      expect(isDivergingColormap(n)).toBe(true);
    }
    expect(isDivergingColormap("viridis")).toBe(false);
  });
  it("have 11 valid ColorBrewer stops with a near-neutral middle", () => {
    for (const n of names) {
      const stops = DIVERGING_COLORMAP_STOPS[n];
      expect(stops.length).toBe(11);
      expect(stops.every((c) => c.length === 3 && c.every(inByte))).toBe(true);
      const [r, g, b] = colormapColor(n, 0.5);
      expect(Math.min(r, g, b)).toBeGreaterThan(240); // white-ish centre
      expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeLessThan(6);
    }
  });
  it("RdBu runs blue (low) to red (high), PuOr purple to orange, BrBG brown to teal", () => {
    expect(colormapColor("RdBu", 0)).toEqual([5, 48, 97]);
    expect(colormapColor("RdBu", 1)).toEqual([103, 0, 31]);
    expect(colormapColor("PuOr", 0)).toEqual([45, 0, 75]);
    expect(colormapColor("PuOr", 1)).toEqual([127, 59, 8]);
    expect(colormapColor("BrBG", 0)).toEqual([84, 48, 5]);
    expect(colormapColor("BrBG", 1)).toEqual([0, 60, 48]);
  });
  it("feed mapValueToColor and the Plotly colorscale like the sequential maps", () => {
    expect(mapValueToColor(0, -2, 2, "RdBu").slice(0, 3)).toEqual(colormapColor("RdBu", 0.5));
    expect(mapValueToColor(-2, -2, 2, "RdBu").slice(0, 3)).toEqual([5, 48, 97]);
    const cs = plotlyColorscale("PuOr");
    expect(cs.length).toBe(11);
    expect(cs[0]).toEqual([0, "rgb(45,0,75)"]);
    expect(cs[10]).toEqual([1, "rgb(127,59,8)"]);
  });
});
