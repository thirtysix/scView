import { describe, it, expect } from "vitest";
import {
  colorScaleCaption,
  mergedStats,
  overlayStats,
  resolveColorScale,
  scatterScale,
  wantsDiverging,
  type ColorScaleSettings,
  type OverlayStats,
} from "./colorScale";

const settings = (over: Partial<ColorScaleSettings> = {}): ColorScaleSettings => ({
  sequential: "viridis",
  diverging: "RdBu",
  signedMode: "auto",
  clip: { preset: "none" },
  ...over,
});

describe("overlayStats", () => {
  it("sorts a copy of the finite values and notes the signs", () => {
    const values = Float32Array.from([3, NaN, -1, 0, Infinity, 2]);
    const s = overlayStats(values);
    expect(Array.from(s.sorted)).toEqual([-1, 0, 2, 3]);
    expect([s.n, s.nAll, s.min, s.max, s.hasNegative, s.hasPositive]).toEqual([4, 6, -1, 3, true, true]);
    expect(Array.from(values)).toEqual([3, NaN, -1, 0, Infinity, 2]); // untouched
  });
  it("is empty for no values, all-NaN values, or null", () => {
    for (const v of [null, new Float32Array(0), Float32Array.from([NaN, NaN])]) {
      const s = overlayStats(v);
      expect(s.n).toBe(0);
      expect(Number.isNaN(s.min)).toBe(true);
      expect([s.hasNegative, s.hasPositive]).toEqual([false, false]);
    }
  });
  it("pools several overlays for a shared scale", () => {
    const m = mergedStats([overlayStats([1, 2]), overlayStats([-3, 0.5])]);
    expect(Array.from(m.sorted)).toEqual([-3, 0.5, 1, 2]);
    expect([m.min, m.max, m.hasNegative]).toEqual([-3, 2, true]);
  });
});

describe("wantsDiverging (the auto rule)", () => {
  const signed = overlayStats([-1, 0, 2]);
  const positive = overlayStats([0, 1, 2]);
  const negative = overlayStats([-2, -1]);
  it("auto: a score or protein with values of both signs, never expression or a QC column", () => {
    expect(wantsDiverging("score", signed, "auto")).toBe(true);
    expect(wantsDiverging("protein", signed, "auto")).toBe(true);
    expect(wantsDiverging("expression", signed, "auto")).toBe(false);
    expect(wantsDiverging("obs", signed, "auto")).toBe(false);
    expect(wantsDiverging("score", positive, "auto")).toBe(false);
    expect(wantsDiverging("score", negative, "auto")).toBe(false);
  });
  it("sequential: never; diverging: any kind with both signs", () => {
    expect(wantsDiverging("score", signed, "sequential")).toBe(false);
    expect(wantsDiverging("expression", signed, "diverging")).toBe(true);
    expect(wantsDiverging("expression", positive, "diverging")).toBe(false);
  });
});

describe("resolveColorScale", () => {
  it("sequential: the data range on the sequential map, no centre", () => {
    const r = resolveColorScale("expression", overlayStats([0, 1, 4]), settings({ sequential: "magma" }));
    expect(r).toMatchObject({ colormap: "magma", diverging: false, min: 0, max: 4, centre: null, clip: null, n: 3 });
  });
  it("diverging: symmetric about 0 from the larger magnitude, on the diverging map", () => {
    const r = resolveColorScale("score", overlayStats([-1, 0.5, 3]), settings({ diverging: "PuOr" }));
    expect(r).toMatchObject({ colormap: "PuOr", diverging: true, min: -3, max: 3, centre: 0, dataMin: -1, dataMax: 3 });
    const neg = resolveColorScale("score", overlayStats([-4, 1]), settings());
    expect([neg.min, neg.max]).toEqual([-4, 4]);
  });
  it("percentile clipping narrows the range and counts the cells outside it", () => {
    const values = Float32Array.from({ length: 1000 }, (_, i) => i); // 0..999
    const r = resolveColorScale("expression", overlayStats(values), settings({ clip: { preset: "p1" } }));
    expect(r.min).toBeCloseTo(9.99, 2);
    expect(r.max).toBeCloseTo(989.01, 2);
    expect(r.clip).toEqual({ preset: "p1", fracBelow: 0.01, fracAbove: 0.01 });
    const r5 = resolveColorScale("expression", overlayStats(values), settings({ clip: { preset: "p5" } }));
    expect(r5.clip!.fracAbove).toBeCloseTo(0.05, 3);
  });
  it("clips a diverging scale symmetrically, the shares outside against the final range", () => {
    const values = Float32Array.from({ length: 200 }, (_, i) => i - 100); // -100..99
    const r = resolveColorScale("score", overlayStats(values), settings({ clip: { preset: "p5" } }));
    // 5th pct -90.05, 95th 89.05: the larger magnitude wins
    expect(r.max).toBeCloseTo(90.05, 2);
    expect(r.min).toBeCloseTo(-90.05, 2);
    expect(r.clip!.fracBelow).toBeCloseTo(0.05, 3); // -100..-91
    expect(r.clip!.fracAbove).toBeCloseTo(0.045, 3); // 91..99
  });
  it("manual: the typed ends, a missing end being the data's, reversed ends swapped", () => {
    const st = overlayStats([0, 1, 2, 3, 4]);
    expect(resolveColorScale("expression", st, settings({ clip: { preset: "manual", max: 2 } }))).toMatchObject({
      min: 0, max: 2, clip: { preset: "manual", fracBelow: 0, fracAbove: 0.4 },
    });
    expect(resolveColorScale("expression", st, settings({ clip: { preset: "manual", min: 3, max: 1 } }))).toMatchObject({ min: 1, max: 3 });
    // wider than the data: still reported as set by hand
    const wide = resolveColorScale("expression", st, settings({ clip: { preset: "manual", min: -1, max: 10 } }));
    expect(wide.clip).toEqual({ preset: "manual", fracBelow: 0, fracAbove: 0 });
  });
  it("all-equal values: a flat range, nothing clipped", () => {
    const r = resolveColorScale("expression", overlayStats([2, 2, 2]), settings({ clip: { preset: "p1" } }));
    expect([r.min, r.max, r.clip]).toEqual([2, 2, null]);
  });
  it("no finite values: the 0-1 fallback", () => {
    const r = resolveColorScale("score", overlayStats([NaN]), settings({ clip: { preset: "p2" } }));
    expect(r).toMatchObject({ min: 0, max: 1, clip: null, n: 0 });
  });
  it("does not clip when the percentiles equal the data ends (ties)", () => {
    const r = resolveColorScale("expression", overlayStats([0, 0, 0, 0, 5, 5, 5, 5]), settings({ clip: { preset: "p5" } }));
    expect(r.clip).toBeNull();
  });
  it("scatterScale is the minimal prop", () => {
    const r = resolveColorScale("score", overlayStats([-1, 1]), settings());
    expect(scatterScale(r)).toEqual({ colormap: "RdBu", min: -1, max: 1 });
  });
});

describe("colorScaleCaption", () => {
  const stats = (vals: number[]): OverlayStats => overlayStats(vals);
  it("states the range, and that it is the data range when unclipped", () => {
    expect(colorScaleCaption(resolveColorScale("expression", stats([0, 3.21]), settings()))).toBe("Colour range 0 to 3.21.");
  });
  it("says the clipping and the share of cells outside", () => {
    const values = Float32Array.from({ length: 1000 }, (_, i) => i / 100);
    const r = resolveColorScale("expression", overlayStats(values), settings({ clip: { preset: "p1" } }));
    expect(colorScaleCaption(r)).toBe(
      "Colour range 0.0999 to 9.89, clipped at the 1st–99th percentiles (data 0 to 9.99); 1.0% of cells above, 1.0% below.",
    );
  });
  it("says a diverging scale is centred, with the data range when it differs", () => {
    const r = resolveColorScale("score", stats([-1, 2]), settings());
    expect(colorScaleCaption(r)).toBe("Colour range -2 to 2, centred on 0 (data -1 to 2).");
  });
  it("manual: set by hand, no cells outside when none", () => {
    const r = resolveColorScale("expression", stats([0, 1]), settings({ clip: { preset: "manual", min: -1, max: 2 } }));
    expect(colorScaleCaption(r)).toBe("Colour range -1 to 2, set by hand (data 0 to 1); no cells outside.");
  });
  it("says when there is nothing to colour", () => {
    expect(colorScaleCaption(resolveColorScale("obs", stats([]), settings()))).toBe("No values to colour by.");
  });
});
