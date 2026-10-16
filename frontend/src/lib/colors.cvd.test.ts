import { describe, it, expect } from "vitest";
import { CATEGORICAL_COLORS } from "./colors";

type RGB = readonly [number, number, number];

// Machado, Oliveira & Fernandes 2009, severity 1.0, applied to linear RGB.
const DEUTAN = [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.01182, 0.04294, 0.968881]];
const PROTAN = [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]];

const toLinear = (v: number) => { const c = v / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

function simulate(rgb: RGB, m: number[][]): [number, number, number] {
  const lin = rgb.map(toLinear);
  return m.map((row) => clamp01(row[0]! * lin[0]! + row[1]! * lin[1]! + row[2]! * lin[2]!)) as [number, number, number];
}

function labFromLinear([r, g, b]: [number, number, number]): [number, number, number] {
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const f = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

/** Smallest pairwise CIE76 dE among the first n colours after CVD simulation. */
function minDeltaE(colors: readonly RGB[], m: number[][], n = 8): number {
  const labs = colors.slice(0, n).map((c) => labFromLinear(simulate(c, m)));
  let min = Infinity;
  for (let i = 0; i < labs.length; i++)
    for (let j = i + 1; j < labs.length; j++)
      min = Math.min(min, Math.hypot(labs[i]![0] - labs[j]![0], labs[i]![1] - labs[j]![1], labs[i]![2] - labs[j]![2]));
  return min;
}

// CIE76 dE >= 6 = clearly separable side by side. Measured: new palette min 7.05 (deutan) / 18.1 (protan); old palette 4.48 / 11.2.
const MIN_DE = 6;

// The pre-C008 palette (red/green pairs), kept to prove the test discriminates.
const OLD_PALETTE: RGB[] = [
  [0, 117, 220], [230, 25, 75], [60, 180, 75], [255, 127, 0], [145, 30, 180], [0, 183, 183], [240, 163, 10], [233, 90, 196],
];

describe("CATEGORICAL_COLORS CVD safety", () => {
  it("first 8 stay distinguishable under deuteranopia", () => {
    expect(minDeltaE(CATEGORICAL_COLORS, DEUTAN)).toBeGreaterThanOrEqual(MIN_DE);
  });
  it("first 8 stay distinguishable under protanopia", () => {
    expect(minDeltaE(CATEGORICAL_COLORS, PROTAN)).toBeGreaterThanOrEqual(MIN_DE);
  });
  it("the old red/green palette fails the same check", () => {
    expect(Math.min(minDeltaE(OLD_PALETTE, DEUTAN), minDeltaE(OLD_PALETTE, PROTAN))).toBeLessThan(MIN_DE);
  });
  it("has no duplicate colours", () => {
    const keys = CATEGORICAL_COLORS.map((c) => c.join(","));
    expect(new Set(keys).size).toBe(keys.length);
  });
  it("still has 20 entries of valid bytes", () => {
    expect(CATEGORICAL_COLORS.length).toBe(20);
    expect(CATEGORICAL_COLORS.every((c) => c.length === 3 && c.every((v) => Number.isInteger(v) && v >= 0 && v <= 255))).toBe(true);
  });
});
