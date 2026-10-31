import { describe, expect, it } from "vitest";
import type { VelocityGrid } from "@/api/velocity";
import { arrowColor, arrowFactor, arrowSegments, arrowsBlocked } from "./velocityArrows";

const grid = (u: number[], v: number[]): VelocityGrid => ({
  embedding: "X_umap",
  density: 1,
  n: u.length,
  x: u.map((_, i) => i),
  y: u.map(() => 0),
  u,
  v,
  step: 0.5,
  length_p95: 2,
  reprojected: false,
});

describe("velocity arrows", () => {
  it("draws a shaft and two head strokes per arrow, pointing along the vector", () => {
    const seg = arrowSegments(grid([2], [0]), 1);
    expect(seg).toHaveLength(3);
    const f = arrowFactor(grid([2], [0]), 1);
    expect(seg[0]).toEqual({ from: [0, 0], to: [2 * f, 0] });
    // The head strokes come back from the tip, one to each side.
    expect(seg[1]!.to[0]).toBeLessThan(seg[1]!.from[0]);
    expect(seg[2]!.to[0]).toBeLessThan(seg[2]!.from[0]);
    expect(Math.sign(seg[1]!.to[1])).toBe(-Math.sign(seg[2]!.to[1]));
  });

  it("scale lengthens every arrow, and the typical arrow stays near the grid step", () => {
    expect(arrowFactor(grid([2], [0]), 2)).toBeCloseTo(2 * arrowFactor(grid([2], [0]), 1));
    expect(2 * arrowFactor(grid([2], [0]), 1)).toBeCloseTo(0.5 * 1.6);
  });

  it("drops arrows too short to show a direction", () => {
    expect(arrowSegments(grid([2, 0.01], [0, 0]), 1)).toHaveLength(3);
    expect(arrowSegments({ ...grid([0], [0]), length_p95: 0 }, 1)).toHaveLength(0);
  });

  it("says why nothing is drawn on 3D, a PC pair or the tissue", () => {
    expect(arrowsBlocked(3, "X_umap_3d", false)).toMatch(/3D/);
    expect(arrowsBlocked(2, "X_pca", true)).toMatch(/PC1/);
    expect(arrowsBlocked(2, "spatial", false)).toMatch(/tissue/);
    expect(arrowsBlocked(2, "X_umap", false)).toBeNull();
  });

  it("reads on both backgrounds", () => {
    expect(arrowColor("white")[0]).toBeLessThan(60);
    expect(arrowColor("dark")[0]).toBeGreaterThan(200);
  });
});
