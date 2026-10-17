import { describe, it, expect } from "vitest";
import { fitAvoiding, HIDDEN_MAX } from "./scatterFit";

/** Screen position of a data point under a fit (OrthographicView, flipY false). */
const project = (fit: { target: number[]; zoom: number }, x: number, y: number, w: number, h: number): [number, number] => {
  const s = 2 ** fit.zoom;
  return [w / 2 + (x - fit.target[0]!) * s, h / 2 - (y - fit.target[1]!) * s];
};

/** A uniform grid of points filling [0, 10] x [0, 10]. */
function grid(n = 50): Float32Array {
  const out = new Float32Array(n * n * 2);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      out[(i * n + j) * 2] = (10 * i) / (n - 1);
      out[(i * n + j) * 2 + 1] = (10 * j) / (n - 1);
    }
  return out;
}

describe("fitAvoiding", () => {
  it("fits the bounds with 5% padding, centred, without a box", () => {
    const fit = fitAvoiding(grid(), 220, 110);
    expect(fit.target).toEqual([5, 5, 0]);
    expect(2 ** fit.zoom).toBeCloseTo(110 / 11, 10); // height-limited
  });

  it("keeps the plain fit when the box hides no cells", () => {
    // Points only in the top half of a square plot; the box covers the empty bottom-left.
    const top = new Float32Array([0, 10, 10, 10, 5, 9, 0, 0.0001 + 9]);
    const plain = fitAvoiding(top, 200, 200);
    expect(fitAvoiding(top, 200, 200, { x0: 0, y0: 150, x1: 60, y1: 200 })).toEqual(plain);
  });

  it("moves the cells beside a legend that would hide them, into the larger strip", () => {
    const pts = grid();
    const w = 400;
    const h = 200;
    const legend = { x0: 0, y0: 120, x1: 80, y1: 200 }; // bottom-left
    const fit = fitAvoiding(pts, w, h, legend);
    // Right of the legend (316 x 200 px) beats above it (400 x 116 px) for square data.
    for (const [x, y] of [[0, 0], [0, 10], [10, 0], [10, 10]] as const) {
      const [sx, sy] = project(fit, x, y, w, h);
      expect(sx).toBeGreaterThan(legend.x1);
      expect(sx).toBeLessThanOrEqual(w);
      expect(sy).toBeGreaterThanOrEqual(0);
      expect(sy).toBeLessThanOrEqual(h);
    }
    expect(2 ** fit.zoom).toBeCloseTo(200 / 11, 10); // still height-limited: no zoom lost
  });

  it("puts the cells above a wide, short legend", () => {
    const fit = fitAvoiding(grid(), 200, 400, { x0: 0, y0: 340, x1: 190, y1: 400 });
    const [, bottom] = project(fit, 0, 0, 200, 400);
    expect(bottom).toBeLessThan(340);
  });

  it("keeps the plain fit when avoiding the box would halve the plot", () => {
    const pts = grid();
    const plain = fitAvoiding(pts, 200, 200);
    expect(fitAvoiding(pts, 200, 200, { x0: 0, y0: 0, x1: 150, y1: 150 })).toEqual(plain);
  });

  it("tolerates a few hidden cells", () => {
    const pts = grid(); // 2500 cells; a corner box over one of them stays below HIDDEN_MAX
    const plain = fitAvoiding(pts, 200, 200);
    const [sx, sy] = project(plain, 0, 0, 200, 200);
    expect(HIDDEN_MAX * 2500).toBeGreaterThanOrEqual(1);
    expect(fitAvoiding(pts, 200, 200, { x0: sx - 1, y0: sy - 1, x1: sx + 1, y1: sy + 1 })).toEqual(plain);
  });
});
