import { describe, it, expect } from "vitest";
import { DOT_MAX_PX, MIN_FRACTION, dotSize, heatmapBands, markerColours, scalePerGene } from "./markerPlot";

describe("heatmapBands", () => {
  it("fills the height with the plot when there is nothing above it", () => {
    expect(heatmapBands(400, 0, 0)).toEqual({ main: [0, 1], strip: null, tree: null, extraPx: 0 });
  });

  it("stacks plot, gap, strip, gap, tree from the bottom, keeping the plot's px", () => {
    // 400 + 4 + 12 + 4 + 80 = 500 px in all
    const b = heatmapBands(400, 12, 80);
    expect(b.extraPx).toBe(100);
    expect(b.main).toEqual([0, 400 / 500]);
    expect(b.strip).toEqual([404 / 500, 416 / 500]);
    expect(b.tree).toEqual([420 / 500, 1]);
  });

  it("puts the strip at the top when there is no tree", () => {
    const b = heatmapBands(400, 12, 0);
    expect(b.strip![1]).toBe(1);
    expect(b.tree).toBeNull();
  });
});

describe("dotSize", () => {
  it("gives dot area proportional to the share, the same for every plot", () => {
    expect(dotSize(1)).toBe(DOT_MAX_PX);
    expect(dotSize(0.25)).toBeCloseTo(DOT_MAX_PX / 2);
    // area ratio = share ratio
    expect((dotSize(0.8) / dotSize(0.2)) ** 2).toBeCloseTo(4);
  });
  it("clamps to 0-1 and keeps a tiny dot visible", () => {
    expect(dotSize(2)).toBe(DOT_MAX_PX);
    expect(dotSize(0)).toBe(2);
    expect(dotSize(Number.NaN)).toBe(2);
  });
});

describe("scalePerGene", () => {
  it("leaves missing values out of the gene's range", () => {
    expect(scalePerGene([[1, null], [3, 2], [2, 4]])).toEqual([[0, null], [1, 0], [0.5, 1]]);
  });
});

const rows = {
  mean: [[0.1, 2], [1.5, 0]],
  meanExpressing: [[2.0, 2.5], [3.0, 0]],
  fraction: [[0.03, 0.8], [0.5, 0]],
};

describe("markerColours", () => {
  it("scales the mean over all cells per gene, as before", () => {
    const c = markerColours(rows, "all", "gene", [0, 5]);
    expect(c.z).toEqual([[0, 1], [1, 0]]);
    expect([c.cmin, c.cmax]).toEqual([0, 1]);
    expect(c.title).toContain("scaled per gene");
  });

  it("keeps the values on a fixed range that does not depend on the genes shown", () => {
    const c = markerColours(rows, "all", "fixed", [0, 5]);
    expect(c.z).toEqual(rows.mean);
    expect([c.cmin, c.cmax]).toEqual([0, 5]);
    expect(c.title).toContain("fixed 0-5");
    const one = markerColours({ ...rows, mean: rows.mean.map((r) => [r[0]!]) }, "all", "fixed", [0, 5]);
    expect([one.cmin, one.cmax]).toEqual([0, 5]);
  });

  it("shows the level in expressing cells, with no colour under the minimum share", () => {
    expect(rows.fraction[0]![0]!).toBeLessThan(MIN_FRACTION);
    const c = markerColours(rows, "expressing", "fixed", [0, 5]);
    expect(c.z).toEqual([[null, 2.5], [3.0, null]]);
    expect(c.title).toContain("expressing cells");
    expect(c.caption).toContain("mean in the cells that express it");
  });

  it("falls back to the mean over all cells when the server sent no expressing mean", () => {
    const c = markerColours({ ...rows, meanExpressing: null }, "expressing", "fixed", [0, 5]);
    expect(c.z).toEqual(rows.mean);
  });

  it("names proteins in the per-protein caption", () => {
    expect(markerColours(rows, "all", "gene", [0, 5], "protein").caption).toContain("per protein");
  });
});
