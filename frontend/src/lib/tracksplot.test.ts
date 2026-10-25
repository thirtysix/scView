import { describe, it, expect } from "vitest";
import { TRACKS_MAX_GENES, trackOrder, tracksCaption, tracksFigure, withAlpha } from "./tracksplot";
import type { HeatmapCells } from "./cellHeatmap";

/** 5 sampled cells: B (3) then T (2); two genes. */
const DATA: HeatmapCells = {
  meta: {
    groupby: "cell_type", layer: "raw", genes: ["CD3E", "MS4A1"], missing_genes: [], groups: ["B", "T"],
    n_sampled: [3, 2], n_total: [3, 40], n_cells: 43, scale: "none", value_range: [0, 4], seed: 0,
    per_group: 200, max_cells: 5000,
  },
  cells: Int32Array.from([10, 11, 12, 1, 2]),
  codes: Int32Array.from([0, 0, 0, 1, 1]),
  values: [Float32Array.from([0, 0.5, 0, 3, 2]), Float32Array.from([2, 1, 4, 0, 0])],
};

type Trace = { type: string; x: number[]; y: number[] | string[]; z?: number[][]; yaxis: string; fillcolor?: string; fill?: string };

describe("trackOrder", () => {
  it("keeps the groups and orders cells within each group", () => {
    expect(trackOrder(DATA, { by: "dataset" })).toEqual([0, 1, 2, 3, 4]);
    // by MS4A1, lowest first, within B then within T (a tie keeps dataset order)
    expect(trackOrder(DATA, { by: "gene", gene: 1 })).toEqual([1, 0, 2, 3, 4]);
    expect(trackOrder(DATA, { by: "gene", gene: 0 })).toEqual([0, 2, 1, 4, 3]);
  });
});

describe("tracksFigure", () => {
  it("draws one filled stretch per gene and group, each gene on its own axis and scale", () => {
    const fig = tracksFigure(DATA, ["#ff0000", "rgb(0, 0, 255)"]);
    const traces = fig.data as Trace[];
    const lines = traces.filter((t) => t.type === "scatter");
    expect(lines).toHaveLength(4); // 2 genes x 2 groups
    expect(lines.map((t) => t.yaxis)).toEqual(["y", "y", "y2", "y2"]);
    expect(lines[0]!.x).toEqual([0, 1, 2]);
    expect(lines[1]!.x).toEqual([3, 4]);
    expect(lines[1]!.y).toEqual([3, 2]);
    expect(lines[0]!.fill).toBe("tozeroy");
    expect(lines[0]!.fillcolor).toBe("rgba(255, 0, 0, 0.8)");
    expect(lines[1]!.fillcolor).toBe("rgba(0, 0, 255, 0.8)");
    const layout = fig.layout as Record<string, { range?: number[]; ticktext?: string[]; domain?: number[] }>;
    expect(layout.yaxis!.range![1]).toBeCloseTo(3 * 1.05);
    expect(layout.yaxis2!.range![1]).toBeCloseTo(4 * 1.05);
    expect(layout.yaxis!.ticktext).toEqual(["3.0"]);
    // gene rows top to bottom, the group band at the bottom
    expect(layout.yaxis!.domain![0]).toBeGreaterThan(layout.yaxis2!.domain![1]! - 1e-9);
    const band = traces.find((t) => t.type === "heatmap")!;
    expect(band.yaxis).toBe("y3");
    expect(band.z).toEqual([[0, 0, 0, 1, 1]]);
    expect(layout.yaxis3!.domain![0]).toBe(0);
    const xaxis = (fig.layout as { xaxis: { ticktext: string[]; tickvals: number[]; anchor: string } }).xaxis;
    expect(xaxis.anchor).toBe("y3"); // group names under the band
    expect(xaxis.ticktext).toEqual(["B", "T"]);
    expect(xaxis.tickvals).toEqual([1, 3.5]);
    expect((fig.layout as { shapes: unknown[] }).shapes).toHaveLength(1); // one group boundary
    const notes = (fig.layout as { annotations: { text: string }[] }).annotations;
    expect(notes.map((a) => a.text)).toEqual(["CD3E", "MS4A1"]);
  });

  it("follows the chosen order within groups", () => {
    const fig = tracksFigure(DATA, ["#ff0000", "#0000ff"], { by: "gene", gene: 1 });
    const first = (fig.data as Trace[])[0]!;
    expect(first.y).toEqual([0.5, 0, 0]); // CD3E of cells 1, 0, 2
  });

  it("caps the rows", () => {
    const many: HeatmapCells = {
      ...DATA,
      meta: { ...DATA.meta, genes: Array.from({ length: 40 }, (_, i) => `G${i}`) },
      values: Array.from({ length: 40 }, () => Float32Array.from([1, 0, 0, 0, 1])),
    };
    const fig = tracksFigure(many, ["#ff0000", "#0000ff"]);
    expect((fig.layout as { annotations: unknown[] }).annotations).toHaveLength(TRACKS_MAX_GENES);
  });
});

describe("helpers", () => {
  it("adds alpha to a colour", () => {
    expect(withAlpha("#0a0B0c", 0.5)).toBe("rgba(10, 11, 12, 0.5)");
    expect(withAlpha("rgb(1, 2, 3)", 1)).toBe("rgba(1, 2, 3, 1)");
    expect(withAlpha("red", 0.5)).toBe("red");
  });

  it("says how cells are ordered", () => {
    expect(tracksCaption({ by: "dataset" }, ["A"])).toContain("in dataset order");
    expect(tracksCaption({ by: "gene", gene: 0 }, ["CD3E"])).toContain("by CD3E (lowest first)");
  });
});
