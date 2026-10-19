import { describe, it, expect } from "vitest";
import { RecordBatch, Schema, Table, tableFromArrays, tableToIPC } from "apache-arrow";
import {
  cellHeatmapCaption,
  cellHeatmapFigure,
  decodeHeatmapCells,
  DIVERGING_COLORSCALE,
  discreteColorscale,
  groupSpans,
  type HeatmapCellsMeta,
} from "./cellHeatmap";

const META: HeatmapCellsMeta = {
  groupby: "cell_type", layer: "raw", genes: ["cell", "LYZ"], missing_genes: ["NOPE"],
  groups: ["B", "T"], n_sampled: [2, 1], n_total: [2, 40], n_cells: 45, scale: "zscore",
  value_range: [-2.5, 2.5], seed: 0, per_group: 2, max_cells: 5000,
};

/** The endpoint's layout: cell, group, then one column per gene; meta in the schema. */
function ipc(meta: HeatmapCellsMeta, columns: Record<string, Int32Array | Float32Array>): Uint8Array {
  const t = tableFromArrays(columns);
  const schema = new Schema(t.schema.fields, new Map([["scview", JSON.stringify(meta)]]));
  const batches = t.batches.map((b) => new RecordBatch(schema, b.data));
  return tableToIPC(new Table(schema, batches), "stream");
}

describe("decodeHeatmapCells", () => {
  it("reads the columns by position, so a gene may be called 'cell'", () => {
    // tableFromArrays needs distinct keys: the first gene column is named "g_cell"
    // here; decoding must not care what the columns are called.
    const buf = ipc(META, {
      cell: Int32Array.from([7, 9, 3]),
      group: Int32Array.from([0, 0, 1]),
      g_cell: Float32Array.from([1, -1, 0]),
      LYZ: Float32Array.from([2.5, 0, -2.5]),
    });
    const out = decodeHeatmapCells(buf);
    expect(out.meta.genes).toEqual(["cell", "LYZ"]);
    expect(Array.from(out.cells)).toEqual([7, 9, 3]);
    expect(Array.from(out.codes)).toEqual([0, 0, 1]);
    expect(Array.from(out.values[0]!)).toEqual([1, -1, 0]);
    expect(Array.from(out.values[1]!)).toEqual([2.5, 0, -2.5]);
  });

  it("refuses a stream without its description", () => {
    const buf = tableToIPC(tableFromArrays({ cell: Int32Array.from([1]) }), "stream");
    expect(() => decodeHeatmapCells(buf)).toThrow(/no description/);
  });
});

describe("groupSpans", () => {
  it("gives each group's first and one-past-last column", () => {
    expect(groupSpans([0, 0, 0, 1, 2, 2])).toEqual([
      { group: 0, start: 0, end: 3 },
      { group: 1, start: 3, end: 4 },
      { group: 2, start: 4, end: 6 },
    ]);
    expect(groupSpans([])).toEqual([]);
  });
});

describe("discreteColorscale", () => {
  it("gives code k a flat band around (k + 0.5) / n for z from -0.5 to n - 0.5", () => {
    const scale = discreteColorscale(["red", "green", "blue", "black"]);
    // Plotly interpolates between stops: the colour at t for code k must be its own.
    const at = (t: number) => {
      for (let i = 0; i < scale.length - 1; i++) {
        const [t0, c0] = scale[i]!;
        const [t1, c1] = scale[i + 1]!;
        if (t >= t0 && t <= t1) return c0 === c1 ? c0 : "blend";
      }
      return "none";
    };
    expect([0, 1, 2, 3].map((k) => at((k + 0.5) / 4))).toEqual(["red", "green", "blue", "black"]);
    expect(scale[0]![0]).toBe(0);
    expect(scale.at(-1)![0]).toBe(1);
    // stops never go backwards
    expect(scale.every(([t], i) => i === 0 || t >= scale[i - 1]![0])).toBe(true);
  });
});

describe("cellHeatmapFigure", () => {
  const data = {
    meta: { ...META, genes: ["CD3E", "LYZ"], groups: ["B", "T", "NK"], n_sampled: [2, 3, 1], n_total: [2, 3, 1] },
    cells: Int32Array.from([0, 1, 2, 3, 4, 5]),
    codes: Int32Array.from([0, 0, 1, 1, 1, 2]),
    values: [Float32Array.from([1, 2, 3, 4, 5, 6]), Float32Array.from([-1, -2, -3, -4, -5, -6])],
  };
  type Fig = { data: Record<string, unknown>[]; layout: Record<string, Record<string, unknown> & unknown[]> };
  const SEQ: [number, string][] = [[0, "#fff"], [1, "#000"]];

  it("marks each group: a white line between groups, its name under the middle of its cells", () => {
    const fig = cellHeatmapFigure(data, ["#a00", "#0a0", "#00a"], SEQ) as unknown as Fig;
    const lines = fig.layout.shapes as unknown as { x0: number }[];
    expect(lines.map((s) => s.x0)).toEqual([1.5, 4.5]); // B | T at column 2, T | NK at column 5
    expect(fig.layout.xaxis!.tickvals).toEqual([0.5, 3, 5]);
    expect(fig.layout.xaxis!.ticktext).toEqual(["B", "T", "NK"]);
    const strip = fig.data[1]!;
    expect(strip.yaxis).toBe("y2");
    expect(strip.z).toEqual([[0, 0, 1, 1, 1, 2]]);
    expect(strip.text).toEqual([["B", "B", "T", "T", "T", "NK"]]);
    // the strip sits above the heatmap
    const heatDomain = fig.layout.yaxis!.domain as unknown as number[];
    const stripDomain = fig.layout.yaxis2!.domain as unknown as number[];
    expect(stripDomain[0]).toBeGreaterThan(heatDomain[1]!);
    expect(stripDomain[1]).toBe(1);
  });

  it("draws genes as rows top to bottom and cells as columns", () => {
    const fig = cellHeatmapFigure(data, ["#a00", "#0a0", "#00a"], SEQ) as unknown as Fig;
    const heat = fig.data[0]!;
    expect(heat.y).toEqual(["CD3E", "LYZ"]);
    expect((heat.z as number[][])[1]).toEqual([-1, -2, -3, -4, -5, -6]);
    expect(fig.layout.yaxis!.autorange).toBe("reversed");
  });

  it("centres a diverging scale at 0 for z-scores, a sequential one for values", () => {
    const z = (cellHeatmapFigure(data, [], SEQ) as unknown as Fig).data[0]!;
    expect(z).toMatchObject({ zmin: -2.5, zmax: 2.5, zmid: 0, colorscale: DIVERGING_COLORSCALE });
    const none = { ...data, meta: { ...data.meta, scale: "none" as const, value_range: [0, 5] as [number, number] } };
    const v = (cellHeatmapFigure(none, [], SEQ) as unknown as Fig).data[0]!;
    expect(v).toMatchObject({ zmin: 0, zmax: 5, colorscale: SEQ });
    expect(v.zmid).toBeUndefined();
  });
});

describe("cellHeatmapCaption", () => {
  it("says how many cells per group, of how many, and the seed", () => {
    const meta = { ...META, per_group: 200, n_sampled: [200, 60, 108], n_total: [998, 60, 108],
      groups: ["B", "Eryth", "pDC"], n_cells: 13836 };
    expect(cellHeatmapCaption(meta)).toBe(
      "200 cells per group (sampled), 368 of 13,836 cells, seed 0. " +
        "All cells of the smaller groups Eryth (60), pDC (108).",
    );
  });

  it("says when the total cap scaled the shares down", () => {
    const meta = { ...META, per_group: 500, max_cells: 1000, n_sampled: [500, 500], n_total: [4000, 4000],
      groups: ["A", "B"], n_cells: 8000 };
    expect(cellHeatmapCaption({ ...meta, n_sampled: [500, 500] })).not.toMatch(/scaled down/);
    expect(cellHeatmapCaption({ ...meta, max_cells: 600, n_sampled: [300, 300] })).toMatch(
      /^500 cells per group \(sampled\), 600 of 8,000 cells, seed 0\. Each group's share was scaled down to fit 600 cells\.$/,
    );
  });
});
