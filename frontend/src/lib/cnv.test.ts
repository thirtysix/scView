import { describe, expect, it } from "vitest";
import {
  chromosomeTicks,
  decodeHeatmap,
  groupTicks,
  heatmapShapes,
  patternText,
  progressText,
  referenceColumns,
  scoreBoxTraces,
  suggestReferenceColumn,
  suggestReferenceGroups,
} from "./cnv";

function b64(codes: number[]): string {
  return btoa(String.fromCharCode(...codes.map((c) => (c < 0 ? c + 256 : c))));
}

describe("cnv helpers", () => {
  it("decodes int8 codes row by row with the scale", () => {
    const rows = decodeHeatmap({ values: b64([127, -127, 0, 64, -1, 1]), n_rows: 2, n_cols: 3, scale: 0.5 });
    expect(rows).toEqual([
      [63.5, -63.5, 0],
      [32, -0.5, 0.5],
    ]);
  });

  it("labels chromosomes at their middle and draws boundaries", () => {
    const h = {
      chromosomes: [
        { chromosome: "chr1", start: 0, end: 4 },
        { chromosome: "chrX", start: 4, end: 6 },
      ],
      groups: [
        { group: "T", n_sampled: 3, n_total: 30, reference: true },
        { group: "Epi", n_sampled: 2, n_total: 2, reference: false },
      ],
    };
    expect(chromosomeTicks(h)).toEqual({ vals: [1.5, 4.5], text: ["1", "X"] });
    expect(groupTicks(h)).toEqual({ vals: [1, 3.5], text: ["T (ref)", "Epi"] });
    const shapes = heatmapShapes(h);
    expect(shapes).toHaveLength(2);
    expect(shapes[0]).toMatchObject({ x0: 3.5, x1: 3.5 });
    expect(shapes[1]).toMatchObject({ y0: 2.5 });
  });

  it("builds one box per group from the server's quartiles", () => {
    const traces = scoreBoxTraces([
      { group: "T", n: 10, mean: 0.01, q1: 0.005, median: 0.01, q3: 0.015, lo: 0, hi: 0.02, reference: true, ratio: 1 },
      { group: "Epi", n: 5, mean: 0.05, q1: 0.04, median: 0.05, q3: 0.06, lo: 0.03, hi: 0.08, reference: false, ratio: 5 },
    ]) as unknown as Record<string, unknown>[];
    expect(traces).toHaveLength(2);
    expect(traces[1]).toMatchObject({ type: "box", q1: [0.04], median: [0.05], upperfence: [0.08] });
    expect((traces[0]!.marker as { color: string }).color).not.toBe((traces[1]!.marker as { color: string }).color);
  });

  it("words progress and patterns", () => {
    expect(progressText(null)).toMatch(/Placing/);
    expect(progressText({ stage: "windows", done: 1000, total: 2000, elapsed_s: 3 })).toBe("1,000 of 2,000 cells smoothed");
    expect(progressText({ stage: "clusters", done: 1, total: 1, elapsed_s: 1 })).toMatch(/Clustering/);
    expect(patternText(["chr7"], ["chr10", "chr13"])).toBe("chr7 gain; chr10, chr13 loss");
    expect(patternText([], [])).toBe("");
  });

  it("suggests a cell-type column and its normal groups", () => {
    const cols = [
      { name: "sample", dtype: "category", n_unique: 4 },
      { name: "cell_type", dtype: "category", n_unique: 6 },
      { name: "barcode", dtype: "object", n_unique: 100 },
      { name: "cnv_leiden", dtype: "category", n_unique: 5 },
      { name: "n_genes", dtype: "int64", n_unique: 90 },
    ];
    const ok = referenceColumns(cols, 100);
    expect(ok.map((c) => c.name)).toEqual(["sample", "cell_type"]);
    expect(suggestReferenceColumn(ok)).toBe("cell_type");
    expect(suggestReferenceGroups(["Epithelial cell", "T cell CD4", "B cell", "Macrophage", "Fibroblast", "Tumour"])).toEqual([
      "T cell CD4",
      "B cell",
      "Macrophage",
      "Fibroblast",
    ]);
  });
});
