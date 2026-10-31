import { describe, it, expect } from "vitest";
import type { Coverage, QcValuesGroup } from "@/api/atacFragments";
import { coverageQuery } from "@/api/atacFragments";
import { binned, coverageFigure, histogramFigure, lengthFigure, tssProfileFigure, violinFigure } from "./atacPlots";

const COV: Coverage = {
  region: { chrom: "chr1", start: 1000, end: 2000 },
  bins: 2,
  bin_size: 500,
  x: [1250, 1750],
  group_by: "g",
  tracks: [{ name: "0", n_cells: 10, n_fragments_in_window: 1, total_fragments: 5, values: [1, 2] },
           { name: "1", n_cells: 2000, n_fragments_in_window: 1, total_fragments: 5, values: [3, 4] }],
  ymax: 4,
  normalisation: "x",
  notes: [],
  n_fragments: 2,
  max_window: 2000000,
  peaks: [{ start: 1100, end: 1200 }],
  genes: [{ name: "PLUS", id: "a", start: 1200, end: 1800, strand: "+", type: "protein_coding" },
          { name: "MINUS", id: "b", start: 1300, end: 1500, strand: "-", type: "lncRNA" }],
  gene: null,
  genes_note: null,
  seconds: 0,
};

describe("coverageFigure", () => {
  const { data, layout } = coverageFigure(COV, (g) => (g === "0" ? "rgb(1,2,3)" : "rgb(4,5,6)"));
  const l = layout as Record<string, { domain?: number[]; range?: number[]; visible?: boolean }>;

  it("gives each group its own row on one x and one y scale, in the group's colour", () => {
    expect(data).toHaveLength(2);
    expect((data[0] as { yaxis: string }).yaxis).toBe("y");
    expect((data[1] as { yaxis: string }).yaxis).toBe("y2");
    expect((data[0] as { xaxis: string }).xaxis).toBe("x");
    expect((data[1] as { xaxis: string }).xaxis).toBe("x");
    expect((data[0] as { fillcolor: string }).fillcolor).toBe("rgb(1,2,3)");
    expect(l.yaxis?.range).toEqual(l.yaxis2?.range);
    expect(l.yaxis?.range?.[1]).toBeCloseTo(4.2);
    expect(l.yaxis?.domain?.[1]).toBe(1);
    expect(l.yaxis?.domain?.[0]).toBeGreaterThan(l.yaxis2?.domain?.[1] ?? 1 - 0.01);
    expect(l.xaxis?.range).toEqual([1000, 2000]);
  });

  it("puts peaks and genes in a hidden row beneath, arrows along the strand", () => {
    expect(l.yaxis3?.visible).toBe(false);
    expect(l.yaxis3?.domain?.[0]).toBe(0);
    expect((l.yaxis2?.domain?.[0] ?? 0)).toBeGreaterThanOrEqual(l.yaxis3?.domain?.[1] ?? 1);
    const shapes = layout.shapes ?? [];
    expect(shapes.filter((s) => s.type === "rect")).toHaveLength(1);
    expect(shapes.filter((s) => s.type === "line")).toHaveLength(2);
    const notes = layout.annotations ?? [];
    const arrows = notes.filter((n) => n.showarrow);
    expect(arrows[0]?.x).toBe(1800);
    expect(arrows[0]?.ax).toBe(1200);
    expect(arrows[1]?.x).toBe(1300);
    expect(arrows[1]?.ax).toBe(1500);
    const texts = notes.map((n) => n.text);
    expect(texts).toContain("PLUS");
    expect(texts).toContain("1 (n = 2,000)");
  });

  it("an empty window still has an axis to draw", () => {
    const empty = coverageFigure({ ...COV, ymax: 0, tracks: [{ ...COV.tracks[0]!, values: [0, 0] }], peaks: [], genes: [] }, () => "red");
    expect((empty.layout as Record<string, { range?: number[] }>).yaxis?.range).toEqual([0, 1]);
  });
});

describe("QC figures", () => {
  it("skips missing values and marks the cut-off", () => {
    const f = histogramFigure("TSS", [1, null, 3], { low: 3 });
    const y = (f.data[0] as { type: string; y: number[] }).y;
    expect((f.data[0] as { type: string }).type).toBe("bar");
    expect(y.reduce((a, b) => a + b, 0)).toBe(2);
    expect(f.layout.shapes).toHaveLength(1);
  });

  it("bins up to the 99.5th percentile, the odd cell in the last bin", () => {
    const v = [...Array.from({ length: 999 }, (_, i) => i / 100), 500];
    const b = binned(v, 10);
    expect(b.counts.reduce((x, y) => x + y, 0)).toBe(1000);
    expect(b.centers[9]! < 11).toBe(true);
    expect(binned([]).counts).toEqual([]);
    expect(binned([2, 2]).counts[0]).toBe(2);
  });

  it("names every violin group on a category axis", () => {
    const groups: QcValuesGroup[] = [
      { name: "0", n: 2, n_shown: 2, tss_enrichment: [1, 2], nucleosome_signal: [null, 1], n_fragments: [1, 1] },
      { name: "1", n: 2, n_shown: 2, tss_enrichment: [3, 4], nucleosome_signal: [1, 1], n_fragments: [1, 1] },
    ];
    const f = violinFigure("TSS", groups, "nucleosome_signal", () => "blue");
    expect(f.layout.yaxis?.type).toBe("category");
    expect(f.layout.yaxis?.tickvals).toEqual(["0", "1"]);
    expect((f.data[0] as { x: number[] }).x).toEqual([1]);
  });

  it("leaves the long-fragment overflow off the length plot", () => {
    const f = lengthFigure({ counts: [0, 5, 9, 100], max: 3, n: 114 });
    expect((f.data[0] as { y: number[] }).y).toEqual([0, 5, 9]);
    expect(f.layout.xaxis?.range).toEqual([0, 800]);
  });

  it("centres the profile's bins", () => {
    const f = tssProfileFigure({ start: -10, bin: 5, fold: [1, 2, 3, 4], n_tss: 1, chromosomes: 1 });
    expect((f.data[0] as { x: number[] }).x).toEqual([-7.5, -2.5, 2.5, 7.5]);
  });
});

describe("coverageQuery", () => {
  it("repeats groups and encodes a gene", () => {
    const q = coverageQuery({ gene: "A B", groupBy: "c t", groups: ["x", "y"], bins: 50 });
    expect(q).toBe("gene=A+B&group_by=c+t&groups=x&groups=y&bins=50");
  });
});
