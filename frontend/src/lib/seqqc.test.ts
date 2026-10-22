import { describe, it, expect } from "vitest";
import type { KneeSummary, SeqFlag } from "@/api/seqqc";
import { flagFor, flagTip, formatMetric, kneeShapes, rankTraces, sampleColumns } from "./seqqc";

const flag = (over: Partial<SeqFlag>): SeqFlag => ({
  metric: "frac_reads_in_cells",
  label: "Reads in cells",
  level: "warn",
  title: "Low fraction reads in cells",
  value: 0.62,
  threshold: 0.7,
  warn: 0.7,
  error: 0.5,
  direction: "below",
  ideal: "Ideal > 70%",
  message: "",
  source: "Cell Ranger 10.1 web summary alert",
  url: "",
  ...over,
});

const KNEE = {
  knee: 5199.8,
  inflection: 376,
  curve: { rank: [1, 5, 40, 41, 3000], umi: [9000, 7000, 300, 280, 1], kept: [1, 1, 0, 1, 0] },
} as unknown as KneeSummary;

describe("seqqc helpers", () => {
  it("formats counts and fractions", () => {
    expect(formatMetric("mean_reads_per_cell", 54502)).toBe("54,502");
    expect(formatMetric("sequencing_saturation", 0.708)).toBe("70.8%");
    expect(formatMetric("antisense", 0.01)).toBe("1.0%");
    expect(formatMetric("estimated_cells", undefined)).toBe("—");
  });

  it("picks the worst flag of a metric and explains both lines", () => {
    const flags = [flag({ level: "warn" }), flag({ level: "error", threshold: 0.5 }), flag({ metric: "q30_umi" })];
    expect(flagFor(flags, "frac_reads_in_cells")?.level).toBe("error");
    expect(flagFor(flags, "valid_barcodes")).toBeUndefined();
    expect(flagTip(flags[0]!)).toBe(
      "Low fraction reads in cells: 62.0% (warning below 70.0%, error below 50.0%; Ideal > 70%). Source: Cell Ranger 10.1 web summary alert.",
    );
    const depth = flag({ metric: "mean_reads_per_cell", value: 14468, warn: 20000, error: null, level: "info", ideal: "Minimum 20,000 read pairs per cell", source: "10x sequencing requirements" });
    expect(flagTip(depth)).toBe(
      "Low fraction reads in cells: 14,468 (flagged below 20,000; Minimum 20,000 read pairs per cell). Source: 10x sequencing requirements.",
    );
  });

  it("splits the curve into cells and other barcodes", () => {
    const [other, cells] = rankTraces(KNEE);
    expect(other!.name).toBe("Other barcodes");
    expect(other!.x).toEqual([40, 3000]);
    expect(cells!.x).toEqual([1, 5, 41]);
    expect(cells!.y).toEqual([9000, 7000, 280]);
  });

  it("draws the knee and inflection at their UMI totals (annotations in log10 on a log axis)", () => {
    const { shapes, annotations } = kneeShapes(KNEE);
    expect(shapes.map((s) => (s as { y0: number }).y0)).toEqual([5199.8, 376]);
    const ys = annotations.map((a) => (a as { y: number }).y);
    expect(ys[0]).toBeCloseTo(Math.log10(5199.8));
    expect(annotations.map((a) => (a as { text: string }).text)).toEqual(["knee 5,200 UMIs", "inflection 376 UMIs"]);
    expect(kneeShapes({ ...KNEE, knee: null, inflection: null }).shapes).toEqual([]);
  });

  it("offers categorical columns with 2-50 values as sample columns", () => {
    const cols = [
      { name: "sample", dtype: "category", n_unique: 4, values: ["a", "b", "c", "d"] },
      { name: "barcode", dtype: "object", n_unique: 5000, values: [] },
      { name: "n_genes", dtype: "int64", n_unique: 40 },
      { name: "one", dtype: "category", n_unique: 1, values: ["x"] },
    ];
    expect(sampleColumns(cols).map((c) => c.name)).toEqual(["sample"]);
  });
});
