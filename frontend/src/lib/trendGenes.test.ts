import { describe, it, expect } from "vitest";
import type { TrendGene, TrendGenesResult } from "@/api/trajectory";
import { heatmapOrder, scaleRow, tickBins, trendGenesCsv } from "./trendGenes";

const gene = (name: string, peak: number | null, f: number | null, pattern: TrendGene["pattern"] = "rising"): TrendGene => ({
  gene: name,
  f_stat: f,
  pval: 1e-10,
  pval_adj: 1e-8,
  r2: 0.5,
  amplitude: 2,
  peak_pseudotime: peak,
  pattern,
  n_detected: 100,
  mean: [0, 1, 2],
  fitted: [0, 1, 2],
});

describe("pseudotime gene heatmap", () => {
  it("orders rows by where the trend peaks, then by the strongest change", () => {
    const genes = [gene("Late", 0.9, 50), gene("Early", 0.1, 10), gene("Mid", 0.5, 5), gene("Late2", 0.9, 80), gene("None", null, 1)];
    expect(heatmapOrder(genes).map((g) => g.gene)).toEqual(["Early", "Mid", "Late2", "Late", "None"]);
    expect(genes[0]!.gene).toBe("Late"); // not sorted in place
  });

  it("scales each row to 0..1, a flat row to 0", () => {
    expect(scaleRow([1, 3, 2])).toEqual([0, 1, 0.5]);
    expect(scaleRow([2, 2])).toEqual([0, 0]);
  });

  it("puts ticks on the first and last bin", () => {
    expect(tickBins(40)).toEqual([0, 8, 16, 23, 31, 39]);
    expect(tickBins(4)).toEqual([0, 1, 2, 3]);
    expect(tickBins(0)).toEqual([]);
  });

  it("exports the ranked list as a table", () => {
    const result = { genes: [gene("Hba-a2", 0.3, 380.7), gene("Apoe", 0, 120, "falling")] } as TrendGenesResult;
    const { headers, rows } = trendGenesCsv(result);
    expect(headers.slice(0, 3)).toEqual(["rank", "gene", "pattern"]);
    expect(rows[1]!.slice(0, 3)).toEqual([2, "Apoe", "falling"]);
  });
});
