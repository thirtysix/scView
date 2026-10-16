import { describe, expect, it } from "vitest";
import type { MeanVariance } from "@/api/normalization";
import { HVG_COLOR, logSpace, meanVarianceTraces, splitGenes } from "./meanVariance";

function data(over: Partial<MeanVariance> = {}): MeanVariance {
  return {
    n_cells: 100,
    n_cells_used: 100,
    n_genes: 4,
    n_genes_undetected: 0,
    theta: 100,
    clip: 10,
    counts_source: "layers['counts']",
    normalization_method: "pearson_residuals",
    genes: ["A", "B", "C", "D"],
    mean: [0.01, 0.1, 1, 10],
    var_counts: [0.01, 0.12, 2, 30],
    var_log: [0.001, 0.05, 0.3, 0],
    var_residuals: [1, 1.1, 5, 0.9],
    hvg: [false, false, true, false],
    ...over,
  };
}

describe("logSpace", () => {
  it("spans the positive range on a log scale", () => {
    const xs = logSpace([0, 0.01, 100], 5);
    expect(xs).toHaveLength(5);
    expect(xs[0]).toBeCloseTo(0.01);
    expect(xs[2]).toBeCloseTo(1);
    expect(xs[4]).toBeCloseTo(100);
    expect(logSpace([0, -1])).toEqual([]);
  });
});

describe("splitGenes", () => {
  it("separates the variable genes and hides what a log axis cannot show", () => {
    const d = data();
    expect(splitGenes(d.mean, d.var_log, d.hvg)).toEqual({ hvg: [2], other: [0, 1], hidden: 1 });
    expect(splitGenes(d.mean, d.var_residuals, null)).toEqual({ hvg: [], other: [0, 1, 2, 3], hidden: 0 });
  });
});

describe("meanVarianceTraces", () => {
  it("draws counts with Poisson and NB references", () => {
    const traces = meanVarianceTraces(data(), "before", "counts");
    expect(traces.map((t) => t.name)).toEqual([
      "Other genes",
      "Variable genes",
      "Poisson (var = mean)",
      "NB, θ = 100",
    ]);
    const hv = traces[1]!;
    expect(hv.text).toEqual(["C"]);
    expect(hv.y).toEqual([2]);
    expect(hv.marker!.color).toBe(HVG_COLOR);
    const nb = traces[3]!;
    const m = nb.x[nb.x.length - 1]!;
    expect(nb.y[nb.y.length - 1]).toBeCloseTo(m + (m * m) / 100);
  });

  it("draws residual variance against the sampling-noise line at 1", () => {
    const traces = meanVarianceTraces(data({ hvg: null }), "after");
    expect(traces.map((t) => t.name)).toEqual(["Genes", "Sampling noise (1)"]);
    expect(traces[0]!.y).toEqual([1, 1.1, 5, 0.9]);
    expect(new Set(traces[1]!.y)).toEqual(new Set([1]));
  });

  it("shows the shifted log without reference lines", () => {
    const traces = meanVarianceTraces(data(), "before", "log");
    expect(traces.map((t) => t.name)).toEqual(["Other genes", "Variable genes"]);
  });
});
