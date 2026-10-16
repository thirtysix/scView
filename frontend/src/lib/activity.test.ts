import { describe, expect, it } from "vitest";
import type { ActivitySummary } from "@/api/activity";
import { deProfile, filterRows, heatmapData, sourceRows } from "./activity";

function summary(over: Partial<ActivitySummary> = {}): ActivitySummary {
  return {
    network: "collectri",
    label: "CollecTRI",
    kind: "tf",
    citation: "x",
    species: "human",
    matrix_source: "log-normalised .raw",
    n_cells: 10,
    n_genes_measured: 100,
    sources: ["IRF1", "MYC", "STAT1"],
    n_targets: [20, 10, 25],
    groupby: "stim",
    groups: ["CTRL", "STIM"],
    group_sizes: [5, 5],
    // groups x sources
    mean: [
      [0.5, 1.0, 1.0],
      [3.5, 0.8, 9.0],
    ],
    elapsed_seconds: 0.1,
    ...over,
  };
}

describe("sourceRows", () => {
  it("puts the source that best separates the groups first, with where it peaks", () => {
    const rows = sourceRows(summary());
    expect(rows.map((r) => r.source)).toEqual(["STAT1", "IRF1", "MYC"]);
    expect(rows[0]).toMatchObject({ index: 2, n_targets: 25, top_group: "STIM", top_mean: 9 });
    expect(rows[2]).toMatchObject({ source: "MYC", top_group: "CTRL", top_mean: 1 });
  });

  it("sorts by name when there are no groups", () => {
    const rows = sourceRows(summary({ mean: undefined, groups: undefined, sources: ["b", "a"], n_targets: [5, 6] }));
    expect(rows.map((r) => [r.source, r.top_group])).toEqual([["a", null], ["b", null]]);
  });
});

describe("filterRows", () => {
  it("matches names ignoring case", () => {
    const rows = sourceRows(summary());
    expect(filterRows(rows, "st").map((r) => r.source)).toEqual(["STAT1"]);
    expect(filterRows(rows, "  ")).toHaveLength(3);
  });
});

describe("heatmapData", () => {
  it("shows the top sources as rows, strongest on top, groups as columns", () => {
    const h = heatmapData(summary(), 2)!;
    expect(h.x).toEqual(["CTRL", "STIM"]);
    expect(h.y).toEqual(["IRF1", "STAT1"]); // Plotly draws the last row on top
    expect(h.z).toEqual([[0.5, 3.5], [1.0, 9.0]]);
    expect(h.zmax).toBe(9);
    expect(heatmapData(summary({ mean: undefined }), 2)).toBeNull();
  });
});

describe("deProfile", () => {
  it("keeps every gene with a finite statistic, to 4 significant digits", () => {
    const genes = [
      { gene: "A", logfoldchange: 1, pval: 0, pval_adj: 0, stat: 56.19661331176758 },
      { gene: "B", logfoldchange: -1, pval: 0.1, pval_adj: 0.2, stat: null },
      { gene: "C", logfoldchange: 0, pval: 1, pval_adj: 1 },
      { gene: "D", logfoldchange: -2, pval: 0, pval_adj: 0, stat: -12.5 },
      { gene: "E", logfoldchange: 0, pval: 0.9, pval_adj: 1, stat: 0.000123456 },
    ];
    expect(deProfile(genes)).toEqual({ genes: ["A", "D", "E"], stats: [56.2, -12.5, 0.0001235] });
  });

  it("keeps a 14k-gene request well under the backend's 512 KB JSON cap", () => {
    const genes = Array.from({ length: 14053 }, (_, i) => ({
      gene: `GENE${i}`,
      logfoldchange: 0,
      pval: 0.5,
      pval_adj: 0.5,
      stat: (i % 2 ? -1 : 1) * (Math.PI * 10 + i / 7),
    }));
    const body = JSON.stringify(deProfile(genes));
    expect(body.length).toBeLessThan(260 * 1024);
  });
});
