import { describe, it, expect, beforeEach } from "vitest";
import type { ReliabilityGroup, ReliabilityResponse, ReliabilitySummary } from "@/api/reliability";
import {
  applyReliability,
  errorText,
  fmtScore,
  fmtShare,
  groupCells,
  reliabilityCaption,
  reliabilityLegendNote,
  reliabilityName,
  sameView,
  sortGroups,
  summaryMatches,
  summaryOf,
  toOverlayValues,
} from "./embeddingReliability";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { colorScaleCaption, fixedColorScale, overlayStats } from "@/lib/colorScale";

const groups: ReliabilityGroup[] = [
  { group: "CD4 T", n: 2600, median: 0, q1: 0, q3: 0.067, mean: 0.03, share_low: 0.98 },
  { group: "B", n: 998, median: 0.067, q1: 0, q3: 0.13, mean: 0.1, share_low: 0.82 },
  { group: "pDC", n: 108, median: 0.4, q1: 0.27, q3: 0.53, mean: 0.44, share_low: 0.1 },
];

const SUMMARY: ReliabilitySummary = {
  embedding: "X_umap",
  dims: [0, 1],
  n_dims: 2,
  k: 15,
  graph: { source: "stored", key: "neighbors", k: 15, use_rep: "X_pca_harmony", n_pcs: null, metric: "euclidean" },
  n_cells: 13836,
  n_scored: 13836,
  sampled: false,
  seed: 0,
  chance: 15 / 13835,
  low_threshold: 0.2,
  low_reason: "below 0.2: fewer than 1 in 5 of the cell's graph neighbours are among its nearest cells on the plot",
  notes: [],
  method: "Per cell: ...",
  seconds: 0.1,
  overall: { n: 13836, median: 0.0667, q1: 0, q3: 0.133, mean: 0.091, share_low: 0.83 },
  shuffled: { median: 0, mean: 0.00107, seed: 1 },
  group_by: "cluster",
  group_reason: "requested",
  groups,
  cached: false,
};

describe("plot reliability helpers", () => {
  beforeEach(() => useUnifiedViewStore.getState().clearOverlayState());

  it("names the overlay after the plot", () => {
    expect(reliabilityName("X_umap")).toBe("Neighbour preservation (UMAP)");
    expect(reliabilityName("X_umap_3d", 3)).toBe("Neighbour preservation (UMAP 3D)");
    expect(reliabilityName("X_pca", 2, [2, 3])).toBe("Neighbour preservation (PCA, PC3 vs PC4)");
  });

  it("prints a chance level without rounding it to zero", () => {
    expect(fmtScore(15 / 13835)).toBe("0.0011");
    expect(fmtScore(0.0667)).toBe("0.067");
    expect(fmtScore(0.4)).toBe("0.40");
    expect(fmtScore(1)).toBe("1.00");
    expect(fmtScore(0)).toBe("0");
    expect(fmtScore(null)).toBe("n/a");
    expect(fmtShare(0.83)).toBe("83%");
    expect(fmtShare(0.004)).toBe("<1%");
    expect(fmtShare(0)).toBe("0%");
  });

  it("turns nulls (cells left out of a sample) into NaN", () => {
    const v = toOverlayValues([0.5, null, 1]);
    expect(v).toBeInstanceOf(Float32Array);
    expect(v[0]).toBe(0.5);
    expect(Number.isNaN(v[1]!)).toBe(true);
  });

  it("sorts groups, keeping the server's worst-first order by default", () => {
    expect(sortGroups(groups, null, false).map((g) => g.group)).toEqual(["CD4 T", "B", "pDC"]);
    expect(sortGroups(groups, "median", true).map((g) => g.group)).toEqual(["pDC", "B", "CD4 T"]);
    expect(sortGroups(groups, "n", false).map((g) => g.group)).toEqual(["pDC", "B", "CD4 T"]);
    expect(sortGroups(groups, "share_low", true).map((g) => g.group)).toEqual(["CD4 T", "B", "pDC"]);
    expect(sortGroups(groups, "group", false).map((g) => g.group)).toEqual(["B", "CD4 T", "pDC"]);
    expect(groups.map((g) => g.group)).toEqual(["CD4 T", "B", "pDC"]); // not sorted in place
  });

  it("finds a group's cells from the label codes", () => {
    expect([...groupCells(Int32Array.from([0, 1, 1, 2, 1]), ["a", "b", "c"], "b")]).toEqual([1, 2, 4]);
    expect(groupCells([0, 1], ["a", "b"], "zzz").size).toBe(0);
  });

  it("captions what the number means, with n, chance and random placement", () => {
    const text = reliabilityCaption(SUMMARY);
    expect(text).toContain("share of its 15 graph neighbours (the kNN graph the clustering used)");
    expect(text).toContain("Median 0.067 over 13,836 cells; 83% are low (below 0.2");
    expect(text).toContain("Chance is 0.0011; random placement would score about 0.0011.");
    expect(text).toContain("don't read its position closely");
    const sampled = reliabilityCaption({ ...SUMMARY, sampled: true, n_scored: 100000, n_cells: 300000 });
    expect(sampled).toContain("100,000 of 300,000 cells (a seeded sample; the rest are grey)");
    const computed = reliabilityCaption({ ...SUMMARY, graph: { ...SUMMARY.graph, source: "computed", use_rep: "X_pca" } });
    expect(computed).toContain("an exact kNN in X_pca (no graph is stored)");
  });

  it("puts the scores on the scatter as a score overlay that knows its plot", () => {
    const res: ReliabilityResponse = { ...SUMMARY, scores: [1, null, 0.25] };
    applyReliability("kang", res);
    const u = useUnifiedViewStore.getState();
    expect(u.scatterOverlay).toEqual({
      type: "score",
      name: "Neighbour preservation (UMAP)",
      reliability: { embedding: "X_umap", k: 15, chance: 15 / 13835, random: 0.00107 },
    });
    expect(u.overlayLabel).toBe("Neighbour preservation (UMAP)");
    expect(u.overlayDatasetId).toBe("kang");
    expect(Array.from(u.overlayValues!).map((v) => (Number.isNaN(v) ? "nan" : v))).toEqual([1, "nan", 0.25]);
    expect(reliabilityLegendNote({ embedding: "X_umap", k: 15, chance: 15 / 13835, random: 0.00107 })).toBe(
      "1 = all 15 graph neighbours kept around the cell. Chance 0.0011; random placement about 0.0011.",
    );
    expect(Object.keys(summaryOf(res))).not.toContain("scores");
    // A PCA's default pair is named, but not saved as a pair.
    applyReliability("kang", { ...res, embedding: "X_pca", dims: [0, 1] });
    expect(useUnifiedViewStore.getState().scatterOverlay).toMatchObject({
      name: "Neighbour preservation (PCA, PC1 vs PC2)",
      reliability: { embedding: "X_pca" },
    });
    const saved = useUnifiedViewStore.getState().scatterOverlay;
    expect(saved?.type === "score" && saved.reliability?.dims).toBeUndefined();
  });

  it("matches a summary and a view to the overlay's plot", () => {
    const ref = { embedding: "X_umap", k: 15, chance: 0.001, random: 0.001 };
    expect(summaryMatches(ref, SUMMARY)).toBe(true);
    expect(summaryMatches({ ...ref, k: 5 }, SUMMARY)).toBe(false);
    expect(summaryMatches({ ...ref, embedding: "X_pca", dims: [2, 3] }, { ...SUMMARY, embedding: "X_pca" })).toBe(false);
    expect(summaryMatches({ ...ref, embedding: "X_pca", dims: [2, 3] }, { ...SUMMARY, embedding: "X_pca", dims: [2, 3] })).toBe(true);
    expect(summaryMatches({ ...ref, embedding: "X_umap_3d" }, { ...SUMMARY, embedding: "X_umap_3d", dims: [0, 1, 2], n_dims: 3 })).toBe(true);
    expect(sameView(ref, "X_umap")).toBe(true);
    expect(sameView(ref, "X_umap", [0, 1])).toBe(true);
    expect(sameView(ref, "X_tsne")).toBe(false);
    expect(sameView({ ...ref, embedding: "X_pca", dims: [2, 3] }, "X_pca")).toBe(false);
  });

  it("reads the server's detail out of an API error", () => {
    expect(errorText(new Error('API error 422: {"detail":"The stored graph keeps 15 neighbours"}'))).toBe(
      "The stored graph keeps 15 neighbours",
    );
    expect(errorText(new Error("API error 502: Backend down"))).toBe("Backend down");
  });
});

describe("fixedColorScale", () => {
  it("keeps 0-1 whatever the values, never clipped", () => {
    const scale = fixedColorScale(overlayStats(Float32Array.from([0.125, 0.375, NaN])), "viridis", 0, 1);
    expect(scale).toMatchObject({ min: 0, max: 1, dataMin: 0.125, dataMax: 0.375, clip: null, diverging: false, centre: null, fixed: true, n: 2 });
    expect(colorScaleCaption(scale)).toBe("Colour range 0 to 1, fixed (not clipped to the data).");
  });
});
