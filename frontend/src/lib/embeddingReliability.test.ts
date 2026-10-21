import { describe, it, expect, beforeEach } from "vitest";
import type { ReliabilityGroup, ReliabilityResponse, ReliabilitySummary } from "@/api/reliability";
import {
  applyReliability,
  errorText,
  exactLine,
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
  { group: "CD4 T", n: 2600, median: 0.667, q1: 0.533, q3: 0.8, mean: 0.65, share_low: 0.22, exact_median: 0 },
  { group: "B", n: 998, median: 0.933, q1: 0.8, q3: 1, mean: 0.88, share_low: 0.05, exact_median: 0.067 },
  { group: "pDC", n: 108, median: 1, q1: 1, q3: 1, mean: 0.98, share_low: 0, exact_median: 0.4 },
];

const SUMMARY: ReliabilitySummary = {
  embedding: "X_umap",
  dims: [0, 1],
  n_dims: 2,
  k: 15,
  window: { share: 0.05, cells: 692, radius: "exact", ref_cells: null },
  graph: { source: "stored", key: "neighbors", k: 15, use_rep: "X_pca_harmony", n_pcs: null, metric: "euclidean" },
  n_cells: 13836,
  n_scored: 13836,
  sampled: false,
  seed: 0,
  chance: 692 / 13835,
  exact_chance: 15 / 13835,
  low_threshold: 0.5,
  low_reason: "below 0.5: fewer than half of the cell's graph neighbours lie near it on the plot",
  notes: [],
  method: "Per cell: ...",
  seconds: 0.1,
  overall: { n: 13836, median: 0.9333, q1: 0.6667, q3: 1, mean: 0.84, share_low: 0.089 },
  exact: { median: 0.0667, q1: 0, q3: 0.133, mean: 0.091, chance: 15 / 13835 },
  shuffled: { median: 0, mean: 0.0501, seed: 1 },
  group_by: "cluster",
  group_reason: "requested",
  groups,
  cached: false,
};

describe("plot reliability helpers", () => {
  beforeEach(() => useUnifiedViewStore.getState().clearOverlayState());

  it("names the overlay after the plot", () => {
    expect(reliabilityName("X_umap")).toBe("Neighbour placement (UMAP)");
    expect(reliabilityName("X_umap_3d", 3)).toBe("Neighbour placement (UMAP 3D)");
    expect(reliabilityName("X_pca", 2, [2, 3])).toBe("Neighbour placement (PCA, PC3 vs PC4)");
  });

  it("prints a chance level without rounding it to zero", () => {
    expect(fmtScore(15 / 13835)).toBe("0.0011");
    expect(fmtScore(692 / 13835)).toBe("0.05");
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
    expect(text).toContain(
      "share of its 15 graph neighbours (the kNN graph the clustering used) that lie among its nearest 692 cells on this plot (5% of the cells)",
    );
    expect(text).toContain("Median 0.93 over 13,836 cells; 9% are low (below 0.5: fewer than half");
    expect(text).toContain("Chance is 0.05; random placement would score about 0.05.");
    expect(text).toContain("A low cell sits away from its real neighbours on the plot");
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
      name: "Neighbour placement (UMAP)",
      reliability: { embedding: "X_umap", k: 15, window: 692, chance: 692 / 13835, random: 0.0501 },
    });
    expect(u.overlayLabel).toBe("Neighbour placement (UMAP)");
    expect(u.overlayDatasetId).toBe("kang");
    expect(Array.from(u.overlayValues!).map((v) => (Number.isNaN(v) ? "nan" : v))).toEqual([1, "nan", 0.25]);
    expect(reliabilityLegendNote({ embedding: "X_umap", k: 15, window: 692, chance: 692 / 13835, random: 0.0501 })).toBe(
      "1 = all 15 graph neighbours among the cell's nearest 692 cells. Chance 0.05; random placement about 0.05.",
    );
    expect(Object.keys(summaryOf(res))).not.toContain("scores");
    // A PCA's default pair is named, but not saved as a pair.
    applyReliability("kang", { ...res, embedding: "X_pca", dims: [0, 1] });
    expect(useUnifiedViewStore.getState().scatterOverlay).toMatchObject({
      name: "Neighbour placement (PCA, PC1 vs PC2)",
      reliability: { embedding: "X_pca" },
    });
    const saved = useUnifiedViewStore.getState().scatterOverlay;
    expect(saved?.type === "score" && saved.reliability?.dims).toBeUndefined();
  });

  it("matches a summary and a view to the overlay's plot", () => {
    const ref = { embedding: "X_umap", k: 15, window: 692, chance: 0.05, random: 0.05 };
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

  it("gives the strict exact-neighbour number as context, with the caveat only when few are kept", () => {
    expect(exactLine(SUMMARY)).toBe(
      "Exact neighbours kept: median 1 of 15 (chance 0.0011). UMAP keeps few exact neighbours anywhere, so read positions within an island loosely.",
    );
    expect(exactLine({ ...SUMMARY, embedding: "X_pca", exact: { ...SUMMARY.exact, median: 0 } })).toContain(
      "median 0 of 15 (chance 0.0011). PCA, PC1 vs PC2 keeps few exact neighbours anywhere",
    );
    expect(exactLine({ ...SUMMARY, exact: { ...SUMMARY.exact, median: 0.6 } })).toBe(
      "Exact neighbours kept: median 9 of 15 (chance 0.0011).",
    );
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
