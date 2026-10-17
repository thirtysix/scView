import { describe, it, expect } from "vitest";
import type { GroupDistances } from "@/api/clustering";
import { nearestGroups, nonlinearEmbedding, orderedHeatmap, spaceLabel } from "./clusterDistances";

const D: GroupDistances = {
  column: "cluster",
  groups: ["B", "CD14 Mono", "NK", "CD16 Mono"],
  sizes: [100, 400, 60, 90],
  distances: [
    [0, 13.9, 15.7, 12.7],
    [13.9, 0, 17.6, 8.6],
    [15.7, 17.6, 0, 16.5],
    [12.7, 8.6, 16.5, 0],
  ],
  order: [2, 0, 1, 3],
  representation: "X_pca_harmony",
  n_pcs: 50,
  rep_source: "graph",
  linkage: "complete",
  metric: "euclidean",
  n_left_out: 0,
};

describe("nonlinearEmbedding", () => {
  it("names UMAP and t-SNE embeddings, and nothing for PCA or others", () => {
    expect(nonlinearEmbedding("X_umap")).toBe("UMAP");
    expect(nonlinearEmbedding("X_umap_uncorrected")).toBe("UMAP");
    expect(nonlinearEmbedding("X_umap_3d")).toBe("UMAP");
    expect(nonlinearEmbedding("X_tsne")).toBe("t-SNE");
    expect(nonlinearEmbedding("X_t-SNE")).toBe("t-SNE");
    expect(nonlinearEmbedding("X_pca")).toBeNull();
    expect(nonlinearEmbedding("X_draw_graph_fa")).toBeNull();
    expect(nonlinearEmbedding(null)).toBeNull();
  });
});

describe("orderedHeatmap", () => {
  it("puts rows and columns in the clustering's order, with a blank diagonal", () => {
    const h = orderedHeatmap(D);
    expect(h.labels).toEqual(["NK", "B", "CD14 Mono", "CD16 Mono"]);
    expect(h.sizes).toEqual([60, 100, 400, 90]);
    expect(h.z[0]).toEqual([null, 15.7, 17.6, 16.5]);
    expect(h.z[2]).toEqual([17.6, 13.9, null, 8.6]);
    expect(h.z.map((row, i) => row[i])).toEqual([null, null, null, null]);
  });

  it("falls back to the column's order when the order does not fit", () => {
    expect(orderedHeatmap({ ...D, order: [0, 1] }).labels).toEqual(D.groups);
  });
});

describe("nearestGroups", () => {
  it("lists the closest groups first, without the group itself", () => {
    expect(nearestGroups(D, "CD14 Mono")).toEqual([
      { group: "CD16 Mono", distance: 8.6 },
      { group: "B", distance: 13.9 },
      { group: "NK", distance: 17.6 },
    ]);
    expect(nearestGroups(D, "CD14 Mono", 1)).toEqual([{ group: "CD16 Mono", distance: 8.6 }]);
    expect(nearestGroups(D, "Eryth")).toEqual([]);
  });
});

describe("spaceLabel", () => {
  it("says which PCs, and whether they are the neighbour graph's", () => {
    expect(spaceLabel(D)).toBe("50 Harmony-corrected PCs, the space the neighbour graph was built in");
    expect(spaceLabel({ ...D, representation: "X_pca", n_pcs: 30 })).toBe(
      "30 PCs, the space the neighbour graph was built in",
    );
    expect(spaceLabel({ ...D, representation: "X_pca", rep_source: "no_graph" })).toBe(
      "50 PCs (no neighbour graph recorded)",
    );
    expect(spaceLabel({ ...D, rep_source: "missing" })).toMatch(/graph's own space is not in the file/);
  });
});
