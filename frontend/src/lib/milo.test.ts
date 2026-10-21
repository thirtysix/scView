import { describe, it, expect } from "vitest";
import { MILO } from "@/test/miloFixture";
import {
  beeswarmOffsets,
  describeMilo,
  edgeSegments,
  labelOrder,
  logFcRange,
  miloCsv,
  miloQuestion,
  nhoodRows,
  nodeSize,
  NO_LABEL,
  pickLayout,
  shifted,
} from "./milo";

describe("milo helpers", () => {
  it("zips the columns into rows and marks the significant ones", () => {
    const rows = nhoodRows(MILO);
    expect(rows).toHaveLength(4);
    expect(rows[0]).toMatchObject({ id: 0, label: "CD14 Mono", logFC: 2.1, significant: true, counts: [10, 30] });
    expect(rows[2]!.significant).toBe(false);
    expect(rows[3]).toMatchObject({ logFC: null, spatialFdr: null, significant: false });
    expect(shifted(rows, "up").map((r) => r.id)).toEqual([0]);
    expect(shifted(rows, "down").map((r) => r.id)).toEqual([1]);
    const unlabelled = nhoodRows({ ...MILO, label_key: null, nhoods: { ...MILO.nhoods, label: null, label_fraction: null } });
    expect(unlabelled.every((r) => r.label === NO_LABEL)).toBe(true);
  });

  it("orders labels by median logFC, Mixed last", () => {
    expect(labelOrder(nhoodRows(MILO))).toEqual(["B", "CD14 Mono", "Mixed"]);
  });

  it("spreads overlapping points and keeps separate ones on the centre line", () => {
    expect(beeswarmOffsets([0, 5, 10], 1, 0.1)).toEqual([0, 0, 0]);
    const off = beeswarmOffsets([0, 0.1, 0.2, 0.3], 1, 0.1);
    expect(new Set(off).size).toBe(4);
    expect(off[0]).toBe(0);
    // A crowded row stays within its band.
    const many = beeswarmOffsets(Array.from({ length: 200 }, (_, i) => i * 0.001), 1, 0.1, 0.3);
    expect(Math.max(...many.map(Math.abs))).toBeLessThanOrEqual(0.3 + 1e-9);
  });

  it("lays out the graph on the scatter's embedding when the result has it", () => {
    expect(pickLayout(MILO, "X_pca")).toBe("X_pca");
    expect(pickLayout(MILO, "X_tsne")).toBe("X_umap");
    expect(pickLayout({ ...MILO, layouts: { X_pca: [] } }, null)).toBe("X_pca");
    // Edges to a missing or unplaced node are skipped.
    const seg = edgeSegments(MILO, "X_umap");
    expect(seg.x).toEqual([5, 6, null, 6, 8, null]);
    expect(edgeSegments(MILO, "X_pca").x).toHaveLength(6);
  });

  it("sizes nodes by area and colours symmetric around 0", () => {
    expect(nodeSize(80, 80)).toBe(18);
    expect(nodeSize(20, 80)).toBe(11);
    expect(logFcRange(nhoodRows(MILO))).toBe(2.5);
    expect(logFcRange([])).toBe(1);
  });

  it("exports one CSV row per neighbourhood", () => {
    const { headers, rows } = miloCsv(MILO);
    expect(headers).toEqual([
      "nhood", "index_cell", "n_cells", "cluster", "label_fraction", "log2FC_STIM_vs_CTRL",
      "p_value", "spatial_fdr", "cells_CTRL", "cells_STIM",
    ]);
    expect(rows[0]).toEqual([0, "c1", 40, "CD14 Mono", 0.9, 2.1, 0.001, 0.01, 10, 30]);
    expect(rows[3]![5]).toBeNull();
  });

  it("describes the run and asks the co-pilot with its own numbers", () => {
    expect(describeMilo(MILO)).toContain("~ Donor + stim, paired by Donor (CTRL 8, STIM 8 samples)");
    const q = miloQuestion(MILO);
    expect(q).toContain("STIM vs CTRL with Donor as replicates: 2 of 4 neighbourhoods");
    expect(q).toContain("CD14 Mono: 1 up, 1 down of 2");
    expect(q).not.toContain("B: 0 up");
  });
});
