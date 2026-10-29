import { describe, it, expect } from "vitest";
import { extremes, planText, sampleCells, stageText, thumbOf } from "./wnn";
import { prettyEmbeddingLabel } from "./formatting";
import { stepLabel } from "./stepLabel";

describe("WNN card text", () => {
  it("says what a run uses", () => {
    expect(
      planText({ n_proteins: 14, controls: ["IgG1", "IgG2a", "IgG2b"], d_protein: 13, d_rna: 50, rep: "X_pca",
                 batch_key: null, estimate_seconds: 78 }),
    ).toBe("14 proteins (3 isotype controls left out), 13 protein PCs, 50 RNA PCs (PCA); about 1 min 20 s.");
    expect(
      planText({ n_proteins: 6, controls: [], d_protein: 5, d_rna: 30, rep: "X_pca_harmony", batch_key: "donor" }),
    ).toBe("6 proteins, 5 protein PCs, 30 RNA PCs (Harmony PCA), protein PCs Harmony-corrected on donor.");
  });

  it("names the stage", () => {
    expect(stageText(null)).toBe("Starting…");
    expect(stageText({ stage: "knn_protein", label: "Protein neighbours", index: 3, total: 9, elapsed_s: 44 }))
      .toBe("Step 3 of 9: Protein neighbours (45 s so far)");
    expect(stageText({ stage: "protein_pca", label: "Protein PCA", index: 1, total: 9, elapsed_s: 0 }))
      .toBe("Step 1 of 9: Protein PCA");
  });

  it("samples the same cells for every layout", () => {
    expect(sampleCells(3, 5)).toEqual([0, 1, 2]);
    const s = sampleCells(100, 10);
    expect(s).toHaveLength(10);
    expect(s[0]).toBe(0);
    expect(s[9]).toBe(90);
    const t = thumbOf({ x: Float32Array.from([1, 2, 3]), y: Float32Array.from([4, 5, 6]) }, [0, 2]);
    expect(t).toEqual({ cells: [0, 2], x: [1, 3], y: [4, 6] });
    expect(thumbOf({ x: [1] }, [0])).toBeNull();
  });

  it("finds the groups at both ends, leaving out tiny ones", () => {
    const rows = [
      { group: "pDC", mean_rna_weight: 0.75, n_cells: 30 },
      { group: "CD4 T", mean_rna_weight: 0.25, n_cells: 1300 },
      { group: "tiny", mean_rna_weight: 0.99, n_cells: 3 },
    ];
    expect(extremes(rows)).toEqual({ rna: "pDC", protein: "CD4 T" });
    expect(extremes(rows.slice(0, 1))).toBeNull();
  });

  it("labels the step and the protein layout", () => {
    expect(stepLabel("wnn")).toBe("Joint RNA + protein (WNN)");
    expect(prettyEmbeddingLabel("X_umap_protein")).toBe("UMAP (protein)");
    expect(prettyEmbeddingLabel("X_umap_wnn")).toBe("UMAP (WNN)");
  });
});
