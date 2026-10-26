import { describe, expect, it } from "vitest";
import { GENE_CCL2, MILODE } from "@/test/milodeFixture";
import {
  deNhoodRows,
  describeMiloDE,
  geneRange,
  genePoints,
  miloDEGeneCsv,
  miloDEQuestion,
  pickDELayout,
  sortGenes,
} from "./milode";

describe("milode helpers", () => {
  it("turns the neighbourhood columns into rows and flags one-sided ones", () => {
    const rows = deNhoodRows(MILODE);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toMatchObject({ id: 0, label: "CD14 Mono", cells: [200, 200], nDe: 120, oneSided: false });
    expect(rows[1]!.oneSided).toBe(true);
    expect(rows[2]).toMatchObject({ tested: false, oneSided: false });
  });

  it("sorts and filters genes, missing values last", () => {
    expect(sortGenes(MILODE.genes, "n_de", "desc").map((g) => g.gene)).toEqual(["ISG15", "CCL2", "ACTB"]);
    expect(sortGenes(MILODE.genes, "median_logfc", "asc").map((g) => g.gene)).toEqual(["ACTB", "CCL2", "ISG15"]);
    expect(sortGenes(MILODE.genes, "gene", "asc", "c").map((g) => g.gene)).toEqual(["ACTB", "CCL2"]);
    const withNull = [...MILODE.genes, { ...MILODE.genes[2]!, gene: "X", median_logfc: null }];
    expect(sortGenes(withNull, "median_logfc", "desc").at(-1)!.gene).toBe("X");
  });

  it("splits a gene's neighbourhoods into DE and not DE, leaving out untested ones", () => {
    expect(genePoints(MILODE, GENE_CCL2, "X_umap")).toEqual({ sig: [0], grey: [1] });
    expect(geneRange(GENE_CCL2)).toBe(2.5);
    expect(pickDELayout(MILODE, "X_pca")).toBe("X_pca");
    expect(pickDELayout(MILODE, "X_tsne")).toBe("X_umap");
  });

  it("describes the run, exports the genes and asks the co-pilot", () => {
    expect(describeMiloDE(MILODE)).toContain("25-nearest-neighbour graph of X_pca_harmony");
    const { headers, rows } = miloDEGeneCsv(MILODE);
    expect(headers).toContain("median_log2FC_STIM_vs_CTRL");
    expect(rows[0]).toEqual(["ISG15", 2, 2, 1, 2, 0, 4.1, 1e-20]);
    expect(miloDEQuestion(MILODE)).toContain("ISG15 2/2");
  });
});
