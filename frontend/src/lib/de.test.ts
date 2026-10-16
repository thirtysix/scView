import { describe, it, expect } from "vitest";
import type { DEGene, DEResponse } from "@/api/de";
import type { ObsColumnInfo } from "@/api/types";
import {
  deGeneQuestion,
  deResultQuestion,
  describeComparison,
  replicateColumns,
  suggestReplicateColumn,
  type DEComparison,
} from "./de";

const col = (name: string, n_unique: number, dtype = "category"): ObsColumnInfo => ({
  name,
  dtype,
  n_unique,
});

describe("replicateColumns", () => {
  it("keeps low-cardinality non-float columns that are not already in use", () => {
    const cols = [
      col("stim", 2),
      col("ind", 8, "int64"),
      col("cell_type", 13),
      col("total_counts", 5000, "float32"),
      col("barcode", 13836, "object"),
      col("constant", 1),
    ];
    expect(replicateColumns(cols, ["stim", "cell_type"]).map((c) => c.name)).toEqual(["ind"]);
  });
});

describe("suggestReplicateColumn", () => {
  it("prefers a donor column over a sample column", () => {
    expect(suggestReplicateColumn([col("sample_id", 16), col("donor_id", 8)])).toBe("donor_id");
  });

  it("matches ind exactly, not inside other names", () => {
    expect(suggestReplicateColumn([col("kind", 3), col("ind", 8)])).toBe("ind");
    expect(suggestReplicateColumn([col("kind", 3), col("index_group", 4)])).toBe("");
  });

  it("falls back to Seurat's orig.ident", () => {
    expect(suggestReplicateColumn([col("seurat_clusters", 10), col("orig.ident", 6)])).toBe(
      "orig.ident",
    );
  });
});

const gene = (name: string, lfc: number, padj: number): DEGene => ({
  gene: name,
  logfoldchange: lfc,
  pval: padj / 10,
  pval_adj: padj,
});

const genes = [gene("ISG15", 4.2, 1e-30), gene("IFIT1", 3.9, 1e-20), gene("CCL2", -1.5, 0.001)];

const pseudobulkRes: DEResponse = {
  n_selected: 2100,
  n_rest: 1900,
  label: "STIM (CD14 Mono)",
  genes,
  replicate_key: "ind",
  method: "pseudobulk",
  n_samples_a: 8,
  n_samples_b: 8,
  design: "~ind + condition",
};

const groups: DEComparison = {
  a: "STIM",
  b: "CTRL",
  scope: "cell_type = CD14 Mono",
  betweenGroups: true,
};

describe("DE co-pilot questions", () => {
  it("describes a pseudobulk comparison with donors and pairing", () => {
    expect(describeComparison(groups, pseudobulkRes)).toBe(
      "STIM vs CTRL within cell_type = CD14 Mono, pseudobulk DESeq2 over 8 vs 8 donors " +
        "(column ind, paired by donor; 2,100 vs 1,900 cells)",
    );
  });

  it("flags cell-level between-groups results as optimistic, but not selection vs rest", () => {
    const wilcoxon: DEResponse = { n_selected: 40, n_rest: 60, label: "A", genes, underreplicated: true };
    expect(describeComparison({ a: "A", b: "B", betweenGroups: true }, wilcoxon)).toContain(
      "under-replicated, so p-values are optimistic",
    );
    expect(describeComparison({ a: "B cells", b: "rest", betweenGroups: false }, wilcoxon)).toBe(
      "B cells vs rest, cell-level Wilcoxon (40 vs 60 cells)",
    );
  });

  it("says which side a gene is higher in", () => {
    const q = deGeneQuestion(groups, pseudobulkRes, genes[2]!);
    expect(q).toContain("CCL2 has log2 fold change -1.50 (higher in CTRL), adjusted p 0.0010");
    expect(q).toContain("What is CCL2");
  });

  it("lists the top genes on each side of a result", () => {
    const q = deResultQuestion(groups, pseudobulkRes, genes, 1);
    expect(q).toContain("3 of 3 genes pass adjusted p <= 0.05 and |log2FC| >= 1");
    expect(q).toContain("higher in STIM: ISG15, IFIT1.");
    expect(q).toContain("higher in CTRL: CCL2.");
    expect(deResultQuestion(groups, pseudobulkRes, genes.slice(0, 2), 1)).toContain(
      "higher in CTRL: none.",
    );
  });
});
