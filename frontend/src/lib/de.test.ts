import { describe, it, expect } from "vitest";
import type { DEGene, DEResponse } from "@/api/de";
import type { ObsColumnInfo } from "@/api/types";
import {
  deEnrichmentQuery,
  deGeneQuestion,
  deResultQuestion,
  describeComparison,
  isSignificant,
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
    expect(suggestReplicateColumn([col("orig_ident", 6)])).toBe("orig_ident");
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

describe("describeComparison with confounding", () => {
  it("passes the batch warning to the co-pilot", () => {
    const r = {
      n_selected: 100, n_rest: 120, label: "STIM", genes: [], method: "wilcoxon" as const,
      confounding: {
        batch: "dataset", condition: "dataset", kind: "confounded" as const,
        message: "The two sides are different 'dataset' groups, i.e. different batches.",
      },
    };
    const text = describeComparison({ a: "STIM", b: "CTRL", betweenGroups: true }, r);
    expect(text).toContain("Caution: The two sides are different 'dataset' groups");
  });
});

describe("deEnrichmentQuery", () => {
  const res: DEResponse = {
    n_selected: 50,
    n_rest: 50,
    label: "STIM",
    genes: [
      gene("ISG15", 4.2, 1e-30),
      gene("IFIT1", 0.8, 1e-20), // significant p, but under the |log2FC| cut-off at 1
      gene("CCL2", -1.5, 0.001),
      gene("FOS", -2.0, 0.2), // big change, not significant
      gene("ACTB", 0, 1e-5),
    ],
  };

  it("splits the significant genes by direction, with every tested gene as background", () => {
    expect(deEnrichmentQuery(res, 0.05, 1)).toEqual({
      up: ["ISG15"],
      down: ["CCL2"],
      background: ["ISG15", "IFIT1", "CCL2", "FOS", "ACTB"],
    });
  });

  it("uses exactly the table's rule, so lowering the cut-off adds genes", () => {
    expect(deEnrichmentQuery(res, 0.05, 0.5).up).toEqual(["ISG15", "IFIT1"]);
    const table = res.genes.filter((g) => isSignificant(g, 0.05, 0.5) && g.logfoldchange !== 0);
    const q = deEnrichmentQuery(res, 0.05, 0.5);
    expect([...q.up, ...q.down].sort()).toEqual(table.map((g) => g.gene).sort());
  });

  it("puts a gene with no change in neither list", () => {
    const q = deEnrichmentQuery(res, 0.05, 0);
    expect(q.up).not.toContain("ACTB");
    expect(q.down).not.toContain("ACTB");
  });
});
