import { describe, expect, it } from "vitest";
import type { DEMethodsResponse, DEResponse } from "@/api/de";
import { deGeneQuestion, describeComparison, isSignificant } from "@/lib/de";
import { agreementCsv, agreementRows, hasPValues, listedGenes, methodLabel } from "@/lib/deMethods";

const base = { n_selected: 10, n_rest: 20, label: "A" };

const logreg: DEResponse = {
  ...base,
  method: "logreg",
  genes: [
    { gene: "UP1", logfoldchange: 2, pval: null, pval_adj: null, stat: 0.3 },
    { gene: "UP2", logfoldchange: 1.5, pval: null, pval_adj: null, stat: 0.9 },
    { gene: "MIXED", logfoldchange: 2, pval: null, pval_adj: null, stat: -0.5 }, // signs disagree
    { gene: "SMALL", logfoldchange: 0.2, pval: null, pval_adj: null, stat: 1.2 }, // under |log2FC| 1
    { gene: "DOWN", logfoldchange: -3, pval: null, pval_adj: null, stat: -0.4 },
  ],
};

describe("cell-level DE methods", () => {
  it("say which results have p-values", () => {
    expect(hasPValues({ method: "wilcoxon" })).toBe(true);
    expect(hasPValues({ method: "t-test" })).toBe(true);
    expect(hasPValues({ method: "pseudobulk" })).toBe(true);
    expect(hasPValues({ method: undefined })).toBe(true); // an older server: Wilcoxon
    expect(hasPValues({ method: "logreg" })).toBe(false);
    expect(methodLabel("t-test")).toBe("Welch t-test");
  });

  it("list a logistic regression by coefficient, never as significant", () => {
    expect(logreg.genes.some((g) => isSignificant(g, 0.05, 0))).toBe(false);
    expect(listedGenes(logreg, 0.05, 1, isSignificant).map((g) => g.gene)).toEqual(["UP2", "DOWN", "UP1"]);
  });

  it("list a p-value result by significance, as before", () => {
    const r: DEResponse = {
      ...base,
      method: "t-test",
      genes: [
        { gene: "B", logfoldchange: 2, pval: 1e-4, pval_adj: 1e-3 },
        { gene: "A", logfoldchange: 3, pval: 1e-9, pval_adj: 1e-8 },
        { gene: "NS", logfoldchange: 3, pval: 0.5, pval_adj: 0.9 },
      ],
    };
    expect(listedGenes(r, 0.05, 1, isSignificant).map((g) => g.gene)).toEqual(["A", "B"]);
  });

  it("describe the test to the co-pilot without inventing a p-value", () => {
    const c = { a: "A", b: "rest", betweenGroups: false };
    expect(describeComparison(c, logreg)).toContain("cell-level logistic regression");
    expect(describeComparison(c, logreg)).toContain("no p-values");
    const q = deGeneQuestion(c, logreg, logreg.genes[1]!);
    expect(q).toContain("coefficient 0.900 (no p-value)");
    expect(q).not.toContain("adjusted p");
    expect(describeComparison(c, { ...logreg, method: "t-test" })).toContain("cell-level Welch t-test");
  });
});

const METHODS: DEMethodsResponse = {
  ...base,
  top_n: 50,
  methods: [
    { method: "wilcoxon", label: "Wilcoxon rank-sum", stat_label: "z-score", has_pvalues: true, n_significant: 3242, seconds: 11.7 },
    { method: "t-test", label: "Welch t-test", stat_label: "t statistic", has_pvalues: true, n_significant: 6938, seconds: 0.5 },
    { method: "logreg", label: "Logistic regression", stat_label: "coefficient", has_pvalues: false, n_significant: null, seconds: 1 },
  ],
  pairs: [
    { a: "wilcoxon", b: "t-test", jaccard_up: 0.9608, shared_up: 49, jaccard_down: 0.6667, shared_down: 40, spearman: 0.9765 },
    { a: "wilcoxon", b: "logreg", jaccard_up: 0.1765, shared_up: 15, jaccard_down: null, shared_down: 0, spearman: null },
  ],
  unique: [{ gene: "PPBP", method: "logreg", side: "down", ranks: { wilcoxon: 2464, "t-test": 2702, logreg: 1 } }],
  unavailable: { MAST: "R only" },
  min_pct: 0.01,
  n_genes_tested: 8086,
  n_genes_skipped: 5967,
};

describe("method agreement", () => {
  it("labels each pair and its overlaps", () => {
    const rows = agreementRows(METHODS);
    expect(rows[0]).toMatchObject({ pair: "Wilcoxon rank-sum vs Welch t-test", up: "0.96 (49)", down: "0.67 (40)", spearman: "0.98" });
    expect(rows[1]).toMatchObject({ down: "– (0)", spearman: "–" });
  });

  it("exports pairs and single-test genes with every test's rank", () => {
    const { headers, rows } = agreementCsv(METHODS);
    expect(headers).toEqual([
      "row", "method_or_pair", "gene", "jaccard_up_top50", "jaccard_down_top50", "spearman",
      "rank_wilcoxon", "rank_t-test", "rank_logreg",
    ]);
    expect(rows[0]).toEqual(["pair", "wilcoxon vs t-test", "", 0.9608, 0.6667, 0.9765, null, null, null]);
    expect(rows[2]).toEqual(["only_in_top", "logreg", "PPBP (down)", null, null, null, 2464, 2702, 1]);
  });
});

describe("methodLabel names the pseudobulk model", () => {
  it("defaults to DESeq2 and adds the model and RUV k when given", () => {
    expect(methodLabel("pseudobulk")).toBe("pseudobulk DESeq2");
    expect(methodLabel("pseudobulk", { pb_method: "voom", ruv_k: 1 })).toBe("pseudobulk limma-voom + RUV k=1");
    expect(methodLabel("pseudobulk", { pb_method: "voombygroup", ruv_k: 0 })).toBe("pseudobulk voomByGroup");
  });
});
