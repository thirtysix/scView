import { describe, it, expect } from "vitest";
import type { DEResponse } from "@/api/de";
import type { GseaResponse, GseaTerm } from "@/api/enrichment";
import {
  GSEA_CSV_HEADERS,
  describeGseaRun,
  formatPermP,
  gseaCsvRows,
  gseaGenesFromDE,
  nextGseaSort,
  sortGseaResults,
} from "./gsea";

const term = (name: string, nes: number, fdr: number, lead = 3): GseaTerm => ({
  term: name, es: nes / 3, nes, pvalue: fdr / 2, fdr, set_size: 40,
  leading_edge: ["ISG15", "IFIT3", "MX1", "RSAD2"].slice(0, lead), collection: "h.all",
});

const RUN: GseaResponse = {
  rank_metric: "wilcoxon_score",
  rank_metric_label: "Wilcoxon statistic (z-score)",
  n_genes_ranked: 13890, n_excluded_ribo_mito: 163, n_sets_tested: 50, permutations: 1000,
  min_size: 15, max_size: 500, collections: ["h.all"], exclude_ribo_mito: true,
  up: { n_terms: 2, n_significant: 2, results: [term("IFNG", 2.05, 0), term("IFNA", 2.0, 0.001)] },
  down: { n_terms: 1, n_significant: 1, results: [term("OXPHOS", -1.83, 0.01, 2)] },
  seconds: 1.64, cached: false,
};

describe("gseaGenesFromDE", () => {
  it("sends every tested gene with its statistic, null when there is none", () => {
    const r = {
      n_selected: 1, n_rest: 1, label: "STIM",
      genes: [
        { gene: "ISG15", logfoldchange: 4, pval: 1e-30, pval_adj: 1e-28, score: 25 },
        { gene: "CD14", logfoldchange: -1, pval: 1e-4, pval_adj: 1e-3 },
      ],
    } as DEResponse;
    expect(gseaGenesFromDE(r)).toEqual([
      { gene: "ISG15", logfoldchange: 4, pval: 1e-30, score: 25 },
      { gene: "CD14", logfoldchange: -1, pval: 1e-4, score: null },
    ]);
  });
});

describe("GSEA results order", () => {
  it("sorts NES by size so the negative side reads like the positive one", () => {
    const rows = [term("a", -1.2, 0.2), term("b", -2.1, 0.01), term("c", 1.5, 0.05)];
    expect(sortGseaResults(rows, "nes", "desc").map((r) => r.term)).toEqual(["b", "c", "a"]);
    expect(sortGseaResults(rows, "fdr", "asc").map((r) => r.term)).toEqual(["b", "c", "a"]);
  });

  it("starts FDR and term ascending, NES and leading edge descending", () => {
    expect(nextGseaSort({ field: "fdr", dir: "asc" }, "nes")).toEqual({ field: "nes", dir: "desc" });
    expect(nextGseaSort({ field: "nes", dir: "desc" }, "fdr")).toEqual({ field: "fdr", dir: "asc" });
    expect(nextGseaSort({ field: "fdr", dir: "asc" }, "fdr")).toEqual({ field: "fdr", dir: "desc" });
  });
});

describe("formatPermP", () => {
  it("shows a zero as below the permutation resolution", () => {
    expect(formatPermP(0, 1000)).toBe("< 0.0010");
    expect(formatPermP(0, 100)).toBe("< 0.0100");
    expect(formatPermP(0.0123, 100)).toBe("0.0123");
  });
});

describe("describeGseaRun", () => {
  it("says what the genes were ranked by, how many sets and permutations", () => {
    const text = describeGseaRun(RUN);
    expect(text).toContain("13,890 genes ranked by Wilcoxon statistic (z-score)");
    expect(text).toContain("163 ribosomal / mito left out");
    expect(text).toContain("50 gene sets with 15-500 genes in the ranking, 1,000 permutations (1.6 s)");
    expect(describeGseaRun({ ...RUN, cached: true, seconds: 0 })).toContain("(from cache)");
  });
});

describe("gseaCsvRows", () => {
  it("writes both directions with the leading edge", () => {
    const rows = gseaCsvRows(RUN, { up: "Higher in STIM", down: "Higher in CTRL" });
    expect(rows).toHaveLength(3);
    const first = Object.fromEntries(GSEA_CSV_HEADERS.map((h, i) => [h, rows[0]![i]]));
    expect(first.direction).toBe("Higher in STIM");
    expect(first.leading_edge).toBe("ISG15;IFIT3;MX1");
    expect(first.leading_edge_size).toBe(3);
    expect(rows[2]![0]).toBe("Higher in CTRL");
  });
});
