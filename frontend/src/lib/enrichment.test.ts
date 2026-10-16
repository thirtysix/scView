import { describe, it, expect } from "vitest";
import type { EnrichmentResult } from "@/api/enrichment";
import { nextSort, sortEnrichmentResults } from "./enrichment";

const term = (name: string, adj: number, overlap: number): EnrichmentResult => ({
  term: name,
  pvalue: adj / 10,
  adjusted_pvalue: adj,
  overlap_count: overlap,
  gene_count: 50,
  genes: [],
  collection: "h.all",
});

const results = [term("B_TERM", 0.01, 3), term("A_TERM", 0.2, 9), term("C_TERM", 0.001, 5)];

describe("sortEnrichmentResults", () => {
  it("sorts by adjusted p, overlap or name, either way, without touching the input", () => {
    const names = (rs: EnrichmentResult[]) => rs.map((r) => r.term);
    expect(names(sortEnrichmentResults(results, "adj_pval", "asc"))).toEqual(["C_TERM", "B_TERM", "A_TERM"]);
    expect(names(sortEnrichmentResults(results, "overlap", "desc"))).toEqual(["A_TERM", "C_TERM", "B_TERM"]);
    expect(names(sortEnrichmentResults(results, "term", "asc"))).toEqual(["A_TERM", "B_TERM", "C_TERM"]);
    expect(names(sortEnrichmentResults(results, "term", "desc"))).toEqual(["C_TERM", "B_TERM", "A_TERM"]);
    expect(names(results)).toEqual(["B_TERM", "A_TERM", "C_TERM"]);
  });
});

describe("nextSort", () => {
  it("flips the current column and starts a new one at its natural order", () => {
    expect(nextSort({ field: "adj_pval", dir: "asc" }, "adj_pval")).toEqual({ field: "adj_pval", dir: "desc" });
    expect(nextSort({ field: "adj_pval", dir: "asc" }, "overlap")).toEqual({ field: "overlap", dir: "desc" });
    expect(nextSort({ field: "overlap", dir: "desc" }, "adj_pval")).toEqual({ field: "adj_pval", dir: "asc" });
  });
});
