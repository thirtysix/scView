import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { EnrichmentTermsTable } from "./EnrichmentTermsTable";

const row = {
  term: "HALLMARK_INTERFERON_ALPHA_RESPONSE",
  pvalue: 1e-12,
  adjusted_pvalue: 5e-11,
  overlap_count: 23,
  gene_count: 97,
  genes: ["ISG15", "IFI6"],
  collection: "msigdb:h",
};

describe("EnrichmentTermsTable", () => {
  it("explains each column in its header tooltip", () => {
    render(
      <EnrichmentTermsTable
        results={[row]}
        sortField="adj_pval"
        sortDir="asc"
        onSort={vi.fn()}
        onTermClick={vi.fn()}
        onAsk={vi.fn()}
        scoringTerm={null}
      />,
    );
    expect(screen.getByRole("columnheader", { name: /Adj\.P/ })).toHaveAttribute(
      "title",
      "Hypergeometric p of the overlap between the query genes and the set, with the tested genes as background; Benjamini-Hochberg adjusted over the sets tested",
    );
    expect(screen.getByRole("columnheader", { name: /Overlap/ })).toHaveAttribute("title", "Query genes in the set / genes in the set");
    expect(screen.getByRole("columnheader", { name: /Term/ }).getAttribute("title")).toMatch(/^Gene set \(term\)/);
    expect(screen.getByText("23/97")).toBeInTheDocument();
  });
});
