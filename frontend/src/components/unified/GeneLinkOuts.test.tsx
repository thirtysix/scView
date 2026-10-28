import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import type { GeneReport, ReportSection } from "@/api/report";
import { GeneSectionBody } from "@/components/report/GeneSections";
import { GeneLinkOuts } from "./GeneLinkOuts";

describe("GeneLinkOuts", () => {
  it("links the gene's pages; Open Targets by Ensembl id when known, else its search", () => {
    const { rerender } = render(<GeneLinkOuts gene="IL7R" ensembl="ENSG00000168685" />);
    expect(screen.getByRole("link", { name: /Open Targets/ })).toHaveAttribute(
      "href", "https://platform.opentargets.org/target/ENSG00000168685");
    rerender(<GeneLinkOuts gene="IL7R" />);
    expect(screen.getByRole("link", { name: /Open Targets/ })).toHaveAttribute(
      "href", "https://platform.opentargets.org/search?q=IL7R");
    for (const l of screen.getAllByRole("link")) {
      expect(l).toHaveAttribute("target", "_blank");
      expect(l).toHaveAttribute("rel", "noopener noreferrer");
    }
  });

  it("sits under the gene report's marker databases", () => {
    const section = { id: "databases", status: "ok", data: { types: [], n_types: 0 } } as unknown as ReportSection;
    const report = { kind: "gene", gene: "CD14", groupby: "cluster" } as unknown as GeneReport;
    render(<>{GeneSectionBody({ section, report, onGene: () => undefined, onGroup: () => undefined })}</>);
    expect(screen.getByRole("link", { name: /GWAS Catalog/ })).toHaveAttribute("href", "https://www.ebi.ac.uk/gwas/genes/CD14");
    expect(screen.getByRole("link", { name: /GTEx/ })).toHaveAttribute("href", "https://gtexportal.org/home/gene/CD14");
  });
});
