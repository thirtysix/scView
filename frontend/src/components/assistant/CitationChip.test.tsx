import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CitationChip } from "./CitationChip";

describe("CitationChip", () => {
  it("renders a literature tag as an external PubMed link", () => {
    render(<CitationChip tag="lit:PMID:12345" />);
    const link = screen.getByRole("link", { name: "PMID 12345" });
    expect(link).toHaveAttribute("href", "https://pubmed.ncbi.nlm.nih.gov/12345/");
    expect(link).toHaveAttribute("target", "_blank");
  });

  it("renders a result tag as a clickable chip that fires onClick with the tag", () => {
    const onClick = vi.fn();
    render(<CitationChip tag="result:groups:NK cells" onClick={onClick} />);
    const btn = screen.getByRole("button", { name: "NK cells" });
    fireEvent.click(btn);
    expect(onClick).toHaveBeenCalledWith("result:groups:NK cells");
  });

  it("treats a reformatted data tag (cell-type ...) as clickable", () => {
    const onClick = vi.fn();
    render(<CitationChip tag="cell-type annotation:Monocyte" onClick={onClick} />);
    fireEvent.click(screen.getByRole("button", { name: "cell type" }));
    expect(onClick).toHaveBeenCalledWith("cell-type annotation:Monocyte");
  });

  it("renders an unknown/non-data tag as a non-interactive chip", () => {
    const onClick = vi.fn();
    render(<CitationChip tag="provenance:clustering" onClick={onClick} />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByText("steps")).toBeInTheDocument();
  });
});
