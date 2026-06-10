import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MarkdownLite } from "./MarkdownLite";

describe("MarkdownLite", () => {
  it("renders bold, italic, and code inline spans", () => {
    const { container } = render(<MarkdownLite text="A **bold** and *italic* and `code` word." />);
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("em")?.textContent).toBe("italic");
    expect(container.querySelector("code")?.textContent).toBe("code");
  });

  it("renders bullet lists", () => {
    const { container } = render(<MarkdownLite text={"- one\n- two\n- three"} />);
    expect(container.querySelectorAll("ul li")).toHaveLength(3);
  });

  it("renders numbered lists", () => {
    const { container } = render(<MarkdownLite text={"1. first\n2. second"} />);
    expect(container.querySelectorAll("ol li")).toHaveLength(2);
  });

  it("turns a [result:...] token into a clickable citation chip", () => {
    const onCitation = vi.fn();
    render(<MarkdownLite text="See [result:groups:NK cells] for details." onCitation={onCitation} />);
    fireEvent.click(screen.getByRole("button", { name: "NK cells" }));
    expect(onCitation).toHaveBeenCalledWith("result:groups:NK cells");
  });

  it("renders a reformatted citation tag as a chip, not raw brackets", () => {
    render(<MarkdownLite text="Per [cell-type annotation: B cells], that cluster is B." />);
    // The bracketed text should not survive verbatim as a citation.
    expect(screen.queryByText(/\[cell-type annotation/)).toBeNull();
    expect(screen.getByText("cell type")).toBeInTheDocument();
  });

  it("does not treat a markdown link as a citation chip", () => {
    const { container } = render(<MarkdownLite text="A [link: here](http://x.test) stays text." />);
    // No chip button; the bracket text remains as plain text.
    expect(container.querySelector("button")).toBeNull();
    expect(container.textContent).toContain("[link: here](http://x.test)");
  });
});
