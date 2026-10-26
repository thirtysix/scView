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

  it("renders a fenced code block verbatim, keeping its line breaks", () => {
    const text = "Run this:\n```python\nimport scanpy as sc\nsc.pl.dotplot(adata, **kw)\n```\nDone.";
    const { container } = render(<MarkdownLite text={text} />);
    expect(container.querySelector("pre code")?.textContent).toBe(
      "import scanpy as sc\nsc.pl.dotplot(adata, **kw)",
    );
    expect(container.querySelector("pre strong")).toBeNull(); // no inline markdown inside code
    expect(container.querySelectorAll("p")).toHaveLength(2);
  });

  it("renders an unclosed code fence (answer still streaming) as code", () => {
    const { container } = render(<MarkdownLite text={"```\nx = 1"} />);
    expect(container.querySelector("pre code")?.textContent).toBe("x = 1");
  });

  it("does not treat a markdown link as a citation chip", () => {
    const { container } = render(<MarkdownLite text="A [link: here](http://x.test) stays text." />);
    // No chip button; the bracket text remains as plain text.
    expect(container.querySelector("button")).toBeNull();
    expect(container.textContent).toContain("[link: here](http://x.test)");
  });

  it("renders inline code inside bold, not literal backticks", () => {
    const { container } = render(<MarkdownLite text="- **`CCL2`**, **`CCL7`** — chemokines" />);
    const codes = container.querySelectorAll("strong > code");
    expect(Array.from(codes).map((c) => c.textContent)).toEqual(["CCL2", "CCL7"]);
    expect(container.textContent).not.toContain("`");
  });

  it("renders a GFM pipe table with alignment and inline markdown in cells", () => {
    // The shape of the 2026-10-26 crispr answer's KO table.
    const text = [
      "Only **10 of 25 targets** have KO cells:",
      "",
      "| Target | KO / total | KO % | DE genes |",
      "|:---|---|:---:|---:|",
      "| **STAT2** | 474 / 588 | 81% | 37 |",
      "| `JAK2` | 724 / 924 | 78% | 508 [result:mixscape] |",
      "",
      "After the table.",
    ].join("\n");
    const { container } = render(<MarkdownLite text={text} />);
    const ths = container.querySelectorAll("table thead th");
    expect(Array.from(ths).map((t) => t.textContent)).toEqual(["Target", "KO / total", "KO %", "DE genes"]);
    expect((ths[0] as HTMLElement).style.textAlign).toBe("left");
    expect((ths[2] as HTMLElement).style.textAlign).toBe("center");
    expect((ths[3] as HTMLElement).style.textAlign).toBe("right");
    const rows = container.querySelectorAll("table tbody tr");
    expect(rows).toHaveLength(2);
    expect(rows[0]?.querySelector("strong")?.textContent).toBe("STAT2");
    expect(rows[1]?.querySelector("code")?.textContent).toBe("JAK2");
    expect(rows[1]?.textContent).not.toContain("[result:"); // a citation chip, not the tag
    expect(container.textContent).not.toContain("|");
    expect(container.querySelectorAll("p")).toHaveLength(2); // the lines around the table
  });

  it("keeps a lone pipe line as text and pads short rows", () => {
    const { container } = render(<MarkdownLite text={"a | b without a rule\n\n| x | y |\n|---|---|\n| 1 |"} />);
    expect(container.querySelector("p")?.textContent).toBe("a | b without a rule");
    const cells = container.querySelectorAll("table tbody td");
    expect(Array.from(cells).map((c) => c.textContent)).toEqual(["1", ""]);
  });
});
