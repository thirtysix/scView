import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { LabelCurationNote, type LabelCuration } from "./LabelCurationNote";

const INFO: LabelCuration = {
  curated_column: "cell_type",
  author_column: "cell_type_original",
  n_author_labels: 3,
  n_cells: 2900,
  rows: [
    { author: "CM", n_cells: 1230, curated: "cardiac muscle cell", share: 0.976, others: [{ curated: "fibroblast", n_cells: 30 }] },
    { author: "Myeloid", n_cells: 900, curated: "immature innate lymphoid cell", share: 1, others: [],
      flag: "'Myeloid' reads as myeloid but its cells are curated as 'immature innate lymphoid cell' (lymphoid)" },
    { author: "Fib", n_cells: 770, curated: "fibroblast", share: 1, others: [] },
  ],
  n_flagged: 1,
  flags: ["'Myeloid' reads as myeloid but its cells are curated as 'immature innate lymphoid cell' (lymphoid)"],
  note: "The curated 'cell_type' may not keep the meaning of the authors' 'cell_type_original' for 1 label. Nothing was renamed.",
};

describe("LabelCurationNote", () => {
  it("says which mappings look wrong, with the cross-tab", () => {
    render(<LabelCurationNote info={INFO} />);
    const note = screen.getByTestId("label-curation-note");
    expect(note).toHaveTextContent("Nothing was renamed.");
    expect(note).toHaveTextContent("'Myeloid' reads as myeloid");
    const table = screen.getByRole("table", { name: "Author labels against curated labels" });
    expect(within(table).getByText("cardiac muscle cell (98%)")).toBeInTheDocument();
    expect(within(table).getByText("fibroblast (30)")).toBeInTheDocument();
    expect(within(table).getByText("Myeloid").closest("tr")!.className).toContain("text-amber-800");
  });

  it("is silent when nothing is flagged or there is no pair", () => {
    const { container } = render(<LabelCurationNote info={{ ...INFO, n_flagged: 0, flags: [], note: null }} />);
    expect(container).toBeEmptyDOMElement();
    const { container: none } = render(<LabelCurationNote info={null} />);
    expect(none).toBeEmptyDOMElement();
  });
});
