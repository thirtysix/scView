import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { useState } from "react";

const onto = vi.hoisted(() => ({ searchCellTypes: vi.fn() }));
vi.mock("@/api/ontology", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/ontology")>()),
  searchCellTypes: onto.searchCellTypes,
}));

import { DialogBox } from "./DialogBox";
import { CellTypeCombobox, type PickedTerm } from "./CellTypeCombobox";
import { seriousViolations } from "@/test/axe";

const TERMS = [
  { id: "CL:0001054", name: "CD14-positive monocyte", matched: null, match: "words", definition: "A monocyte…", parents: [], url: "" },
  { id: "CL:0000576", name: "monocyte", matched: null, match: "tokens", definition: null, parents: [], url: "" },
];

function Harness({ onClose = () => {}, known = [] as string[] }) {
  const [value, setValue] = useState("");
  const [term, setTerm] = useState<PickedTerm | null>(null);
  return (
    <DialogBox label="Test dialog" onClose={onClose}>
      <CellTypeCombobox label="Label" value={value} onChange={setValue} term={term} onTermChange={setTerm} knownLabels={known} />
      <output data-testid="term">{term ? term.id : "none"}</output>
    </DialogBox>
  );
}

describe("CellTypeCombobox", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    onto.searchCellTypes.mockResolvedValue({ query: "", terms: TERMS, version: "v", n_terms: 2 });
  });

  it("searches what the user types and picks a term by keyboard", async () => {
    render(<Harness known={["CD14 Mono (old)"]} />);
    const box = screen.getByRole("combobox", { name: "Label" });
    expect(box).toHaveAttribute("aria-expanded", "false");
    fireEvent.change(box, { target: { value: "cd14" } });
    await waitFor(() => expect(onto.searchCellTypes).toHaveBeenCalledWith("cd14", 10, expect.anything()));
    const options = await screen.findAllByRole("option");
    expect(options.map((o) => o.textContent)).toEqual([
      "CD14 Mono (old) in this column",
      "CD14-positive monocyteCL:0001054",
      "monocyteCL:0000576",
    ]);
    expect(box).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("2 Cell Ontology terms, 1 label in this column")).toHaveAttribute("aria-live", "polite");
    fireEvent.keyDown(box, { key: "ArrowDown" });
    fireEvent.keyDown(box, { key: "ArrowDown" });
    expect(box).toHaveAttribute("aria-activedescendant", options[1]!.id);
    expect(options[1]).toHaveAttribute("aria-selected", "true");
    fireEvent.keyDown(box, { key: "Enter" });
    expect(box).toHaveValue("CD14-positive monocyte");
    expect(screen.getByTestId("term")).toHaveTextContent("CL:0001054");
    expect(box).toHaveAttribute("aria-expanded", "false");
    const link = screen.getByRole("link", { name: "CL:0001054" });
    expect(link).toHaveAttribute("href", expect.stringContaining("ols4/ontologies/cl/classes?obo_id=CL%3A0001054"));
    fireEvent.click(screen.getByRole("button", { name: /Remove the Cell Ontology term/ }));
    expect(screen.getByTestId("term")).toHaveTextContent("none");
    expect(box).toHaveValue("CD14-positive monocyte"); // the name stays
  });

  it("Escape closes the list before it closes the dialog", async () => {
    const onClose = vi.fn();
    render(<Harness onClose={onClose} />);
    const box = screen.getByRole("combobox", { name: "Label" });
    fireEvent.change(box, { target: { value: "mono" } });
    await screen.findAllByRole("option");
    fireEvent.keyDown(box, { key: "Escape" });
    expect(box).toHaveAttribute("aria-expanded", "false");
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("says when the search is unavailable and keeps free text", async () => {
    onto.searchCellTypes.mockRejectedValue(new Error("The Cell Ontology is not downloaded"));
    render(<Harness />);
    const box = screen.getByRole("combobox", { name: "Label" });
    fireEvent.change(box, { target: { value: "my odd cells" } });
    expect(await screen.findByText(/search unavailable: The Cell Ontology is not downloaded/)).toBeInTheDocument();
    expect(box).toHaveValue("my odd cells");
    expect(screen.getByTestId("term")).toHaveTextContent("none");
  });

  it("has no serious accessibility violations with the list open", async () => {
    const { container } = render(<Harness />);
    fireEvent.change(screen.getByRole("combobox", { name: "Label" }), { target: { value: "mono" } });
    await screen.findAllByRole("option");
    expect(await seriousViolations(container)).toEqual([]);
  });
});
