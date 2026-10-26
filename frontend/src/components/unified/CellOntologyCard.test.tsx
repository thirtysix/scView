import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";

const onto = vi.hoisted(() => ({
  getLabelTerms: vi.fn(),
  setLabelTerms: vi.fn(),
  clearLabelTerms: vi.fn(),
  suggestLabelTerms: vi.fn(),
  searchCellTypes: vi.fn(),
}));
vi.mock("@/api/ontology", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/ontology")>()),
  ...onto,
}));

import type { LabelTerms } from "@/api/ontology";
import { seriousViolations } from "@/test/axe";
import { CellOntologyCard } from "./CellOntologyCard";

const row = (label: string, n: number, id: string | null = null, name: string | null = null) => ({
  label, n_cells: n, id, name, url: null,
});

function terms(rows: ReturnType<typeof row>[]): LabelTerms {
  const missing = rows.filter((r) => !r.id).map((r) => r.label);
  return {
    column: "cell_type", labels: rows, n_labels: rows.length, n_mapped: rows.length - missing.length,
    missing, n_cells_unlabelled: 0, complete: missing.length === 0, version: "v", columns: ["cell_type"],
  };
}

const sug = (id: string, name: string, how = "exact") => ({
  id, name, matched: null, match: null, definition: null, parents: [], url: "", how, how_text: "same name",
});

describe("CellOntologyCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    onto.getLabelTerms.mockResolvedValue(terms([row("CD14 Mono", 40), row("NK", 10, "CL:0000623", "natural killer cell")]));
  });

  it("lists each label with its term and counts", async () => {
    render(<CellOntologyCard datasetId="ds" column="cell_type" onClose={() => {}} />);
    const dialog = await screen.findByRole("dialog", { name: /Cell Ontology terms of/ });
    expect(await within(dialog).findByText("1 of 2 labels have a term.")).toBeInTheDocument();
    expect(within(dialog).getByRole("link", { name: "CL:0000623" })).toBeInTheDocument();
    expect(within(dialog).getByText("No term")).toBeInTheDocument();
    expect(onto.getLabelTerms).toHaveBeenCalledWith("ds", "cell_type");
    expect(await seriousViolations(dialog)).toEqual([]);
  });

  it("suggests terms; accepting saves one, rejecting saves nothing", async () => {
    onto.suggestLabelTerms.mockResolvedValue({
      column: "cell_type", n_labels: 2, n_suggested: 2, version: "v",
      labels: [
        { label: "CD14 Mono", n_cells: 40, current: null,
          suggestions: [sug("CL:0001054", "CD14-positive monocyte"), sug("CL:0000576", "monocyte", "subset")] },
        { label: "NK", n_cells: 10, current: "CL:0000623", suggestions: [sug("CL:0000623", "natural killer cell")] },
      ],
    });
    onto.setLabelTerms.mockResolvedValue(
      terms([row("CD14 Mono", 40, "CL:0001054", "CD14-positive monocyte"), row("NK", 10, "CL:0000623", "natural killer cell")]),
    );
    render(<CellOntologyCard datasetId="ds" column="cell_type" onClose={() => {}} />);
    await screen.findByText("1 of 2 labels have a term.");
    fireEvent.click(screen.getByRole("button", { name: /Suggest terms/ }));
    // NK already has its suggested term: nothing to show for it
    expect(await screen.findByText(/Suggestions for 1 label; accept or reject each/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Accept natural killer cell/ })).not.toBeInTheDocument();
    expect(onto.setLabelTerms).not.toHaveBeenCalled(); // suggesting saves nothing
    fireEvent.click(screen.getByRole("button", { name: /Accept CD14-positive monocyte for CD14 Mono/ }));
    await waitFor(() => expect(onto.setLabelTerms).toHaveBeenCalledWith("ds", "cell_type", { "CD14 Mono": "CL:0001054" }));
    expect(await screen.findByText(/2 of 2 labels have a term. Complete/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Accept monocyte/ })).not.toBeInTheDocument();
  });

  it("rejecting a suggestion only hides it", async () => {
    onto.suggestLabelTerms.mockResolvedValue({
      column: "cell_type", n_labels: 2, n_suggested: 1, version: "v",
      labels: [{ label: "CD14 Mono", n_cells: 40, current: null, suggestions: [sug("CL:0001054", "CD14-positive monocyte")] }],
    });
    render(<CellOntologyCard datasetId="ds" column="cell_type" onClose={() => {}} />);
    await screen.findByText("1 of 2 labels have a term.");
    fireEvent.click(screen.getByRole("button", { name: /Suggest terms/ }));
    fireEvent.click(await screen.findByRole("button", { name: /Reject the suggestions for CD14 Mono/ }));
    expect(screen.queryByRole("button", { name: /Accept CD14-positive monocyte/ })).not.toBeInTheDocument();
    expect(onto.setLabelTerms).not.toHaveBeenCalled();
  });

  it("marks a label unknown, clears one, and clears all after confirming", async () => {
    onto.setLabelTerms.mockResolvedValue(terms([row("CD14 Mono", 40, "unknown"), row("NK", 10, "CL:0000623", "natural killer cell")]));
    onto.clearLabelTerms.mockResolvedValue(terms([row("CD14 Mono", 40), row("NK", 10)]));
    render(<CellOntologyCard datasetId="ds" column="cell_type" onClose={() => {}} />);
    await screen.findByText("1 of 2 labels have a term.");
    fireEvent.click(screen.getByRole("button", { name: "Unknown cell type for CD14 Mono" }));
    await waitFor(() => expect(onto.setLabelTerms).toHaveBeenCalledWith("ds", "cell_type", { "CD14 Mono": "unknown" }));
    expect(await screen.findByText("unknown cell type")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear the term of NK" }));
    await waitFor(() => expect(onto.setLabelTerms).toHaveBeenLastCalledWith("ds", "cell_type", { NK: null }));
    fireEvent.click(screen.getByRole("button", { name: "Clear all terms" }));
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    await waitFor(() => expect(onto.clearLabelTerms).toHaveBeenCalledWith("ds", "cell_type"));
    expect(await screen.findByText("0 of 2 labels have a term.")).toBeInTheDocument();
  });

  it("changes a term by searching", async () => {
    onto.searchCellTypes.mockResolvedValue({
      query: "mono", version: "v", n_terms: 1,
      terms: [{ id: "CL:0000576", name: "monocyte", matched: null, match: "name", definition: null, parents: [], url: "" }],
    });
    onto.setLabelTerms.mockResolvedValue(terms([row("CD14 Mono", 40, "CL:0000576", "monocyte"), row("NK", 10, "CL:0000623")]));
    render(<CellOntologyCard datasetId="ds" column="cell_type" onClose={() => {}} />);
    await screen.findByText("1 of 2 labels have a term.");
    fireEvent.click(screen.getByRole("button", { name: "Choose the term of CD14 Mono" }));
    const box = screen.getByRole("combobox", { name: "Search a term for CD14 Mono" });
    fireEvent.change(box, { target: { value: "mono" } });
    fireEvent.click(await screen.findByRole("option", { name: /monocyte/ }));
    await waitFor(() => expect(onto.setLabelTerms).toHaveBeenCalledWith("ds", "cell_type", { "CD14 Mono": "CL:0000576" }));
  });

  it("shows why the mapping could not load", async () => {
    onto.getLabelTerms.mockRejectedValue(new Error("'x' is not a label column."));
    render(<CellOntologyCard datasetId="ds" column="x" onClose={() => {}} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("not a label column");
  });
});
