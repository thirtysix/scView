import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent, act, waitFor, within } from "@testing-library/react";
import type { CoexpressionResponse } from "@/api/coexpression";
import { sortCoexpressed } from "@/lib/coexpressedGenes";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useGeneListsStore } from "@/stores/geneListsStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

const { fetchCoexpression } = vi.hoisted(() => ({ fetchCoexpression: vi.fn() }));
vi.mock("@/api/coexpression", () => ({ fetchCoexpression }));

import { CoexpressedGenes } from "./CoexpressedGenes";

const g = (gene: string, r: number, n = 100) => ({ gene, r, n_expressing: n, pct_expressing: n / 10 });

const RESULT: CoexpressionResponse = {
  gene: "ISG15",
  method: "pearson",
  scope: "all cells",
  n_cells: 13836,
  n_cells_total: 13836,
  query_n_expressing: 9099,
  min_expressing: 139,
  n_genes_tested: 7380,
  n_skipped_rare: 6672,
  n_skipped_constant: 0,
  layer: "raw",
  values: "log-normalised",
  positive: [g("IFIT3", 0.846, 7046), g("IFIT1", 0.8449, 6712), g("IFI6", 0.8221, 8085)],
  negative: [g("RPL7", -0.4532, 13108), g("RPL6", -0.4514, 11761)],
};

const onGeneSelect = vi.fn();
const renderIt = (activeGene: string | null = "ISG15") =>
  render(<CoexpressedGenes datasetId="kang" activeGene={activeGene} onGeneSelect={onGeneSelect} />);
const rowsOf = (name: string) =>
  within(screen.getByRole("table", { name })).getAllByRole("row").slice(1).map((r) => r.textContent ?? "");

beforeEach(() => {
  fetchCoexpression.mockReset().mockResolvedValue(RESULT);
  onGeneSelect.mockReset();
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  useGeneGridStore.setState({ datasetId: null, genes: [], pair: null });
  useSplitViewStore.setState({ geneGrid: null, splitBy: null, compareColumns: null });
});

describe("CoexpressedGenes", () => {
  it("adds the query and the shown partners to a new gene list", async () => {
    useGeneListsStore.setState({ lists: [], selectedId: null });
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    await screen.findByText("IFIT3");
    fireEvent.click(screen.getByRole("button", { name: "Negative (2)" }));
    fireEvent.click(screen.getByRole("button", { name: "Add to gene list" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "New list…" }));
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    const [list] = useGeneListsStore.getState().lists;
    expect([list!.name, list!.genes]).toEqual(["Co-expressed with ISG15", ["ISG15", "RPL7", "RPL6"]]);
  });

  it("asks for the overlaid gene over every cell and lists both directions", async () => {
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: "Find genes co-expressed with ISG15" }));
    await screen.findByText(/over 13,836 cells/);
    expect(fetchCoexpression).toHaveBeenCalledWith("kang", { gene: "ISG15", method: "pearson", top_n: 25 });
    expect(screen.getByText(/6,672 skipped, expressed in fewer than 139 of these cells/)).toBeTruthy();
    expect(rowsOf("positive co-expressed genes")[0]).toContain("IFIT3");
    expect(rowsOf("positive co-expressed genes")[0]).toContain("0.846");
    fireEvent.click(screen.getByRole("button", { name: "Negative (2)" }));
    expect(rowsOf("negative co-expressed genes")[0]).toContain("RPL7");
    expect(rowsOf("negative co-expressed genes")[0]).toContain("-0.453");
  });

  it("overlays a clicked gene", async () => {
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    fireEvent.click(await screen.findByText("IFI6"));
    expect(onGeneSelect).toHaveBeenCalledWith("IFI6");
  });

  it("sorts by any column", async () => {
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    await screen.findByText("IFIT3");
    fireEvent.click(screen.getByRole("button", { name: "Expressing" }));
    expect(rowsOf("positive co-expressed genes")[0]).toContain("IFI6"); // most cells first
    fireEvent.click(screen.getByRole("button", { name: "Gene" }));
    expect(rowsOf("positive co-expressed genes")[0]).toContain("IFI6"); // A-Z
  });

  it("restricts to the selection or the highlighted group", async () => {
    act(() => useSelectionStore.setState({ selectedCellIndices: new Set([1, 2, 3]) }));
    renderIt();
    const scope = screen.getByRole("combobox", { name: "Cells to correlate over" }) as HTMLSelectElement;
    // Three cells are too few to pick.
    expect((within(scope).getByText(/Selection \(needs 10\+ cells\)/) as HTMLOptionElement).disabled).toBe(true);
    act(() =>
      useSelectionStore.setState({
        selectedCellIndices: new Set(Array.from({ length: 12 }, (_, i) => i)),
        highlightedGroup: { column: "cell_type", value: "CD14 Mono" },
      }),
    );
    fireEvent.change(scope, { target: { value: "selection" } });
    fireEvent.click(screen.getByRole("button", { name: "Spearman" }));
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    await screen.findByText("IFIT3");
    expect(fetchCoexpression.mock.calls[0]![1]).toMatchObject({
      method: "spearman",
      indices: Array.from({ length: 12 }, (_, i) => i),
    });
    fireEvent.change(scope, { target: { value: "group" } });
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    await waitFor(() => expect(fetchCoexpression).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByRole("button", { name: /Find genes/ })).toBeEnabled());
    expect(fetchCoexpression.mock.calls[1]![1]).toMatchObject({ column: "cell_type", value: "CD14 Mono" });
    expect(fetchCoexpression.mock.calls[1]![1]).not.toHaveProperty("indices");
  });

  it("sends the query and its partners to Several genes, and an open feature grid follows", async () => {
    useSplitViewStore.getState().setGeneGrid({ datasetId: "kang", kind: "features", genes: ["CD3D", "CD14"] });
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    await screen.findByText("IFIT3");
    fireEvent.click(screen.getByRole("button", { name: "To grid" }));
    expect(useGeneGridStore.getState().genes).toEqual(["ISG15", "IFIT3", "IFIT1", "IFI6"]);
    expect(useSplitViewStore.getState().geneGrid?.genes).toEqual(["ISG15", "IFIT3", "IFIT1", "IFI6"]);
  });

  it("shows the server's reason when a gene cannot be correlated", async () => {
    fetchCoexpression.mockRejectedValue(new Error("ISG15 is expressed in 3 of these 40 cells"));
    renderIt();
    fireEvent.click(screen.getByRole("button", { name: /Find genes/ }));
    expect(await screen.findByText(/expressed in 3 of these 40 cells/)).toBeTruthy();
  });
});

describe("sortCoexpressed", () => {
  it("orders by |r| so the strongest negative comes first too", () => {
    const neg = [g("A", -0.2), g("B", -0.6), g("C", -0.4)];
    expect(sortCoexpressed(neg, "r", true).map((x) => x.gene)).toEqual(["B", "C", "A"]);
    expect(sortCoexpressed(neg, "r", false).map((x) => x.gene)).toEqual(["A", "C", "B"]);
  });
});
