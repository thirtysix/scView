import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({ filterCells: vi.fn(), cellsMatching: vi.fn() }));
vi.mock("@/api/cells", () => ({ filterCells: api.filterCells, cellsMatching: api.cellsMatching }));

import type { DatasetInfo } from "@/api/types";
import { decodeMask, type CellFilter } from "@/lib/cellFilter";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { CellFilterControl, clearCellFilterDrafts } from "./CellFilter";
import { UnifiedToolbar } from "./UnifiedToolbar";

const dataset: DatasetInfo = {
  id: "ds",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 12,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] },
    { name: "n_counts", dtype: "float64", n_unique: 12 },
  ],
  expression_layers: [{ key: "raw", label: "Log-normalized (raw)", n_genes: 10 }],
  default_expression_layer: "raw",
  created_at: "",
};

/** A response for cells 1, 3 and 9 of 12, as the backend packs them. */
const mask = btoa(String.fromCharCode(0b00001010, 0b00000010));

function open() {
  render(<CellFilterControl />);
  fireEvent.click(screen.getByRole("button", { name: /filter/i }));
  return screen.getByRole("dialog", { name: "Filter cells" });
}

describe("CellFilterControl", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearCellFilterDrafts();
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds" });
    useSelectionStore.setState({
      selectedCellIndices: null,
      selectionFilter: null,
      highlightedGroup: { column: "cell_type", value: "B" },
    });
    useSettingsStore.setState({ expressionLayer: "raw" });
    api.filterCells.mockImplementation(async (_id: string, _f: CellFilter, withMask: boolean) => ({
      n_cells: 12,
      n_matching: 3,
      mask: withMask ? mask : null,
    }));
    api.cellsMatching.mockImplementation(async () => decodeMask(mask, 12));
  });

  it("builds gene AND metadata conditions, counts them live, and selects the cells", async () => {
    const dialog = open();
    const rows = () => within(dialog).getAllByTestId("filter-condition");

    // Row 1: CD14 > 1 (a typed gene is taken on Enter)
    const gene = within(rows()[0]!).getByPlaceholderText("Gene");
    fireEvent.change(gene, { target: { value: "CD14" } });
    fireEvent.keyDown(gene, { key: "Enter" });
    fireEvent.change(within(rows()[0]!).getByLabelText("Value"), { target: { value: "1" } });

    // Row 2: stim = STIM
    fireEvent.click(within(dialog).getByRole("button", { name: "Condition" }));
    fireEvent.change(within(rows()[1]!).getByLabelText("Condition on"), { target: { value: "stim" } });
    fireEvent.click(within(rows()[1]!).getByLabelText("STIM"));

    expect(within(dialog).getByLabelText("Filter as text")).toHaveTextContent("CD14 > 1 AND stim = STIM");
    await waitFor(() => expect(within(dialog).getByRole("status")).toHaveTextContent("3 of 12 cells match"));
    const counted = api.filterCells.mock.calls.at(-1)!;
    expect(counted[2]).toBe(false); // the count only
    expect(counted[1]).toEqual({
      op: "and",
      layer: "raw",
      items: [
        { kind: "gene", gene: "CD14", op: ">", value: 1 },
        { kind: "category", column: "stim", values: ["STIM"] },
      ],
    });

    fireEvent.click(within(dialog).getByRole("button", { name: "Select" }));
    await waitFor(() => expect(useSelectionStore.getState().selectedCellIndices).not.toBeNull());
    const s = useSelectionStore.getState();
    expect([...s.selectedCellIndices!].sort((a, b) => a - b)).toEqual([1, 3, 9]);
    expect(s.selectionFilter).toEqual(counted[1]);
    expect(api.cellsMatching).toHaveBeenCalledWith("ds", counted[1]);
    expect(s.highlightedGroup).toBeNull(); // a filter is not a group, as with a lasso
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("joins a group with OR and leaves the layer out without gene conditions", async () => {
    const dialog = open();
    fireEvent.change(within(dialog).getByLabelText("Condition on"), { target: { value: "n_counts" } });
    fireEvent.change(within(dialog).getByLabelText("Minimum"), { target: { value: "500" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Group" }));
    const rows = within(dialog).getAllByTestId("filter-condition");
    fireEvent.change(within(rows[1]!).getByLabelText("Condition on"), { target: { value: "cell_type" } });
    fireEvent.click(within(rows[1]!).getByLabelText("NK"));
    fireEvent.click(within(rows[1]!).getByLabelText("B"));
    // top level: OR
    fireEvent.click(within(dialog).getAllByRole("radio", { name: /ANY/ })[0]!);

    expect(within(dialog).getByLabelText("Filter as text")).toHaveTextContent("n_counts ≥ 500 OR cell_type in {B, NK}");
    await waitFor(() => expect(api.filterCells).toHaveBeenCalled());
    expect(api.filterCells.mock.calls.at(-1)![1]).toEqual({
      op: "or",
      layer: "",
      items: [
        { kind: "range", column: "n_counts", min: 500, max: null },
        { kind: "group", op: "or", conditions: [{ kind: "category", column: "cell_type", values: ["B", "NK"] }] },
      ],
    });
  });

  it("shows the server's reason and does not select", async () => {
    api.filterCells.mockRejectedValue(new Error("Gene 'CD99' is not in layer 'raw'."));
    const dialog = open();
    const gene = within(dialog).getByPlaceholderText("Gene");
    fireEvent.change(gene, { target: { value: "CD99" } });
    fireEvent.keyDown(gene, { key: "Enter" });
    fireEvent.change(within(dialog).getByLabelText("Value"), { target: { value: "0" } });
    await waitFor(() => expect(within(dialog).getByRole("alert")).toHaveTextContent("not in layer"));
    expect(within(dialog).getByRole("button", { name: "Select" })).toBeDisabled();
  });

  it("reopens on the filter the selection came from", () => {
    useSelectionStore.getState().setSelection(new Set([1]), {
      op: "and",
      layer: "",
      items: [{ kind: "category", column: "stim", values: ["CTRL"] }],
    });
    const dialog = open();
    expect(within(dialog).getByLabelText("Filter as text")).toHaveTextContent("stim = CTRL");
    expect(within(dialog).getByLabelText("CTRL")).toBeChecked();
  });
});

describe("selection store and toolbar", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds" });
  });

  it("forgets the filter when another selection replaces it", () => {
    const f: CellFilter = { op: "and", layer: "", items: [{ kind: "category", column: "stim", values: ["STIM"] }] };
    useSelectionStore.getState().setSelection(new Set([1, 2]), f);
    expect(useSelectionStore.getState().selectionFilter).toBe(f);
    useSelectionStore.getState().setSelection(new Set([5])); // a lasso
    expect(useSelectionStore.getState().selectionFilter).toBeNull();
    useSelectionStore.getState().setSelection(new Set([1, 2]), f);
    useSelectionStore.getState().clearSelection();
    expect(useSelectionStore.getState().selectionFilter).toBeNull();
  });

  it("shows the filter as a sentence next to the selection count", () => {
    useSelectionStore.getState().setSelection(new Set([1, 3, 9]), {
      op: "and",
      layer: "raw",
      items: [
        { kind: "gene", gene: "CD14", op: ">", value: 1 },
        { kind: "category", column: "stim", values: ["STIM"] },
      ],
    });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} />
      </QueryClientProvider>,
    );
    expect(screen.getByText(/3 selected/)).toBeInTheDocument();
    expect(screen.getByLabelText("Selection filter")).toHaveTextContent("by CD14 > 1 AND stim = STIM");
  });
});
