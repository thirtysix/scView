import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({
  makeSubset: vi.fn(),
  getDataset: vi.fn(),
  listDatasets: vi.fn(),
}));
vi.mock("@/api/subset", () => ({ makeSubset: api.makeSubset }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset, listDatasets: api.listDatasets }));

import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { decodeMask } from "@/lib/cellFilter";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { defaultSubsetName, SubsetSelectionControls, subsetSelector } from "./SubsetSelection";

const dataset: DatasetInfo = {
  id: "ds",
  name: "pbmc",
  filename: "pbmc.h5ad",
  n_cells: 300,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [],
  created_at: "",
};

const plan = {
  dataset_id: null,
  name: "pbmc - 120 selected cells",
  n_cells: 120,
  n_parent_cells: 300,
  described: "a selection of 120 cells",
  carried: ["5 obs columns", "the parent's UMAP, as 'UMAP (parent)'"],
  not_carried: ["PCA, the neighbour graph and other layouts: processing computes them again"],
  renamed: {},
  dry_run: true,
  next: "",
};

const range = (n: number) => new Set(Array.from({ length: n }, (_, i) => i * 2));

describe("SubsetSelectionControls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds", availableDatasets: [] });
    useSelectionStore.setState({ selectedCellIndices: range(120), selectionFilter: null, selectionGroups: null });
    api.listDatasets.mockResolvedValue([]);
  });

  it("is hidden with no selection", () => {
    useSelectionStore.setState({ selectedCellIndices: null });
    const { container } = render(<SubsetSelectionControls />);
    expect(container).toBeEmptyDOMElement();
  });

  it("checks what the subset holds, makes it from the cells and opens it", async () => {
    api.makeSubset.mockImplementation((_id: string, _sel: unknown, opts: { dryRun?: boolean }) =>
      Promise.resolve(opts.dryRun ? plan : { ...plan, dataset_id: "new1", dry_run: false, n_genes: 80, name: "T cells" }),
    );
    api.getDataset.mockResolvedValue({ ...dataset, id: "new1", available_embeddings: ["X_umap_parent"] });
    render(<SubsetSelectionControls />);
    fireEvent.click(screen.getByRole("button", { name: /New dataset/ }));
    const dialog = screen.getByRole("dialog", { name: "New dataset from the selection" });
    expect(dialog).toHaveTextContent("New dataset from 120 selected cells");
    await waitFor(() => expect(screen.getByTestId("subset-plan")).toHaveTextContent("UMAP (parent)"));
    expect(screen.getByTestId("subset-plan")).toHaveTextContent("Not carried: PCA");

    const [id, selector, opts] = api.makeSubset.mock.calls[0]!;
    expect(id).toBe("ds");
    expect(opts).toMatchObject({ dryRun: true });
    expect([...decodeMask((selector as { mask: string }).mask, 300)]).toEqual([...range(120)]);

    const name = screen.getByRole("textbox");
    expect(name).toHaveValue("pbmc - 120 selected cells");
    fireEvent.change(name, { target: { value: "T cells" } });
    fireEvent.click(screen.getByRole("button", { name: "Make dataset" }));
    await waitFor(() => expect(screen.getByTestId("subset-made")).toHaveTextContent("Made \u201cT cells\u201d: 120 cells x 80 genes"));
    expect(api.makeSubset.mock.calls[1]![2]).toEqual({ name: "T cells" });
    expect(api.listDatasets).toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Open it to process" }));
    await waitFor(() => expect(useDatasetStore.getState().currentDataset?.id).toBe("new1"));
    expect(useSettingsStore.getState().embedding).toBe("X_umap_parent");
    expect(useViewStore.getState().activePanel).toBe("assessment");
  });

  it("sends the filter a selection came from, and shows a refusal", async () => {
    const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "CD3E", op: ">", value: 1 }] };
    useSelectionStore.setState({ selectedCellIndices: range(60), selectionFilter: filter });
    api.makeSubset.mockRejectedValue(new Error("All 300 cells match: that is the whole dataset. Choose fewer cells."));
    render(<SubsetSelectionControls />);
    fireEvent.click(screen.getByRole("button", { name: /New dataset/ }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("whole dataset"));
    expect(api.makeSubset.mock.calls[0]![1]).toEqual({ filter });
    expect(screen.getByRole("button", { name: "Make dataset" })).toBeDisabled();
    expect(screen.getByRole("textbox")).toHaveValue("pbmc - 60 filtered cells");
  });

  it("records a group pick as groups and names the subset after them", async () => {
    useSelectionStore.getState().setGroupSelection(range(120), "author_fine", ["Neutrophil 1", "Neutrophil 2"]);
    api.makeSubset.mockResolvedValue({ ...plan, steps_note: "Left out: filtering. The parent was filtered by scView." });
    render(<SubsetSelectionControls />);
    fireEvent.click(screen.getByRole("button", { name: /New dataset/ }));
    await waitFor(() => expect(screen.getByTestId("subset-plan")).toHaveTextContent("Left out: filtering"));
    expect(api.makeSubset.mock.calls[0]![1]).toEqual({
      groups: { column: "author_fine", values: ["Neutrophil 1", "Neutrophil 2"] },
    });
    expect(screen.getByRole("textbox", { name: "Name" })).toHaveValue("pbmc - Neutrophil 1, Neutrophil 2");
    expect(screen.getByRole("dialog")).toHaveTextContent("Chosen as author_fine: Neutrophil 1, Neutrophil 2");
  });

  it("says too few cells without asking the server", () => {
    useSelectionStore.setState({ selectedCellIndices: range(12) });
    render(<SubsetSelectionControls />);
    fireEvent.click(screen.getByRole("button", { name: /New dataset/ }));
    expect(screen.getByRole("alert")).toHaveTextContent("at least 50 cells; 12 are selected");
    expect(api.makeSubset).not.toHaveBeenCalled();
  });
});

describe("subset helpers", () => {
  it("packs a selection as the server's mask, or keeps its filter", () => {
    const filter: CellFilter = { op: "or", layer: "", items: [{ kind: "category", column: "c", values: ["a"] }] };
    expect(subsetSelector(new Set([1]), filter, 10)).toEqual({ filter });
    const sel = subsetSelector(new Set([0, 9]), null, 10) as { mask: string };
    expect([...decodeMask(sel.mask, 10)]).toEqual([0, 9]);
  });

  it("names it as the server would, within 120 characters", () => {
    expect(defaultSubsetName("kidney", 4120, false)).toBe("kidney - 4,120 selected cells");
    expect(defaultSubsetName("x".repeat(130), 5, true)).toHaveLength(120);
  });
});
