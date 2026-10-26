import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  listLabelColumns: vi.fn(),
  labelCells: vi.fn(),
  undoLabel: vi.fn(),
  getDataset: vi.fn(),
}));
vi.mock("@/api/labels", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/labels")>()),
  listLabelColumns: api.listLabelColumns,
  labelCells: api.labelCells,
  undoLabel: api.undoLabel,
}));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
const onto = vi.hoisted(() => ({ searchCellTypes: vi.fn() }));
vi.mock("@/api/ontology", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/ontology")>()),
  searchCellTypes: onto.searchCellTypes,
}));

import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useLabelUndoStore } from "@/stores/labelUndoStore";
import { useObsRevisionStore } from "@/stores/obsRevisionStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { LabelSelectionControls } from "./LabelSelection";

const dataset = (extra: DatasetInfo["obs_columns"] = []): DatasetInfo => ({
  id: "ds",
  name: "ds",
  filename: "ds.h5ad",
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "leiden", dtype: "category", n_unique: 3, values: ["0", "1", "2"] }, ...extra],
  active_clustering: "leiden",
  created_at: "",
});

function renderControls() {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <LabelSelectionControls />
    </QueryClientProvider>,
  );
}

describe("LabelSelectionControls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ currentDataset: dataset(), currentDatasetId: "ds" });
    useSelectionStore.setState({ selectedCellIndices: new Set([4, 2, 9]), highlightedGroup: null });
    useLabelUndoStore.setState({ stacks: {}, lastColumn: {} });
    useObsRevisionStore.setState({ revisions: {} });
    useSettingsStore.setState({ colorBy: "leiden" });
  });

  it("is hidden with no selection and nothing to undo", () => {
    useSelectionStore.setState({ selectedCellIndices: null });
    const { container } = renderControls();
    expect(container).toBeEmptyDOMElement();
  });

  it("labels the selection into a new column, shows it, and undoes it", async () => {
    api.listLabelColumns.mockResolvedValue({ columns: [], unassigned: "Unassigned" });
    api.labelCells.mockResolvedValue({
      column: "manual_labels",
      label: "Doublets",
      n_cells: 3,
      created: true,
      previous: [{ value: "Unassigned", indices: [2, 4, 9] }],
    });
    const labelled = dataset([
      { name: "manual_labels", dtype: "category", n_unique: 2, values: ["Unassigned", "Doublets"] },
    ]);
    api.getDataset.mockResolvedValueOnce(labelled).mockResolvedValueOnce(dataset());
    api.undoLabel.mockResolvedValue({ column: "manual_labels", n_cells: 3, removed: true });

    renderControls();
    fireEvent.click(screen.getByRole("button", { name: /Label selection/ }));
    expect(screen.getByText("Label 3 selected cells")).toBeInTheDocument();
    await waitFor(() => expect(api.listLabelColumns).toHaveBeenCalledWith("ds"));
    fireEvent.change(screen.getByPlaceholderText(/activated T cells/), { target: { value: " Doublets " } });
    fireEvent.click(screen.getByRole("button", { name: "Apply label" }));

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    const call = api.labelCells.mock.calls[0]!;
    expect(call[0]).toBe("ds");
    expect(call[1]).toMatchObject({ column: "manual_labels", label: "Doublets", newColumn: true });
    expect([...call[1].indices].sort((a: number, b: number) => a - b)).toEqual([2, 4, 9]);
    // The new column is listed and coloured by straight away.
    expect(useDatasetStore.getState().currentDataset).toBe(labelled);
    expect(useSettingsStore.getState().colorBy).toBe("manual_labels");
    expect(useObsRevisionStore.getState().revisions["ds\u0000manual_labels"]).toBe(1);

    fireEvent.click(screen.getByRole("button", { name: /Undo label/ }));
    await waitFor(() =>
      expect(api.undoLabel).toHaveBeenCalledWith("ds", {
        column: "manual_labels",
        previous: [{ value: "Unassigned", indices: [2, 4, 9] }],
        created: true,
        previous_terms: {},
      }),
    );
    // The column is gone, so the plot goes back to the clustering.
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe("leiden"));
    expect(screen.queryByRole("button", { name: /Undo label/ })).not.toBeInTheDocument();
  });

  it("attaches a Cell Ontology term picked in the label box, and the undo puts the old term back", async () => {
    onto.searchCellTypes.mockResolvedValue({
      query: "nk",
      version: "v",
      n_terms: 1,
      terms: [
        {
          id: "CL:0000623", name: "natural killer cell", matched: "NK cell", match: "synonym",
          definition: "A lymphocyte…", parents: [], url: "",
        },
      ],
    });
    api.listLabelColumns.mockResolvedValue({ columns: ["mine"], unassigned: "Unassigned" });
    api.labelCells.mockResolvedValue({
      column: "mine", label: "NK", n_cells: 3, created: false,
      previous: [{ value: "Unassigned", indices: [2, 4, 9] }],
      previous_terms: { NK: "CL:0000576" },
    });
    api.getDataset.mockResolvedValue(dataset());
    api.undoLabel.mockResolvedValue({ column: "mine", n_cells: 3, removed: false });
    renderControls();
    fireEvent.click(screen.getByRole("button", { name: /Label selection/ }));
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Column" })).toHaveValue("mine"));
    const box = screen.getByRole("combobox", { name: "Label" });
    fireEvent.change(box, { target: { value: "nk" } });
    await screen.findByRole("option", { name: /natural killer cell/ });
    fireEvent.keyDown(box, { key: "ArrowDown" });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(box).toHaveValue("natural killer cell");
    // the user's own name keeps the term
    fireEvent.change(box, { target: { value: "NK" } });
    expect(screen.getByText("CL:0000623")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Apply label" }));
    await waitFor(() => expect(api.labelCells).toHaveBeenCalled());
    expect(api.labelCells.mock.calls[0]![1]).toMatchObject({ label: "NK", ontologyId: "CL:0000623" });

    fireEvent.click(await screen.findByRole("button", { name: /Undo label/ }));
    await waitFor(() =>
      expect(api.undoLabel).toHaveBeenCalledWith("ds", expect.objectContaining({
        previous_terms: { NK: "CL:0000576" },
      })),
    );
  });

  it("offers the column last labelled into and checks the label before sending", async () => {
    useDatasetStore.setState({
      currentDataset: dataset([
        { name: "mine", dtype: "category", n_unique: 2, values: ["Unassigned", "B cells"] },
      ]),
    });
    useLabelUndoStore.setState({ stacks: {}, lastColumn: { ds: "mine" } });
    api.listLabelColumns.mockResolvedValue({ columns: ["mine"], unassigned: "Unassigned" });
    renderControls();
    fireEvent.click(screen.getByRole("button", { name: /Label selection/ }));
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Column" })).toHaveValue("mine"));
    fireEvent.click(screen.getByRole("button", { name: "Apply label" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Type a label");
    expect(api.labelCells).not.toHaveBeenCalled();
  });

  it("refuses an undo after the dataset's cells changed", async () => {
    useSelectionStore.setState({ selectedCellIndices: null });
    useLabelUndoStore.getState().push({
      datasetId: "ds",
      column: "mine",
      label: "x",
      nCells: 1,
      created: false,
      previous: [{ value: "Unassigned", indices: [0] }],
      datasetCells: 120,
    });
    renderControls();
    fireEvent.click(screen.getByRole("button", { name: /Undo label/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("cells changed");
    expect(api.undoLabel).not.toHaveBeenCalled();
  });
});
