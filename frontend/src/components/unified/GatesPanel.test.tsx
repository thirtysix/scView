import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  filterCells: vi.fn(),
  labelCells: vi.fn(),
  getDataset: vi.fn(),
  obsColumnValues: vi.fn(),
  downloadCsv: vi.fn(),
}));
vi.mock("@/api/cells", () => ({ filterCells: api.filterCells, cellsMatching: vi.fn() }));
vi.mock("@/api/labels", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/api/labels")>()),
  labelCells: api.labelCells,
}));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/gates", () => ({ obsColumnValues: api.obsColumnValues }));
vi.mock("@/lib/csv", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/csv")>()),
  downloadCsv: api.downloadCsv,
}));

import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { bitsetFromIndices, bitsetToSelection } from "@/lib/gates";
import { encodeMask } from "@/lib/session";
import { useDatasetStore } from "@/stores/datasetStore";
import { loadGatesFromStorage, useGatesStore } from "@/stores/gatesStore";
import { useLabelUndoStore } from "@/stores/labelUndoStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { GatesControl } from "./GatesControl";

const cellsSel = (cells: number[], n: number) => bitsetToSelection(bitsetFromIndices(cells, n));
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, k) => a + k);
const kang = (extra: Partial<DatasetInfo> = {}): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "CD14 Mono"] },
    { name: "Donor", dtype: "category", n_unique: 2, values: ["d1", "d2"] },
  ],
  cells_fingerprint: "fp-kang",
  created_at: "",
  ...extra,
});
const filter: CellFilter = { op: "and", layer: "", items: [{ kind: "gene", gene: "ISG15", op: ">", value: 2 }] };

function renderControl() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <div className="flex flex-wrap">
        <GatesControl />
      </div>
    </QueryClientProvider>,
  );
}

const select = (cells: number[], f: CellFilter | null = null) =>
  act(() => useSelectionStore.getState().setSelection(new Set(cells), f));
const row = (name: string) => screen.getByTestId(`gate-row-${name}`);
const cellsOfRow = (name: string) => within(row(name)).getAllByRole("cell").slice(1, 4).map((c) => c.textContent);
const selected = () => [...(useSelectionStore.getState().selectedCellIndices ?? [])].sort((a, b) => a - b);

async function saveGate(name: string) {
  fireEvent.click(screen.getByRole("button", { name: /Save selection as gate/ }));
  const form = screen.getByRole("form", { name: "Save selection as gate" });
  const input = within(form).getAllByRole("textbox")[0]!;
  fireEvent.change(input, { target: { value: name } });
  fireEvent.click(within(form).getByRole("button", { name: "Save gate" }));
  await screen.findByTestId(`gate-row-${name}`);
}

/** Monocytes = cells 0..39 (lasso), ISG-high = lasso 30..59 inside it. */
async function buildTree() {
  renderControl();
  fireEvent.click(screen.getByRole("button", { name: /Gates \(0\)/ }));
  select(range(0, 40));
  await saveGate("Monocytes");
  select(range(30, 60));
  await saveGate("ISG-high");
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  loadGatesFromStorage();
  useGatesStore.setState({ panelOpen: false });
  useDatasetStore.setState({ currentDataset: kang(), currentDatasetId: "kang" });
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, highlightedGroup: null });
  useLabelUndoStore.setState({ stacks: {}, lastColumn: {} });
  api.getDataset.mockResolvedValue(kang());
});

describe("GatesPanel", () => {
  it("opens from the toolbar button; saving needs a selection", () => {
    renderControl();
    fireEvent.click(screen.getByRole("button", { name: /Gates \(0\)/ }));
    expect(screen.getByText(/No gates yet/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save selection as gate/ })).toBeDisabled();
    select([1, 2]);
    expect(screen.getByRole("button", { name: /Save selection as gate/ })).toBeEnabled();
  });

  it("saves a gate, then a child inside it with only the parent's cells, with n and percentages", async () => {
    renderControl();
    fireEvent.click(screen.getByRole("button", { name: /Gates \(0\)/ }));
    select(range(0, 40));
    await saveGate("Monocytes");
    expect(cellsOfRow("Monocytes")).toEqual(["40", "40%", "40%"]);
    expect(screen.getByTestId("gates-active")).toHaveTextContent("New gates go inside Monocytes");
    expect(screen.getByRole("button", { name: /Gates \(1\)/ })).toBeInTheDocument();

    select(range(30, 60)); // a lasso reaching past Monocytes
    fireEvent.click(screen.getByRole("button", { name: /Save selection as gate/ }));
    expect(screen.getByTestId("gate-save-caption")).toHaveTextContent(
      "Saves 10 of the 30 selected cells, the ones inside “Monocytes”; 20 outside it are left out.",
    );
    const form = screen.getByRole("form", { name: "Save selection as gate" });
    fireEvent.change(within(form).getAllByRole("textbox")[0]!, { target: { value: "ISG-high" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save gate" }));
    await screen.findByTestId("gate-row-ISG-high");
    expect(cellsOfRow("ISG-high")).toEqual(["10", "25%", "10%"]);
    expect(screen.getByRole("status")).toHaveTextContent("20 selected cells outside it were left out");
    // The saved child is the selection now.
    expect(selected()).toEqual(range(30, 40));
  });

  it("colours a leaf gate as its label-column colour; a gate with gates inside is hollow", async () => {
    await buildTree();
    const swatch = (name: string) => within(row(name)).getByTestId("gate-swatch");
    expect(swatch("Monocytes")).toHaveClass("border");
    expect(swatch("Monocytes")).toHaveAttribute("title", expect.stringMatching(/Has gates inside it/));
    expect(swatch("ISG-high").getAttribute("style")).toMatch(/background: rgb/);
  });

  it("can save at the top level instead of inside the active gate", async () => {
    await buildTree();
    select(range(90, 100));
    fireEvent.click(screen.getByRole("button", { name: /Save selection as gate/ }));
    const form = screen.getByRole("form", { name: "Save selection as gate" });
    fireEvent.change(within(form).getByRole("combobox", { name: "Parent gate" }), { target: { value: "" } });
    expect(screen.getByTestId("gate-save-caption")).toHaveTextContent("Saves the 10 selected cells at the top level.");
    fireEvent.change(within(form).getAllByRole("textbox")[0]!, { target: { value: "B cells" } });
    fireEvent.click(within(form).getByRole("button", { name: "Save gate" }));
    await screen.findByTestId("gate-row-B cells");
    expect(useGatesStore.getState().records.kang!.gates.find((g) => g.name === "B cells")?.parentId).toBeNull();
  });

  it("saves a Filter's selection as the filter itself", async () => {
    renderControl();
    fireEvent.click(screen.getByRole("button", { name: /Gates \(0\)/ }));
    select([1, 2, 3], filter);
    fireEvent.click(screen.getByRole("button", { name: /Save selection as gate/ }));
    expect(screen.getByTestId("gate-save-caption")).toHaveTextContent("Kept as the filter “ISG15 > 2”");
    fireEvent.click(screen.getByRole("button", { name: "Save gate" }));
    await screen.findByTestId("gate-row-Gate 1");
    expect(useGatesStore.getState().records.kang!.gates[0]!.def).toEqual({ kind: "filter", filter });
    expect(within(row("Gate 1")).getByText("filter")).toBeInTheDocument();
    expect(cellsOfRow("Gate 1")).toEqual(["3", "3%", "3%"]);
    expect(api.filterCells).not.toHaveBeenCalled(); // its cells were known from the selection
  });

  it("counts a stored filter gate by asking the backend", async () => {
    useGatesStore.getState().addGate(kang(), { name: "ISG", parentId: null, def: { kind: "filter", filter } });
    api.filterCells.mockResolvedValue({ n_cells: 100, n_matching: 4, mask: encodeMask([5, 6, 7, 8], 100) });
    renderControl();
    fireEvent.click(screen.getByRole("button", { name: /Gates \(1\)/ }));
    await waitFor(() => expect(cellsOfRow("ISG")).toEqual(["4", "4%", "4%"]));
    expect(api.filterCells).toHaveBeenCalledWith("kang", filter, true);
  });

  it("combines gates with NOT, inside the first gate by default", async () => {
    await buildTree();
    fireEvent.click(screen.getByRole("button", { name: /Combine/ }));
    const form = screen.getByRole("form", { name: "Combine gates" });
    fireEvent.click(within(form).getByRole("radio", { name: "NOT" }));
    expect(within(form).getByRole("combobox", { name: "Start from gate" })).toHaveValue(
      useGatesStore.getState().records.kang!.gates[0]!.id,
    );
    fireEvent.click(within(form).getByRole("checkbox", { name: "ISG-high" }));
    expect(screen.getByTestId("gate-combine-caption")).toHaveTextContent(
      "“Monocytes not ISG-high”: 30 cells, 75% of “Monocytes”.",
    );
    fireEvent.click(within(form).getByRole("button", { name: "Make gate" }));
    await screen.findByTestId("gate-row-Monocytes not ISG-high");
    expect(cellsOfRow("Monocytes not ISG-high")).toEqual(["30", "75%", "30%"]);
    expect(within(row("Monocytes not ISG-high")).getByText("NOT")).toBeInTheDocument();
    expect(selected()).toEqual(range(0, 30));
  });

  it("a click on a gate makes it the selection and the active gate; a second click clears it", async () => {
    await buildTree();
    act(() => useSelectionStore.getState().clearSelection());
    expect(screen.queryByTestId("gates-active")).not.toBeInTheDocument();
    fireEvent.click(within(row("Monocytes")).getByRole("button", { name: "Monocytes" }));
    expect(selected()).toEqual(range(0, 40));
    expect(screen.getByTestId("gates-active")).toHaveTextContent("Monocytes");
    expect(within(row("Monocytes")).getByRole("button", { name: "Show Monocytes on the plot" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(within(row("Monocytes")).getByRole("button", { name: "Show Monocytes on the plot" }));
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });

  it("renames inline, refusing a name already used", async () => {
    await buildTree();
    fireEvent.click(within(row("ISG-high")).getByRole("button", { name: "Rename ISG-high" }));
    const input = within(row("ISG-high")).getByRole("textbox", { name: "Rename ISG-high" });
    fireEvent.change(input, { target: { value: "monocytes" } });
    expect(within(row("ISG-high")).getByRole("alert")).toHaveTextContent("already exists");
    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByTestId("gate-row-ISG-high")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "ISG15 high" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(await screen.findByTestId("gate-row-ISG15 high")).toBeInTheDocument();
  });

  it("deletes a gate with what is inside it, or moves its children up", async () => {
    await buildTree();
    fireEvent.click(within(row("Monocytes")).getByRole("button", { name: "Delete Monocytes" }));
    const dialog = screen.getByRole("dialog", { name: "Delete Monocytes" });
    expect(dialog).toHaveTextContent("has 1 gate directly inside it");
    fireEvent.click(within(dialog).getByRole("button", { name: "Delete it, move its gate up" }));
    expect(screen.queryByTestId("gate-row-Monocytes")).not.toBeInTheDocument();
    expect(useGatesStore.getState().records.kang!.gates.map((g) => [g.name, g.parentId])).toEqual([["ISG-high", null]]);
    // ISG-high was saved as Monocytes' cells, so it keeps them at the top level.
    expect(cellsOfRow("ISG-high")).toEqual(["10", "10%", "10%"]);

    select(range(30, 35));
    await saveGate("Inner");
    fireEvent.click(within(row("ISG-high")).getByRole("button", { name: "Delete ISG-high" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete it and the 1 gate inside" }));
    expect(useGatesStore.getState().records.kang!.gates).toEqual([]);
    expect(screen.getByRole("status")).toHaveTextContent("Deleted “ISG-high”, “Inner”.");
  });

  it("a leaf with nothing built from it is deleted straight away", async () => {
    await buildTree();
    fireEvent.click(within(row("ISG-high")).getByRole("button", { name: "Delete ISG-high" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gate-row-ISG-high")).not.toBeInTheDocument();
  });

  it("writes the leaf gates into one label column, one call per gate, undone in one step", async () => {
    await buildTree();
    select(range(80, 90));
    fireEvent.click(screen.getByRole("button", { name: /Save selection as gate/ }));
    const saveForm = screen.getByRole("form", { name: "Save selection as gate" });
    fireEvent.change(within(saveForm).getByRole("combobox", { name: "Parent gate" }), { target: { value: "" } });
    fireEvent.change(within(saveForm).getAllByRole("textbox")[0]!, { target: { value: "B cells" } });
    fireEvent.click(within(saveForm).getByRole("button", { name: "Save gate" }));
    await screen.findByTestId("gate-row-B cells");

    api.labelCells.mockImplementation(async (_id: string, body: { label: string; indices: number[] }) => ({
      column: "gates",
      label: body.label,
      n_cells: body.indices.length,
      created: true,
      previous: [{ value: "Unassigned", indices: body.indices }],
    }));
    fireEvent.click(screen.getByRole("button", { name: /Label column/ }));
    expect(screen.getByTestId("gate-label-caption")).toHaveTextContent(
      "One label per leaf gate: ISG-high (10), B cells (10); 80 cells in no leaf gate are “Unassigned”. No cell is in more than one leaf gate. 30 of the Unassigned cells are in a gate that has gates inside it",
    );
    fireEvent.click(screen.getByRole("button", { name: "Write labels" }));
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Wrote “gates”: 2 labels"));
    expect(api.labelCells).toHaveBeenCalledTimes(2);
    expect(api.labelCells.mock.calls[0]![1]).toEqual({ column: "gates", label: "ISG-high", indices: range(30, 40), newColumn: true });
    expect(api.labelCells.mock.calls[1]![1]).toEqual({ column: "gates", label: "B cells", indices: range(80, 90), newColumn: false });
    const undo = useLabelUndoStore.getState().stacks.kang!;
    expect(undo).toHaveLength(1);
    expect(undo[0]).toMatchObject({ column: "gates", created: true, nCells: 20, datasetCells: 100 });
    expect(undo[0]!.previous).toEqual([{ value: "Unassigned", indices: [...range(30, 40), ...range(80, 90)] }]);
  });

  it("exports counts with per-donor columns", async () => {
    await buildTree();
    api.obsColumnValues.mockResolvedValue(range(0, 100).map((i) => (i < 35 ? "d1" : "d2")));
    expect(screen.getByRole("combobox", { name: "Per-sample columns in the CSV" })).toHaveValue("Donor");
    fireEvent.click(screen.getByRole("button", { name: /Export counts/ }));
    await waitFor(() => expect(api.downloadCsv).toHaveBeenCalled());
    expect(api.obsColumnValues).toHaveBeenCalledWith("kang", "Donor");
    const [filename, headers, rows] = api.downloadCsv.mock.calls[0]!;
    expect(filename).toBe("scview_Kang_gates.csv");
    expect(headers).toEqual(["gate", "parent", "definition", "n", "pct_parent", "pct_total", "n_d1", "pct_parent_d1", "n_d2", "pct_parent_d2"]);
    expect(rows[1]).toEqual(["Monocytes", "", "40 selected cells, saved by position", 40, 40, 40, 35, 100, 5, 7.692]);
    // ISG-high (30..39) inside Monocytes: d1 30..34 = 5 of Monocytes' 35 d1 cells; d2 35..39 = all 5
    expect(rows[2]).toEqual(["ISG-high", "Monocytes", "10 selected cells, saved by position", 10, 25, 10, 5, 14.286, 5, 100]);
  });

  it("exports without per-sample columns when the dataset has no sample column (paul15)", async () => {
    useDatasetStore.setState({
      currentDataset: kang({ id: "paul15", name: "paul15", obs_columns: [{ name: "paul15_clusters", dtype: "category", n_unique: 19 }] }),
      currentDatasetId: "paul15",
    });
    await buildTree();
    // Nothing looks like a donor or sample column, so none is picked (the user may still pick one).
    expect(screen.getByRole("combobox", { name: "Per-sample columns in the CSV" })).toHaveValue("");
    fireEvent.click(screen.getByRole("button", { name: /Export counts/ }));
    await waitFor(() => expect(api.downloadCsv).toHaveBeenCalled());
    expect(api.obsColumnValues).not.toHaveBeenCalled();
    expect(api.downloadCsv.mock.calls[0]![1]).toEqual(["gate", "parent", "definition", "n", "pct_parent", "pct_total"]);
  });

  it("says why stale gates are not applied and can discard them", () => {
    useGatesStore.getState().addGate(kang(), {
      name: "Monocytes",
      parentId: null,
      def: { kind: "cells", selection: cellsSel(range(0, 40), 100) },
    });
    useDatasetStore.setState({ currentDataset: kang({ n_cells: 90, cells_fingerprint: "fp-filtered" }) });
    select([1, 2]);
    renderControl();
    fireEvent.click(screen.getByRole("button", { name: /Gates \(1\)/ }));
    expect(screen.getByRole("alert")).toHaveTextContent(
      "1 saved gate is not applied. They were drawn on 100 cells and the dataset now has 90 (it was filtered since)",
    );
    expect(screen.queryByTestId("gate-row-Monocytes")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Save selection as gate/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Discard them" }));
    expect(useGatesStore.getState().records.kang).toBeUndefined();
    expect(screen.getByRole("button", { name: /Gates \(0\)/ })).toBeInTheDocument();
  });
});
