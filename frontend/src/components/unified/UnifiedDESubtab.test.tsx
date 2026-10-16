import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { DE_DEFAULTS, useDESettingsStore } from "@/stores/deSettingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { UnifiedDESubtab } from "./UnifiedDESubtab";

const ds = (id = "ds"): DatasetInfo => ({
  id,
  name: id,
  filename: `${id}.h5ad`,
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] },
    { name: "ind", dtype: "category", n_unique: 3, values: ["d1", "d2", "d3"] },
  ],
  created_at: "",
});

const noop = () => {};
const renderDE = () => render(<UnifiedDESubtab onGeneClick={noop} onScoreGeneSet={noop} />);
const selects = () => screen.getAllByRole("combobox") as HTMLSelectElement[];

beforeEach(() => {
  useDESettingsStore.setState({ ...DE_DEFAULTS, scopedHighlight: null });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useDatasetStore.getState().setCurrentDataset(ds());
});

describe("UnifiedDESubtab settings", () => {
  it("keep their values across a tab switch (a remount)", () => {
    const first = renderDE();
    fireEvent.click(screen.getByRole("button", { name: "Between groups" }));
    // Compare stim, CTRL vs STIM, within cell_type = T
    fireEvent.change(selects()[1]!, { target: { value: "CTRL" } });
    fireEvent.change(selects()[2]!, { target: { value: "STIM" } });
    fireEvent.change(selects()[3]!, { target: { value: "cell_type" } });
    fireEvent.change(selects()[4]!, { target: { value: "T" } });
    first.unmount();

    renderDE();
    expect(screen.getByRole("button", { name: "Between groups" })).toHaveClass("bg-primary");
    expect(selects().map((s) => s.value).slice(0, 5)).toEqual(["stim", "CTRL", "STIM", "cell_type", "T"]);
  });

  it("take a highlighted group as the scope once, not over a scope picked since", () => {
    useDESettingsStore.getState().update({ mode: "groups" });
    const first = renderDE();
    act(() => useSelectionStore.getState().setHighlight({ column: "cell_type", value: "NK" }));
    expect(selects()[4]!.value).toBe("NK");
    fireEvent.change(selects()[4]!, { target: { value: "B" } });
    first.unmount();
    renderDE();
    expect(selects()[4]!.value).toBe("B");
  });

  it("start from the defaults on another dataset", () => {
    useDESettingsStore.getState().update({ mode: "groups", splitBy: "ind", fcThreshold: 3 });
    act(() => useDatasetStore.getState().setCurrentDataset(ds("other")));
    renderDE();
    expect(screen.getByRole("button", { name: "Selection vs rest" })).toHaveClass("bg-primary");
    expect(useDESettingsStore.getState().fcThreshold).toBe(1);
  });
});
