import { describe, it, expect, beforeEach } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { DatasetInfo } from "@/api/types";
import { UnifiedToolbar } from "./UnifiedToolbar";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const dataset: DatasetInfo = {
  id: "ds",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 12,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] }],
  expression_layers: [{ key: "raw", label: "Log-normalized (raw)", n_genes: 10 }],
  default_expression_layer: "raw",
  created_at: "",
};

/** The toolbar's row items as the browser wraps them: each one's tag and classes. jsdom
 *  has no layout, so "never re-wraps" is checked as "the items and their sizes' classes
 *  never change". */
const rowItems = (toolbar: HTMLElement) => [...toolbar.children].map((c) => `${c.tagName}.${c.className}`);

describe("UnifiedToolbar layout", () => {
  beforeEach(() => {
    useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds" });
    useSettingsStore.setState({ embedding: "X_umap", colorBy: "cell_type", expressionLayer: "raw" });
    useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, selectionMode: "none" });
  });

  it("keeps the same row items when the lasso turns on and a selection appears or clears", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} onResetView={() => {}} />
      </QueryClientProvider>,
    );
    const lasso = screen.getByRole("button", { name: "Lasso selection" });
    const toolbar = lasso.closest(".flex-wrap") as HTMLElement;
    const before = rowItems(toolbar);

    fireEvent.click(lasso);
    expect(lasso).toHaveAttribute("aria-pressed", "true");
    expect(rowItems(toolbar)).toEqual(before);

    act(() => useSelectionStore.getState().setSelection(new Set([1, 2, 3])));
    const slot = screen.getByTestId("selection-slot");
    expect(within(slot).getByText("3 selected")).toBeInTheDocument();
    expect(within(slot).getByRole("button", { name: "Clear selection" })).toBeInTheDocument();
    expect(within(slot).getByRole("button", { name: /Label selection/ })).toBeInTheDocument();
    expect(rowItems(toolbar)).toEqual(before);

    fireEvent.click(within(slot).getByRole("button", { name: "Clear selection" }));
    expect(within(slot).queryByText(/selected/)).toBeNull();
    expect(rowItems(toolbar)).toEqual(before);
  });

  it("reserves the selection's room in the spacer: a fixed basis with a minimum width", () => {
    render(
      <QueryClientProvider client={new QueryClient()}>
        <UnifiedToolbar numCells={12} />
      </QueryClientProvider>,
    );
    const slot = screen.getByTestId("selection-slot");
    // flex-1 = a basis of 0, so the slot's content never decides where the row breaks;
    // the minimum keeps room for "n selected", Ask, clear and Label selection
    expect(slot.className).toMatch(/\bflex-1\b/);
    expect(slot.className).toMatch(/\bmin-w-\[20rem\]/);
    expect(slot.className).not.toMatch(/\bmin-w-0\b/);
  });
});
