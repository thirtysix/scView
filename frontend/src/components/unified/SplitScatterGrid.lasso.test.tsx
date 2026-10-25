import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { DatasetInfo } from "@/api/types";
import { splitGridLayout } from "@/lib/splitView";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

const { fetchEmbeddingBinary } = vi.hoisted(() => ({ fetchEmbeddingBinary: vi.fn() }));
vi.mock("@deck.gl/react", async () => {
  const { forwardRef } = await import("react");
  return {
    DeckGL: forwardRef<HTMLDivElement, Record<string, unknown>>((_props, ref) => <div ref={ref} data-testid="deck" />),
  };
});
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));

import { SplitScatterGrid } from "./SplitScatterGrid";

const W = 800;
const H = 400;
class FakeResizeObserver {
  constructor(private cb: ResizeObserverCallback) {}
  observe() {
    this.cb([{ contentRect: { width: W, height: H } } as ResizeObserverEntry], this as never);
  }
  disconnect() {}
  unobserve() {}
}

const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 6,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] }],
  created_at: "",
};

// Six cells on the diagonal (cell c at x = y = c): CTRL = 0, 2, 4; STIM = 1, 3, 5.
const positions = Float32Array.from([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);

beforeEach(() => {
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSplitViewStore.setState({ splitBy: "stim", showContext: true, lassoScope: "panel" });
  useSelectionStore.setState({ selectedCellIndices: null, selectionMode: "lasso", highlightedGroup: null });
  const table = tableFromArrays({
    x: Float32Array.from([0, 1, 2, 3, 4, 5]),
    y: Float32Array.from([0, 1, 2, 3, 4, 5]),
    stim: Int32Array.from([0, 1, 0, 1, 0, 1]),
  });
  fetchEmbeddingBinary.mockResolvedValue(tableToIPC(table).slice().buffer);
});
afterEach(() => vi.unstubAllGlobals());

function renderGrid(selected: Set<number> | null = null) {
  // What UnifiedViewPanel does with a closed lasso.
  const onLassoComplete = vi.fn((cells: Set<number>) => useSelectionStore.getState().setSelection(cells));
  const ui = (sel: Set<number> | null) => (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <SplitScatterGrid
        datasetId="ds1"
        embedding="X_umap"
        column="stim"
        positions={positions}
        dimensions={2}
        colorValues={null}
        colorType="categorical"
        selectedIndices={sel}
        onLassoComplete={onLassoComplete}
      />
    </QueryClientProvider>
  );
  const r = render(ui(selected));
  return { ...r, onLassoComplete, rerenderWith: (sel: Set<number> | null) => r.rerender(ui(sel)) };
}

/** Drag a rectangle over the left half of panel `p`'s view (data x below the middle). */
function lassoLeftHalf(p: number) {
  const v = splitGridLayout(2, W, H, 1).cells[p]!.view;
  const svg = screen.getByTestId("grid-lasso");
  const pts = [
    [v.x + 1, v.y + 1],
    [v.x + v.width / 2, v.y + 1],
    [v.x + v.width / 2, v.y + v.height - 1],
    [v.x + 1, v.y + v.height - 1],
  ];
  fireEvent.mouseDown(svg, { clientX: pts[0]![0], clientY: pts[0]![1] });
  for (const [x, y] of pts.slice(1)) fireEvent.mouseMove(svg, { clientX: x, clientY: y });
  fireEvent.mouseUp(svg);
}

describe("SplitScatterGrid lasso (C039)", () => {
  it("selects the panel's own cells under the outline", async () => {
    const { onLassoComplete } = renderGrid();
    await screen.findByTestId("deck");
    lassoLeftHalf(1); // STIM panel: cells 0, 1, 2 are left of the middle; STIM's is 1
    expect(onLassoComplete).toHaveBeenCalledTimes(1);
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([1]);
  });

  it("selects every panel's cells under the outline when asked to", async () => {
    renderGrid();
    await screen.findByTestId("deck");
    fireEvent.change(screen.getByLabelText("Lasso picks"), { target: { value: "all" } });
    lassoLeftHalf(1);
    expect([...useSelectionStore.getState().selectedCellIndices!].sort()).toEqual([0, 1, 2]);
  });

  it("is off when the lasso is", async () => {
    useSelectionStore.setState({ selectionMode: "none" });
    renderGrid();
    await screen.findByTestId("deck");
    expect(screen.queryByTestId("grid-lasso")).toBeNull();
    expect(screen.queryByLabelText("Lasso picks")).toBeNull();
  });

  it("says in each panel's title how many of its cells are selected, and shows the chip", async () => {
    useSelectionStore.setState({ selectionMode: "none", selectedCellIndices: new Set([0, 1, 2]) });
    renderGrid(new Set([0, 1, 2]));
    await screen.findByTestId("deck");
    expect(screen.getByText("2 selected of 3")).toBeInTheDocument(); // CTRL: 0, 2
    expect(screen.getByText("1 selected of 3")).toBeInTheDocument(); // STIM: 1
    expect(screen.getByTestId("selection-chip")).toHaveTextContent("Selection: 3 cells");
  });

  it("shows a gate made the selection like any other", async () => {
    // A gate applied to the plot sets the store's selection (gatesStore.applyGate).
    useSelectionStore.setState({ selectionMode: "none" });
    const { rerenderWith } = renderGrid();
    await screen.findByTestId("deck");
    expect(screen.getAllByText("3 cells")).toHaveLength(2);
    const gate = new Set([3, 5]);
    act(() => useSelectionStore.getState().setSelection(gate));
    rerenderWith(gate);
    expect(screen.getByText("0 selected of 3")).toBeInTheDocument();
    expect(screen.getByText("2 selected of 3")).toBeInTheDocument();
  });
});
