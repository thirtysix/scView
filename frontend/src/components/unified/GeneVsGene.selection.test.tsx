import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";

const { fetchGeneExpression, fetchEmbeddingBinary, plots } = vi.hoisted(() => ({
  fetchGeneExpression: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  plots: [] as Record<string, unknown>[],
}));
vi.mock("@/api/expression", () => ({ fetchGeneExpression }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plots.push(props);
    return <div data-testid="plot" />;
  },
}));

import { GeneVsGene } from "./GeneVsGene";

const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 8,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] }],
  created_at: "",
};
const A = Float32Array.from([1, 2, 3, 4, 0, 0, 0, 0]);
const B = Float32Array.from([2, 1, 4, 0, 5, 1, 0, 0]);

type Trace = { name: string; x: number[]; y: number[]; marker: { opacity: number } };
const last = () => plots.at(-1)! as { data: Trace[]; onSelected: (e: unknown) => void };

beforeEach(() => {
  plots.length = 0;
  fetchGeneExpression.mockReset().mockResolvedValue({ CD3D: A, GNLY: B });
  fetchEmbeddingBinary.mockImplementation(async () =>
    tableToIPC(
      tableFromArrays({ x: new Float32Array(8), y: new Float32Array(8), cell_type: Int32Array.from([0, 0, 1, 1, 1, 2, 2, 2]) }),
    ).slice().buffer,
  );
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSettingsStore.setState({ expressionLayer: "X", embedding: "X_umap" });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
});

const renderIt = () =>
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <GeneVsGene datasetId="ds1" geneA="CD3D" geneB="GNLY" groupColumn="cell_type" />
    </QueryClientProvider>,
  );

describe("GeneVsGene linked selection (C035)", () => {
  it("draws the selected cells on top of the faded groups", async () => {
    useSelectionStore.setState({ selectedCellIndices: new Set([2, 4]) });
    renderIt();
    await vi.waitFor(() => expect(last().data.map((t) => t.name)).toEqual(["B", "NK", "T", "Selection (2 cells)"]));
    const sel = last().data[3]!;
    expect(sel.x).toEqual([3, 0]);
    expect(sel.y).toEqual([4, 5]);
    expect(last().data[0]!.marker.opacity).toBeLessThan(0.5);
    expect(screen.getByTestId("selection-chip")).toHaveTextContent("Selection: 2 cells");
  });

  it("a box drawn on the plot selects every cell inside it", async () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cell_type", value: "B" } });
    renderIt();
    await vi.waitFor(() => expect(last().data).toHaveLength(3));
    // CD3D 0.5-3.5, GNLY 0.5-4.5: cells 0, 1 and 2.
    act(() => last().onSelected({ range: { x: [0.5, 3.5], y: [0.5, 4.5] } }));
    expect([...useSelectionStore.getState().selectedCellIndices!].sort()).toEqual([0, 1, 2]);
    expect(useSelectionStore.getState().highlightedGroup).toBeNull(); // a set of cells, not a group
    await vi.waitFor(() => expect(last().data.at(-1)!.name).toBe("Selection (3 cells)"));
  });
});
