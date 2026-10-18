import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { ReactNode } from "react";
import type { DatasetInfo } from "@/api/types";
import { pearson, spearmanOf } from "@/lib/geneCoexpression";
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

const CELL_TYPE = { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] };
const STIM = { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] };
const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 8,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [CELL_TYPE, STIM],
  created_at: "",
};
// Eight cells: both in 0-2, A only in 3, B only in 4-5, neither in 6-7.
const A = Float32Array.from([1, 2, 3, 4, 0, 0, 0, 0]);
const B = Float32Array.from([2, 1, 4, 0, 5, 1, 0, 0]);
const CODES: Record<string, Int32Array> = {
  cell_type: Int32Array.from([0, 0, 1, 1, 1, 2, 2, 2]),
  stim: Int32Array.from([0, 0, 0, 0, 1, 1, 1, 1]),
};

const withQuery = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{ui}</QueryClientProvider>
);
const renderIt = () =>
  render(withQuery(<GeneVsGene datasetId="ds1" geneA="CD3D" geneB="GNLY" groupColumn="cell_type" />));

beforeEach(() => {
  plots.length = 0;
  fetchGeneExpression.mockReset();
  fetchGeneExpression.mockResolvedValue({ CD3D: A, GNLY: B });
  fetchEmbeddingBinary.mockImplementation(async (_ds: string, _emb: string, column: string) =>
    tableToIPC(
      tableFromArrays({ x: new Float32Array(8), y: new Float32Array(8), [column]: CODES[column]! }),
    ).slice().buffer,
  );
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSettingsStore.setState({ expressionLayer: "X", embedding: "X_umap" });
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
});

describe("GeneVsGene", () => {
  it("plots A against B per cell, one trace per group, with both correlations and the quadrant counts", async () => {
    renderIt();
    await screen.findByTestId("plot");
    const traces = () => plots.at(-1)!.data as { name: string; x: number[]; y: number[] }[];
    await vi.waitFor(() => expect(traces().map((t) => t.name)).toEqual(["B", "NK", "T"]));
    expect(traces()[1]!.x).toEqual([3, 4, 0]);
    const stats = screen.getByTestId("gene-vs-gene-stats").textContent!;
    expect(stats).toContain(`Pearson r ${pearson(A, B)!.toFixed(2)}`);
    expect(stats).toContain(`Spearman ρ ${spearmanOf(A, B)!.toFixed(2)}`);
    const quadrants = screen.getByLabelText("Quadrant counts").textContent!;
    expect(quadrants).toContain("GNLY only2 25.0%");
    expect(quadrants).toContain("both3 37.5%");
    expect(quadrants).toContain("neither2 25.0%");
    expect(quadrants).toContain("CD3D only1 12.5%");
  });

  it("selects the co-expressing cells, or a quadrant's, as a lasso would", async () => {
    useSelectionStore.setState({ highlightedGroup: { column: "cell_type", value: "B" } });
    renderIt();
    fireEvent.click(await screen.findByRole("button", { name: /Select co-expressing cells \(3\)/ }));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([0, 1, 2]);
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
    fireEvent.click(screen.getByTitle(/Select the 2 GNLY only cells/));
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([4, 5]);
  });

  it("adds the correlations within a highlighted group, of any column", async () => {
    useSelectionStore.setState({ highlightedGroup: { column: "stim", value: "CTRL" } });
    renderIt();
    const stats = await screen.findByTestId("gene-vs-gene-stats");
    await vi.waitFor(() => expect(stats.textContent).toContain("In CTRL"));
    const ctrl = [0, 1, 2, 3];
    expect(stats.textContent).toContain(`r ${pearson(A, B, ctrl)!.toFixed(2)}, ρ ${spearmanOf(A, B, ctrl)!.toFixed(2)}`);
    expect(stats.textContent).toContain("both expressed in 75.0%");
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("ds1", "X_umap", "stim");
  });

  it("says which gene the layer lacks", async () => {
    fetchGeneExpression.mockResolvedValue({ CD3D: A });
    renderIt();
    expect(await screen.findByText("GNLY not found in this expression layer.")).toBeInTheDocument();
  });
});
