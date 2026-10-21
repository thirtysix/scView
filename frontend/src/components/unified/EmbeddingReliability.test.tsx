import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

const api = vi.hoisted(() => ({
  getEmbeddingReliability: vi.fn(),
  getReliabilitySummary: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  decodeArrowBuffer: vi.fn(),
}));
vi.mock("@/api/reliability", () => ({
  getEmbeddingReliability: api.getEmbeddingReliability,
  getReliabilitySummary: api.getReliabilitySummary,
}));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary: api.fetchEmbeddingBinary }));
vi.mock("@/lib/arrow", () => ({ decodeArrowBuffer: api.decodeArrowBuffer }));

import type { ReliabilityResponse, ReliabilitySummary } from "@/api/reliability";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { EmbeddingReliabilityControl } from "./EmbeddingReliability";

const DATASET: DatasetInfo = {
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 5,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap", "X_pca"],
  embedding_dimensions: { X_umap: 2, X_pca: 10 },
  obs_columns: [
    { name: "cluster", dtype: "category", n_unique: 3, values: ["B", "CD4 T", "pDC"] },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "n_genes", dtype: "int64", n_unique: 5 },
  ],
  created_at: "",
};

const summary = (over: Partial<ReliabilitySummary> = {}): ReliabilitySummary => ({
  embedding: "X_umap",
  dims: [0, 1],
  n_dims: 2,
  k: 15,
  graph: { source: "stored", key: "neighbors", k: 15, use_rep: "X_pca_harmony", n_pcs: null, metric: "euclidean" },
  n_cells: 5,
  n_scored: 5,
  sampled: false,
  seed: 0,
  chance: 0.0011,
  low_threshold: 0.2,
  low_reason: "below 0.2: fewer than 1 in 5 of the cell's graph neighbours are among its nearest cells on the plot",
  notes: ["A note from the server."],
  method: "Per cell: the share ...",
  seconds: 0.1,
  overall: { n: 5, median: 0.25, q1: 0, q3: 0.5, mean: 0.3, share_low: 0.4 },
  shuffled: { median: 0, mean: 0.001, seed: 1 },
  group_by: "cluster",
  group_reason: "the active clustering",
  // Worst first, as the server sends them.
  groups: [
    { group: "CD4 T", n: 2, median: 0, q1: 0, q3: 0, mean: 0, share_low: 1 },
    { group: "B", n: 2, median: 0.25, q1: 0.25, q3: 0.25, mean: 0.25, share_low: 0 },
    { group: "pDC", n: 1, median: 1, q1: 1, q3: 1, mean: 1, share_low: 0 },
  ],
  cached: false,
  ...over,
});

const response = (over: Partial<ReliabilitySummary> = {}): ReliabilityResponse => ({
  ...summary(over),
  scores: [0.25, 0, 0, 0.25, 1],
});

const button = () => screen.getByRole("button", { name: /Plot reliability/ });
const rowNames = () =>
  within(screen.getByRole("dialog", { name: "Plot reliability" }))
    .getAllByRole("row")
    .slice(1)
    .map((r) => r.querySelector("td")!.textContent);

beforeEach(() => {
  vi.clearAllMocks();
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  useSettingsStore.setState({ embedding: "X_umap", pcaDims: [0, 1] });
  useUnifiedViewStore.getState().clearOverlayState();
  useSelectionStore.setState({ selectedCellIndices: null, highlightedGroup: null });
  api.getEmbeddingReliability.mockImplementation(async (_d: string, embedding: string, q: { dims?: [number, number] }) =>
    response({ embedding, dims: q.dims ?? [0, 1] }),
  );
  api.getReliabilitySummary.mockImplementation(async (_d: string, embedding: string, q: { groupBy?: string }) =>
    summary({ embedding, group_by: q.groupBy ?? "cluster" }),
  );
  api.fetchEmbeddingBinary.mockResolvedValue(new ArrayBuffer(8));
  api.decodeArrowBuffer.mockReturnValue({ cluster: Int32Array.from([0, 1, 1, 0, 2]) });
});

describe("Plot reliability toggle", () => {
  it("colours the plot by the scores on a 0-1 overlay and opens the table, worst first", async () => {
    render(<EmbeddingReliabilityControl />);
    expect(button()).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(button());
    await waitFor(() => expect(screen.getByRole("dialog", { name: "Plot reliability" })).toBeTruthy());
    expect(api.getEmbeddingReliability).toHaveBeenCalledWith("kang", "X_umap", { dims: undefined, groupBy: undefined });
    const u = useUnifiedViewStore.getState();
    expect(u.scatterOverlay).toEqual({
      type: "score",
      name: "Neighbour preservation (UMAP)",
      reliability: { embedding: "X_umap", k: 15, chance: 0.0011, random: 0.001 },
    });
    expect(Array.from(u.overlayValues!)).toEqual([0.25, 0, 0, 0.25, 1]);
    expect(button()).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("reliability-caption")).toHaveTextContent(
      "Each cell: the share of its 15 graph neighbours (the kNN graph the clustering used)",
    );
    expect(screen.getByText("A note from the server.")).toBeTruthy();
    expect(rowNames()).toEqual(["CD4 T", "B", "pDC"]);
    expect(screen.getByText(/Worst first \(lowest median\)/)).toHaveTextContent("n = 3 groups.");
    // Every column says what it is.
    for (const name of ["Group", "Cells", "Median", "Low"]) {
      expect(screen.getByRole("button", { name }).closest("th")!.getAttribute("title")).toBeTruthy();
    }
  });

  it("sorts by a column and back to worst first", async () => {
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    fireEvent.click(screen.getByRole("button", { name: "Median" }));
    expect(rowNames()).toEqual(["CD4 T", "B", "pDC"]); // ascending
    fireEvent.click(screen.getByRole("button", { name: /Median/ }));
    expect(rowNames()).toEqual(["pDC", "B", "CD4 T"]);
    fireEvent.click(screen.getByRole("button", { name: /Group/ }));
    expect(rowNames()).toEqual(["B", "CD4 T", "pDC"]);
    fireEvent.click(screen.getByRole("button", { name: "Worst first" }));
    expect(rowNames()).toEqual(["CD4 T", "B", "pDC"]);
  });

  it("selects and highlights a group's cells on a row click, and a second click clears it", async () => {
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    fireEvent.click(screen.getByText("CD4 T"));
    await waitFor(() => expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: "cluster", value: "CD4 T" }));
    expect(api.fetchEmbeddingBinary).toHaveBeenCalledWith("kang", "X_umap", "cluster");
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([1, 2]);
    // The overlay stays: a highlight does not recolour the plot.
    expect(useUnifiedViewStore.getState().scatterOverlay?.type).toBe("score");
    expect(screen.getByText("CD4 T").closest("tr")).toHaveAttribute("aria-selected", "true");
    fireEvent.click(screen.getByText("pDC"));
    await waitFor(() => expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([4]));
    expect(api.fetchEmbeddingBinary).toHaveBeenCalledTimes(1); // the labels are kept
    fireEvent.click(screen.getByText("pDC"));
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });

  it("fetches again for another embedding or PCA pair while on", async () => {
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    act(() => useSettingsStore.setState({ embedding: "X_pca", pcaDims: [2, 3] }));
    await waitFor(() =>
      expect(api.getEmbeddingReliability).toHaveBeenLastCalledWith("kang", "X_pca", { dims: [2, 3], groupBy: undefined }),
    );
    await waitFor(() =>
      expect(useUnifiedViewStore.getState().overlayLabel).toBe("Neighbour preservation (PCA, PC3 vs PC4)"),
    );
    expect(useUnifiedViewStore.getState().scatterOverlay).toMatchObject({ reliability: { embedding: "X_pca", dims: [2, 3] } });
    await waitFor(() => expect(screen.getByText("Plot reliability: PCA, PC3 vs PC4")).toBeTruthy());
    expect(api.getEmbeddingReliability).toHaveBeenCalledTimes(2);
  });

  it("summarises by another column without refetching the scores", async () => {
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    fireEvent.change(screen.getByLabelText("By"), { target: { value: "stim" } });
    await waitFor(() =>
      expect(api.getReliabilitySummary).toHaveBeenCalledWith("kang", "X_umap", { dims: undefined, groupBy: "stim" }),
    );
    await waitFor(() => expect((screen.getByLabelText("By") as HTMLSelectElement).value).toBe("stim"));
    expect(api.getEmbeddingReliability).toHaveBeenCalledTimes(1);
  });

  it("turns off from the toolbar or the panel, clearing the overlay", async () => {
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    fireEvent.click(screen.getByRole("button", { name: "Turn plot reliability off" }));
    expect(useUnifiedViewStore.getState().scatterOverlay).toBeNull();
    expect(screen.queryByRole("dialog", { name: "Plot reliability" })).toBeNull();
    fireEvent.click(button());
    await waitFor(() => screen.getByRole("dialog", { name: "Plot reliability" }));
    fireEvent.click(button());
    expect(useUnifiedViewStore.getState().scatterOverlay).toBeNull();
  });

  it("leaves another score overlay alone when turned off, and does not count it as on", () => {
    useUnifiedViewStore.setState({ scatterOverlay: { type: "score", name: "IFN", genes: ["ISG15"] }, overlayDatasetId: "kang" });
    render(<EmbeddingReliabilityControl />);
    expect(button()).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("says why when the server cannot answer", async () => {
    api.getEmbeddingReliability.mockRejectedValue(
      new Error('API error 422: {"detail":"No neighbour graph is stored and there is no X_pca to build one from."}'),
    );
    render(<EmbeddingReliabilityControl />);
    fireEvent.click(button());
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Plot reliability: No neighbour graph is stored and there is no X_pca to build one from.",
      ),
    );
    expect(useUnifiedViewStore.getState().scatterOverlay).toBeNull();
  });

  it("fetches the table for an overlay restored from a session", async () => {
    useUnifiedViewStore.setState({
      scatterOverlay: {
        type: "score",
        name: "Neighbour preservation (UMAP)",
        reliability: { embedding: "X_umap", k: 15, chance: 0.0011, random: 0.001 },
      },
      overlayValues: Float32Array.from([0.25, 0, 0, 0.25, 1]),
      overlayDatasetId: "kang",
    });
    render(<EmbeddingReliabilityControl />);
    await waitFor(() => expect(rowNames()).toEqual(["CD4 T", "B", "pDC"]));
    expect(api.getReliabilitySummary).toHaveBeenCalledWith("kang", "X_umap", { dims: undefined, groupBy: undefined });
    expect(api.getEmbeddingReliability).not.toHaveBeenCalled();
  });

  it("is off without an embedding to score", () => {
    useDatasetStore.setState({ currentDataset: { ...DATASET, available_embeddings: [] } });
    render(<EmbeddingReliabilityControl />);
    expect(button()).toBeDisabled();
  });
});
