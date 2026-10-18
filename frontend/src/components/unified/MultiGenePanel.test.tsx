import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { tableFromArrays, tableToIPC } from "apache-arrow";
import type { ReactNode } from "react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useGeneGridStore } from "@/stores/geneGridStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";

const { fetchGeneExpression, fetchEmbeddingBinary, plots } = vi.hoisted(() => ({
  fetchGeneExpression: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  plots: [] as Record<string, unknown>[],
}));
vi.mock("@/api/expression", () => ({ fetchGeneExpression }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary }));
vi.mock("@/api/client", () => ({ apiFetch: vi.fn().mockResolvedValue({ query: "", results: [] }) }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: Record<string, unknown>) => {
    plots.push(props);
    return <div data-testid="plot" />;
  },
}));

import { MultiGenePanel } from "./MultiGenePanel";

const CELL_TYPE = { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "NK", "T"] };
const dataset: DatasetInfo = {
  id: "ds1",
  name: "kang",
  filename: "kang.h5ad",
  n_cells: 6,
  n_genes: 10,
  status: "ready",
  available_embeddings: ["X_umap"],
  obs_columns: [CELL_TYPE],
  created_at: "",
};
const EXPR: Record<string, Float32Array> = {
  CD3D: Float32Array.from([0, 0, 1, 1, 3, 4]),
  MS4A1: Float32Array.from([5, 6, 0, 0, 0, 0]),
  GNLY: Float32Array.from([0, 0, 7, 8, 0, 1]),
};

const withQuery = (ui: ReactNode) => (
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{ui}</QueryClientProvider>
);
const paste = (text: string) =>
  fireEvent.paste(screen.getByPlaceholderText(/Add genes/), { clipboardData: { getData: () => text } });

beforeEach(() => {
  plots.length = 0;
  fetchGeneExpression.mockReset();
  fetchGeneExpression.mockImplementation(async (_ds: string, genes: string[]) =>
    Object.fromEntries(genes.filter((g) => g in EXPR).map((g) => [g, EXPR[g]!])),
  );
  fetchEmbeddingBinary.mockImplementation(async (_ds: string, _emb: string, column: string) =>
    tableToIPC(
      tableFromArrays({
        x: Float32Array.from([0, 1, 2, 3, 4, 5]),
        y: Float32Array.from([0, 1, 2, 3, 4, 5]),
        [column]: Int32Array.from([0, 0, 1, 1, 2, 2]),
      }),
    ).slice().buffer,
  );
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
  useSettingsStore.setState({ expressionLayer: "X", embedding: "X_umap" });
  useGeneGridStore.setState({ datasetId: null, genes: [], pair: null, showViolins: false, showScatter: false });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, geneGrid: null });
});

describe("MultiGenePanel", () => {
  it("takes a pasted list, at most nine genes, and removes one by its chip", () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    expect(screen.getByRole("button", { name: /Feature grid/ })).toBeDisabled();
    paste("CD3D, MS4A1\nGNLY CD3D");
    expect(useGeneGridStore.getState().genes).toEqual(["CD3D", "MS4A1", "GNLY"]);
    paste(Array.from({ length: 12 }, (_, i) => `G${i}`).join(" "));
    expect(useGeneGridStore.getState().genes).toHaveLength(9);
    fireEvent.click(screen.getByRole("button", { name: "Remove MS4A1" }));
    expect(useGeneGridStore.getState().genes).not.toContain("MS4A1");
    expect(useGeneGridStore.getState().genes).toHaveLength(8);
  });

  it("opens the feature grid on the picks, and an open grid follows the picks", () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    paste("CD3D MS4A1 GNLY");
    fireEvent.click(screen.getByRole("button", { name: /Feature grid/ }));
    expect(useSplitViewStore.getState().geneGrid).toEqual({
      datasetId: "ds1",
      kind: "features",
      genes: ["CD3D", "MS4A1", "GNLY"],
    });
    fireEvent.click(screen.getByRole("button", { name: "Remove GNLY" }));
    expect(useSplitViewStore.getState().geneGrid?.genes).toEqual(["CD3D", "MS4A1"]);
    fireEvent.click(screen.getByRole("button", { name: "Remove MS4A1" }));
    expect(useSplitViewStore.getState().geneGrid).toBeNull();
  });

  it("does not show picks made on another dataset", () => {
    useGeneGridStore.setState({ datasetId: "other", genes: ["CD3D", "MS4A1"] });
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    expect(screen.queryByRole("button", { name: "Remove CD3D" })).toBeNull();
  });

  it("draws a stacked violin: a row per gene, a violin per group, from one expression request", async () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    paste("CD3D MS4A1 GNLY");
    fireEvent.click(screen.getByRole("button", { name: /Stacked violin/ }));
    await screen.findByTestId("plot");
    expect(fetchGeneExpression).toHaveBeenCalledTimes(1);
    expect(fetchEmbeddingBinary).toHaveBeenCalledWith("ds1", "X_umap", "cell_type");
    const props = plots.at(-1)!;
    const traces = props.data as { name: string; yaxis: string; y: number[] }[];
    expect(traces).toHaveLength(9);
    expect(traces.map((t) => `${t.yaxis}:${t.name}`).slice(0, 4)).toEqual(["y:B", "y:NK", "y:T", "y2:B"]);
    // CD3D in NK = cells 2, 3; MS4A1 in B = cells 0, 1.
    expect(traces[1]!.y).toEqual([1, 1]);
    expect(traces[3]!.y).toEqual([5, 6]);
    const layout = props.layout as Record<string, { title?: { text: string }; anchor?: string }>;
    expect([layout.yaxis!.title!.text, layout.yaxis2!.title!.text, layout.yaxis3!.title!.text]).toEqual([
      "CD3D",
      "MS4A1",
      "GNLY",
    ]);
    expect(layout.xaxis!.anchor).toBe("y3");
  });

  it("fetches only a newly added gene when the others are cached", async () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    paste("CD3D MS4A1");
    fireEvent.click(screen.getByRole("button", { name: /Stacked violin/ }));
    await screen.findByTestId("plot");
    paste("GNLY,"); // a separator: one name alone goes to the autocomplete
    await waitFor(() => expect(fetchGeneExpression).toHaveBeenCalledTimes(2));
    expect(fetchGeneExpression.mock.calls[1]![1]).toEqual(["GNLY"]);
    await waitFor(() => expect((plots.at(-1)!.data as unknown[]).length).toBe(9));
  });

  it("blends the chosen pair on the embedding, and the blend follows the pair", () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    paste("CD3D MS4A1 GNLY");
    expect((screen.getByRole("combobox", { name: "Gene A" }) as HTMLSelectElement).value).toBe("CD3D");
    fireEvent.click(screen.getByRole("button", { name: /Blend/ }));
    expect(useSplitViewStore.getState().geneGrid).toEqual({ datasetId: "ds1", kind: "blend", genes: ["CD3D", "MS4A1"] });
    fireEvent.change(screen.getByRole("combobox", { name: "Gene B" }), { target: { value: "GNLY" } });
    expect(useSplitViewStore.getState().geneGrid?.genes).toEqual(["CD3D", "GNLY"]);
    // A gene list without one of the pair closes the blend.
    fireEvent.click(screen.getByRole("button", { name: "Remove GNLY" }));
    expect(useSplitViewStore.getState().geneGrid).toBeNull();
    expect(useGeneGridStore.getState().pair).toEqual(["CD3D", "MS4A1"]);
  });

  it("opens gene vs gene for the pair", async () => {
    render(withQuery(<MultiGenePanel datasetId="ds1" groupByColumn="cell_type" />));
    paste("CD3D MS4A1");
    fireEvent.click(screen.getByRole("button", { name: /Gene vs gene/ }));
    expect(await screen.findByText("CD3D vs MS4A1")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /Select co-expressing cells \(0\)/ })).toBeDisabled();
  });
});
