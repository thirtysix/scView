import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerPlot", () => ({
  MarkerPlot: ({ kind }: { kind: string }) => <div data-testid="marker-plot">{kind}</div>,
}));
vi.mock("@/components/plots/CellHeatmap", () => ({
  CellHeatmap: ({ groupby }: { groupby: string }) => <div data-testid="cell-heatmap">{groupby}</div>,
}));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({ MarkerSpecificityPlot: () => null }));

import { MarkerGenesPanel } from "./MarkerGenesPanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

const DATASET = {
  id: "d1", name: "d1", filename: "d1.h5ad", n_cells: 10, n_genes: 5, status: "ready",
  available_embeddings: [], created_at: "",
  obs_columns: [{ name: "leiden", dtype: "category", n_unique: 2 }],
} as unknown as DatasetInfo;

const MARKERS = {
  groups: ["0", "1"],
  markers: {
    "0": [{ gene: "CD3E", group: "0", logfoldchange: 2, pval: 1e-5, pval_adj: 1e-4 }],
    "1": [{ gene: "LYZ", group: "1", logfoldchange: 3, pval: 1e-6, pval_adj: 1e-5 }],
  },
};

describe("MarkerGenesPanel", () => {
  beforeEach(() => {
    apiFetch.mockReset();
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
  });

  it("shows a column's markers when the first, column-less request fails late", async () => {
    // Only per-column markers exist. The request made before the default column
    // was picked 404s after the column's own markers arrived; it used to win.
    let bareCalls = 0;
    apiFetch.mockImplementation((url: string) => {
      if (url.includes("groupby_column=")) return Promise.resolve(MARKERS);
      bareCalls += 1;
      const delay = bareCalls === 1 ? 50 : 0;
      return new Promise((_, reject) => setTimeout(() => reject(new Error("404: no markers")), delay));
    });
    render(<MarkerGenesPanel />);
    await waitFor(() => screen.getByText("CD3E"));
    await new Promise((r) => setTimeout(r, 100));
    expect(screen.queryByText("No marker genes available")).toBeNull();
    expect(screen.getByText("CD3E")).toBeTruthy();
  });

  it("switches the heatmap between group means and single cells", async () => {
    apiFetch.mockResolvedValue(MARKERS);
    useViewStore.setState({ markerView: "heatmap", markerGenes: [] });
    render(<MarkerGenesPanel />);
    expect(await screen.findByTestId("marker-plot")).toHaveTextContent("heatmap");
    fireEvent.click(screen.getByRole("button", { name: "Per cell" }));
    expect(screen.getByTestId("cell-heatmap")).toHaveTextContent("leiden");
    expect(screen.queryByTestId("marker-plot")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Group means" }));
    expect(screen.getByTestId("marker-plot")).toBeTruthy();
    // the dot plot has no per-cell form
    fireEvent.click(screen.getByRole("button", { name: "Per cell" }));
    fireEvent.click(screen.getByRole("button", { name: "Dot plot" }));
    expect(screen.getByTestId("marker-plot")).toHaveTextContent("dotplot");
    expect(screen.queryByRole("button", { name: "Per cell" })).toBeNull();
    useViewStore.setState({ markerView: "table" });
  });
});

describe("MarkerGenesPanel ambient flag", () => {
  beforeEach(() => {
    apiFetch.mockReset();
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: DATASET });
    useViewStore.setState({ markerView: "table" });
  });

  it("marks a marker that looks ambient, as the Unified View table does", async () => {
    apiFetch.mockImplementation(async (url: string) =>
      url.includes("/qc/ambient")
        ? { markers: { column: "leiden", approximate: false, n_flagged: 1,
            groups: { "0": [{ gene: "CD3E", soup_rank: 3, ratio: 1.1 }] } } }
        : MARKERS,
    );
    render(<MarkerGenesPanel />);
    const flag = await screen.findByTestId("ambient-marker-flag");
    expect(flag).toHaveTextContent("ambient");
    expect(flag.closest("tr")).toHaveTextContent("CD3E");
  });
});
