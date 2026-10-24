import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { MarkersResponse } from "@/api/markers";
import { useDatasetStore } from "@/stores/datasetStore";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({ MarkerSpecificityPlot: () => null }));

import { UnifiedMarkersSubtab } from "./UnifiedMarkersSubtab";

const DATASET: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 3, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "",
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["CD14 Mono", "B"] }],
};

const marker = (gene: string, lfc: number) => ({
  gene, group: "CD14 Mono", logfoldchange: lfc, pval: 1e-9, pval_adj: 1e-7,
});

const MARKERS: MarkersResponse = {
  groups: ["CD14 Mono", "B"],
  markers: { "CD14 Mono": [marker("LYZ", 5.1), marker("CD14", 4.2)], B: [] },
};

function ambient(approximate: boolean) {
  return {
    markers: {
      column: "cell_type",
      approximate,
      groups: { "CD14 Mono": [{ gene: "LYZ", soup_rank: 2, ratio: approximate ? null : 1.3 }] },
      n_flagged: 1,
    },
  };
}

const renderTab = () =>
  render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="cell_type" setGroupByColumn={vi.fn()} />);

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  apiFetch.mockReset();
});

describe("UnifiedMarkersSubtab ambient flag", () => {
  it("marks a marker whose signal outside its cluster looks ambient, with why in the tooltip", async () => {
    apiFetch.mockImplementation(async (url: string) => (url.includes("/qc/ambient") ? ambient(false) : MARKERS));
    renderTab();
    const lyz = (await screen.findByText("LYZ")).closest("tr")!;
    const flag = await within(lyz).findByTestId("ambient-marker-flag");
    expect(flag).toHaveTextContent("ambient");
    expect(flag.getAttribute("title")).toContain("1.3x what ambient RNA alone would put there");
    expect(within(screen.getByText("CD14").closest("tr")!).queryByTestId("ambient-marker-flag")).toBeNull();
    expect(apiFetch).toHaveBeenCalledWith("/datasets/kang/qc/ambient?column=cell_type");
  });

  it("an approximate flag (no raw matrix) says so", async () => {
    apiFetch.mockImplementation(async (url: string) => (url.includes("/qc/ambient") ? ambient(true) : MARKERS));
    renderTab();
    const flag = await screen.findByTestId("ambient-marker-flag");
    expect(flag).toHaveTextContent("ambient?");
    expect(flag.getAttribute("title")).toContain("Possibly ambient");
  });

  it("no flags when the estimate fails", async () => {
    apiFetch.mockImplementation(async (url: string) => {
      if (url.includes("/qc/ambient")) throw new Error("not available for large datasets yet");
      return MARKERS;
    });
    renderTab();
    await screen.findByText("LYZ");
    await waitFor(() => expect(apiFetch).toHaveBeenCalledTimes(2));
    expect(screen.queryByTestId("ambient-marker-flag")).toBeNull();
  });
});
