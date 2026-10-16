import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerPlot", () => ({ MarkerPlot: () => null }));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({ MarkerSpecificityPlot: () => null }));

import { MarkerGenesPanel } from "./MarkerGenesPanel";
import { useDatasetStore } from "@/stores/datasetStore";

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
});
