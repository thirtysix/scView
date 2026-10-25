import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import type { MarkersResponse } from "@/api/markers";
import type { SplitMarkersResult } from "@/api/splitMarkers";
import { useDatasetStore } from "@/stores/datasetStore";

const { apiFetch } = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/components/plots/MarkerSpecificityPlot", () => ({ MarkerSpecificityPlot: () => null }));

const RESULT = {
  groups: [{
    group: "pDC",
    genes: [
      { gene: "IGJ", split_pval_adj: 1e-6 },
      { gene: "CD74", split_pval_adj: 0.4 },
    ],
  }],
} as unknown as SplitMarkersResult;

// The check itself is tested in CountSplitCheck.test.tsx: here it only hands over a result.
vi.mock("@/components/unified/CountSplitCheck", () => ({
  CountSplitCheck: (p: { group: string; onResult: (r: SplitMarkersResult) => void; onShowSplitPChange: (on: boolean) => void }) => (
    <button
      type="button"
      onClick={() => {
        p.onResult(RESULT);
        p.onShowSplitPChange(true);
      }}
    >
      fake check of {p.group}
    </button>
  ),
}));

import { UnifiedMarkersSubtab } from "./UnifiedMarkersSubtab";

const DATASET: DatasetInfo = {
  id: "kang", name: "Kang", filename: "kang.h5ad", n_cells: 3, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], created_at: "",
  obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "pDC"] }],
};

const marker = (gene: string) => ({ gene, group: "pDC", logfoldchange: 3, pval: 1e-9, pval_adj: 1e-7 });

const RESPONSE: MarkersResponse = {
  groups: ["pDC", "B"],
  markers: { pDC: [marker("IGJ"), marker("CD74"), marker("TSPAN13")], B: [] },
} as unknown as MarkersResponse;

beforeEach(() => {
  useDatasetStore.setState({ currentDataset: DATASET, currentDatasetId: "kang" });
  apiFetch.mockReset().mockResolvedValue(RESPONSE);
});

describe("UnifiedMarkersSubtab count-split column", () => {
  it("adds the selected group's split adjusted p once the check has run", async () => {
    render(<UnifiedMarkersSubtab onGeneClick={vi.fn()} groupByColumn="cell_type" setGroupByColumn={vi.fn()} />);
    await screen.findByText("IGJ");
    expect(screen.queryByText("Split adj.P")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "fake check of pDC" }));
    expect(screen.getByText("Split adj.P")).toBeInTheDocument();
    const cell = (gene: string) => within(screen.getByText(gene).closest("tr")!).getByTestId("split-p");
    expect(cell("IGJ")).toHaveTextContent("1.00e-6");
    expect(cell("CD74")).toHaveTextContent("0.4000");
    expect(cell("TSPAN13")).toHaveTextContent("–"); // not among the check's genes
  });
});
