import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

const api = vi.hoisted(() => ({ getDataset: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));

import type { DatasetInfo, SubsetOf } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { SubsetSourceCard } from "./SubsetSourceCard";

const parent: DatasetInfo = {
  id: "p1", name: "kidney", filename: "k.h5ad", n_cells: 47000, n_genes: 30000, status: "ready",
  available_embeddings: ["X_umap"], obs_columns: [], created_at: "",
};
const source: SubsetOf = {
  id: "p1", name: "kidney", how: { groups: { column: "cell_type", values: ["PT S1"] } },
  described: "cell_type in {PT S1, PT S2}", n_cells: 9120, n_parent_cells: 46957,
  carried: ["12 obs columns"], not_carried: ["stored results"],
};

describe("SubsetSourceCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({ currentDataset: null, availableDatasets: [parent] });
  });

  it("names the parent, the cells and what came along, and opens the parent", async () => {
    api.getDataset.mockResolvedValue(parent);
    render(<SubsetSourceCard source={source} />);
    const card = screen.getByTestId("subset-source");
    expect(card).toHaveTextContent("9,120 of 46,957 cells, chosen by cell_type in {PT S1, PT S2}.");
    expect(card).toHaveTextContent(/Carried\s*the cells' counts; 12 obs columns/);
    expect(card).toHaveTextContent(/Not carried\s*stored results/);
    fireEvent.click(screen.getByRole("button", { name: "kidney" }));
    await waitFor(() => expect(useDatasetStore.getState().currentDataset?.id).toBe("p1"));
  });

  it("says when the parent is gone", () => {
    useDatasetStore.setState({ availableDatasets: [] });
    render(<SubsetSourceCard source={{ ...source, carried: [], not_carried: [] }} />);
    expect(screen.getByTestId("subset-source")).toHaveTextContent("kidney (no longer in the list)");
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByTestId("subset-source")).not.toHaveTextContent("Carried");
  });
});
