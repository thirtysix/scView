import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";

const client = vi.hoisted(() => ({ apiFetch: vi.fn() }));
vi.mock("@/api/client", () => client);
vi.mock("@/components/panels/CustomCollectionsGroup", () => ({
  CollectionOptions: () => null,
  CustomCollectionsGroup: () => null,
}));

import { UnifiedGeneSetsSubtab } from "./UnifiedGeneSetsSubtab";

const dataset = {
  id: "ds1", name: "kang", filename: "kang.h5ad", n_cells: 6, n_genes: 10, status: "ready",
  available_embeddings: ["X_umap"], obs_columns: [], created_at: "",
} as DatasetInfo;

const sets = (n: number) =>
  Array.from({ length: n }, (_, i) => ({ name: `SET_${i}`, collection: "c2", genes: ["A"], n_genes: 1 }));

function respond(nSets: number, listed: number) {
  client.apiFetch.mockImplementation((url: string) => {
    if (url.includes("/genesets/collections")) {
      return Promise.resolve({
        dataset_id: "ds1",
        collections: [{ id: "c2", name: "C2", category: "MSigDB", n_sets: nSets, available: true }],
      });
    }
    if (url.includes("/genesets/search")) return Promise.resolve({ results: sets(listed) });
    return Promise.reject(new Error(`unexpected ${url}`));
  });
}

beforeEach(() => {
  client.apiFetch.mockReset();
  useDatasetStore.setState({ currentDataset: dataset, currentDatasetId: "ds1" });
});

describe("UnifiedGeneSetsSubtab list cap", () => {
  it("says the list is the first 200 of the collection's sets when it has more", async () => {
    respond(3_050, 200);
    render(<UnifiedGeneSetsSubtab onScoreComplete={vi.fn()} groupByColumn="" setGroupByColumn={vi.fn()} />);
    await screen.findByText("SET_0");
    expect(screen.getByTestId("genesets-cap")).toHaveTextContent(
      "The first 200 of the collection's 3,050 sets, in file order; the filter above searches only these.",
    );
  });

  it("says nothing when the collection fits", async () => {
    respond(50, 50);
    render(<UnifiedGeneSetsSubtab onScoreComplete={vi.fn()} groupByColumn="" setGroupByColumn={vi.fn()} />);
    await screen.findByText("SET_0");
    await waitFor(() => expect(screen.queryByTestId("genesets-cap")).toBeNull());
  });
});
