import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { DatasetInfo } from "@/api/types";

const { listDatasets, getDataset, apiFetch, testComposition } = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  getDataset: vi.fn(),
  apiFetch: vi.fn(),
  testComposition: vi.fn(),
}));
vi.mock("@/api/datasets", () => ({ listDatasets, getDataset }));
vi.mock("@/api/client", () => ({ apiFetch }));
vi.mock("@/api/composition", () => ({ testComposition }));
vi.mock("@/components/plots/Plot", () => ({ default: () => null }));
vi.mock("@/components/compare/JointEmbedding", () => ({ JointEmbedding: () => null }));
vi.mock("@/components/compare/MiloTest", () => ({
  MiloTest: ({ datasetId }: { datasetId: string }) => <div data-testid="milo">{datasetId}</div>,
}));

import { ComparePanel } from "./ComparePanel";
import { useDatasetStore } from "@/stores/datasetStore";
import { useViewStore } from "@/stores/viewStore";

const ds = (id: string, name: string): DatasetInfo => ({
  id,
  name,
  filename: `${id}.h5ad`,
  n_cells: 100,
  n_genes: 10,
  status: "ready",
  available_embeddings: [],
  obs_columns: [
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] },
    { name: "Donor", dtype: "category", n_unique: 4 },
  ],
  created_at: "",
});

describe("ComparePanel composition", () => {
  beforeEach(() => {
    const a = ds("a", "Kang");
    useDatasetStore.setState({ currentDatasetId: "a", currentDataset: a });
    listDatasets.mockResolvedValue([a, ds("b", "Other")]);
    getDataset.mockImplementation(async (id: string) => ds(id, id === "b" ? "Other" : "Kang"));
    apiFetch.mockResolvedValue({ groupby: "cell_type", counts: { B: 30, T: 70 } });
    testComposition.mockReset().mockReturnValue(new Promise(() => {}));
  });

  it("opens on the replicate-aware test of the loaded dataset", async () => {
    render(<ComparePanel />);
    fireEvent.click(screen.getByRole("button", { name: "Composition" }));
    expect(screen.getByRole("button", { name: "Between conditions" })).toHaveClass("bg-primary");
    await waitFor(() =>
      expect(testComposition).toHaveBeenCalledWith("a", {
        condition_key: "stim", group_key: "cell_type", transform: "logit",
      }),
    );
  });

  it("offers Milo next to the group test, with when to use which", async () => {
    render(<ComparePanel />);
    fireEvent.click(screen.getByRole("button", { name: "Composition" }));
    expect(screen.getByText(/Between conditions tests whether whole cell types/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Neighbourhoods (Milo)" }));
    expect(screen.getByTestId("milo").textContent).toBe("a");
    expect(screen.getByText(/needs no clustering/)).toBeInTheDocument();
    expect(screen.getByText(/change in abundance between two conditions/)).toBeInTheDocument();
  });

  it("calls two datasets' differences descriptive, on screen and to the co-pilot", async () => {
    render(<ComparePanel />);
    fireEvent.click(screen.getByRole("button", { name: "Composition" }));
    fireEvent.click(screen.getByRole("button", { name: "Two datasets" }));
    await screen.findByRole("option", { name: "Other" });
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "b" } });
    expect(await screen.findByText(/Each dataset counts as one sample here/)).toBeInTheDocument();
    expect(screen.getByText(/overstate significance/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Ask co-pilot to compare/ }));
    expect(useViewStore.getState().pendingAsk).toContain("descriptive, not tested");
  });
});
