import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  getWnn: vi.fn(),
  getWnnWeights: vi.fn(),
  runWnn: vi.fn(),
  getDataset: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
}));
vi.mock("@/api/wnn", () => ({ getWnn: api.getWnn, getWnnWeights: api.getWnnWeights, runWnn: api.runWnn }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary: api.fetchEmbeddingBinary }));
vi.mock("@/components/plots/Plot", () => ({
  default: ({ data }: { data: { type: string; name?: string }[] }) => (
    <div data-testid="plot">{data.map((t) => `${t.type}:${t.name}`).join(";")}</div>
  ),
}));

import type { WnnInfo, WnnPlan } from "@/api/wnn";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useWnnRunStore } from "@/stores/wnnRunStore";
import { WnnCard } from "./WnnCard";

const PLAN: WnnPlan = {
  n_cells: 4, proteins: ["CD3", "CD4", "CD8a", "CD14", "CD19", "CD45RA"], controls: ["IgG1_control"], n_proteins: 6,
  rep: "X_pca", d_rna: 50, d_protein: 5, batch_key: null, n_zero_protein: 0, k: 20, knn_range: 200,
  resolution: 1, exact: true, warnings: ["Only 6 proteins: RNA weight is expected wherever the panel has no markers."],
  notes: [], estimate_seconds: 40,
};
const NOT_RUN: WnnInfo = { available: false, runnable: { ok: true, reason: null, plan: PLAN }, running: false, n_cells_total: 4 };
const SAVED: WnnInfo = {
  ...NOT_RUN, available: true, run_id: "r1", plan: PLAN, stale: false, stale_reason: null, n_clusters: 3,
  mean_rna_weight: 0.42, embeddings: ["X_umap", "X_umap_protein", "X_umap_wnn"], clusters_column: "wnn_leiden",
  by_group: { column: "cell_type", rows: [
    { group: "pDC", n_cells: 20, mean_rna_weight: 0.75, median_rna_weight: 0.7 },
    { group: "CD4 T", n_cells: 30, mean_rna_weight: 0.25, median_rna_weight: 0.3 },
  ] },
};

function dataset(extra: Partial<DatasetInfo> = {}): DatasetInfo {
  return {
    id: "ds", name: "cite", n_cells: 4,
    obs_columns: [
      { name: "cell_type", dtype: "category", n_unique: 2, values: ["pDC", "CD4 T"] },
      { name: "wnn_leiden", dtype: "category", n_unique: 3, values: ["0", "1", "2"] },
    ],
    available_embeddings: ["X_umap", "X_umap_protein", "X_umap_wnn"],
    ...extra,
  } as unknown as DatasetInfo;
}

function renderCard() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <WnnCard datasetId="ds" groupByColumn="cell_type" />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  useWnnRunStore.getState().reset();
  useDatasetStore.setState({ currentDataset: dataset(), currentDatasetId: "ds" });
  useSettingsStore.getState().setColorBy("cell_type");
  useUnifiedViewStore.getState().setScatterOverlay(null);
  api.fetchEmbeddingBinary.mockReturnValue(new Promise(() => {}));
  api.getDataset.mockResolvedValue(dataset());
  api.getWnnWeights.mockResolvedValue({
    run_id: "r1", values: [0.2, 0.8, 0.5, 0.1], column: "cell_type",
    violin: { pDC: [0.8, 0.5], "CD4 T": [0.2, 0.1] },
  });
});

describe("WNN card", () => {
  it("shows the plan and runs WNN, stage by stage", async () => {
    api.getWnn.mockResolvedValueOnce(NOT_RUN).mockResolvedValue(SAVED);
    let release: () => void = () => {};
    api.runWnn.mockImplementation(async (_id, _body, h) => {
      h.onStart({ ...PLAN, stages: [] });
      h.onStage({ stage: "knn_protein", label: "Protein neighbours", index: 3, total: 9, elapsed_s: 12 });
      await new Promise<void>((r) => (release = r));
      return SAVED;
    });
    renderCard();
    expect(await screen.findByTestId("wnn-plan")).toHaveTextContent(
      "6 proteins (1 isotype control left out), 5 protein PCs, 50 RNA PCs (PCA); about 40 s.");
    expect(screen.getByText(/Only 6 proteins/)).toBeInTheDocument();
    expect(screen.getByText(/Hao et al. 2021, Cell 184:3573/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run WNN" }));
    expect(await screen.findByText("Step 3 of 9: Protein neighbours (10 s so far)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    release();
    expect(await screen.findByRole("button", { name: "Run WNN again" })).toBeInTheDocument();
    expect(api.runWnn).toHaveBeenCalledWith("ds", {}, expect.anything());
    await waitFor(() => expect(api.getDataset).toHaveBeenCalledWith("ds"));
  });

  it("shows a saved run: layouts, weights by group, the caveat; colours the cells", async () => {
    api.getWnn.mockResolvedValue(SAVED);
    renderCard();
    expect(await screen.findByText(/mean RNA weight 0.42/)).toBeInTheDocument();
    for (const key of ["X_umap", "X_umap_protein", "X_umap_wnn"]) {
      expect(screen.getByTestId(`wnn-thumb-${key}`)).toBeInTheDocument();
    }
    expect(api.fetchEmbeddingBinary).toHaveBeenCalledWith("ds", "X_umap_wnn", "cell_type");
    expect(screen.getByText(/highest in pDC, lowest \(protein counts most\) in CD4 T/)).toBeInTheDocument();
    expect(screen.getByText(/not quality scores/)).toBeInTheDocument();
    expect(await screen.findByTestId("plot")).toHaveTextContent("box:pDC;box:CD4 T");

    fireEvent.click(screen.getByRole("button", { name: "Colour by RNA weight" }));
    await waitFor(() => expect(useUnifiedViewStore.getState().scatterOverlay).toEqual(
      { type: "score", name: "RNA weight (WNN)", share: true }));
    const ui = useUnifiedViewStore.getState();
    expect(Array.from(ui.overlayValues ?? [])).toEqual([0.2, 0.8, 0.5, 0.1].map(Math.fround));
    expect(ui.violinTitle).toBe("RNA weight (WNN)");

    fireEvent.click(screen.getByRole("button", { name: "Colour by WNN clusters" }));
    expect(useSettingsStore.getState().colorBy).toBe("wnn_leiden");
  });

  it("says why it cannot run, and when a saved run is stale", async () => {
    api.getWnn.mockResolvedValueOnce({
      ...NOT_RUN, runnable: { ok: false, reason: "WNN needs at least 5 proteins besides isotype controls; this panel has 3.", plan: null },
    });
    const { unmount } = renderCard();
    expect(await screen.findByText(/at least 5 proteins/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run WNN" })).toBeNull();
    unmount();
    api.getWnn.mockResolvedValue({ ...SAVED, stale: true, stale_reason: "The cells changed since WNN ran (a filter or a merge): run it again." });
    renderCard();
    expect(await screen.findByTestId("wnn-stale")).toHaveTextContent("The cells changed since WNN ran");
  });
});
