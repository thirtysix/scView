import { beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach } from "vitest";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { parseSession } from "@/lib/session";
import { captureSession } from "@/lib/sessionRestore";

const api = vi.hoisted(() => ({
  attachVelocity: vi.fn(),
  getVelocity: vi.fn(),
  getVelocityGenes: vi.fn(),
  getVelocityPhase: vi.fn(),
  runVelocity: vi.fn(),
}));
vi.mock("@/api/velocity", () => api);
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => vi.fn() }));
vi.mock("@/hooks/useEmbedding", () => ({ useEmbedding: () => ({ dimensions: 2 }) }));
vi.mock("@/components/plots/Plot", () => ({ default: () => <div data-testid="phase-plot" /> }));

import { VelocitySubtab } from "./VelocitySubtab";

const base = {
  methods: ["stochastic", "steady_state"],
  method_names: { stochastic: "Stochastic", steady_state: "Steady state" },
  defaults: { method: "stochastic", n_top_genes: 2000, n_neighbors: 30, n_pcs: 30 },
  note: "",
  n_cells: 100,
  running: false,
  stale: null,
};
const run = {
  result_id: "r1",
  method: "stochastic",
  method_name: "Stochastic",
  params: { n_top_genes: 2000, n_neighbors: 30, n_pcs: 30, min_shared_counts: 20 },
  n_cells: 100,
  n_dataset_cells: 100,
  cells_without: 0,
  n_genes_store: 500,
  n_genes_kept: 400,
  n_genes_fit: 300,
  n_velocity_genes: 120,
  median_confidence: 0.8,
  median_length: 5,
  unspliced_fraction: 0.16,
  neighbours_note: "Neighbours: stored.",
  pca_note: "PCA",
  group_column: "clusters",
  clusters: [{ group: "Beta", n_cells: 100, coherence: 0.73, length: 14, unspliced_fraction: 0.2, n_velocity_genes: 50, top_genes: ["Ins2"] }],
  warnings: ["Only 2% of the counts are unspliced."],
  note: "direction on a 2D plot, not a fate claim",
  embeddings: ["X_umap"],
};

describe("VelocitySubtab", () => {
  beforeEach(() => {
    Object.values(api).forEach((f) => f.mockReset());
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: { id: "d1", obs_columns: [] } as unknown as DatasetInfo });
    useSettingsStore.setState({ embedding: "X_umap", pcaDims: [0, 1], colorBy: "clusters" });
    useUnifiedViewStore.setState({ velocityRun: null, velocityArrows: { on: true, scale: 1, density: 1 } });
  });
  afterEach(cleanup);

  it("tells a dataset with no counts what to bring and where it comes from, and fetches no run", async () => {
    api.getVelocity.mockResolvedValue({ ...base, attached: false, source: null, run: null, estimate: null });
    render(<VelocitySubtab />);
    await screen.findByText(/no spliced and unspliced counts yet/);
    expect(screen.getByText(/--soloFeatures Velocyto/)).toBeTruthy();
    expect(screen.getByText(/--workflow lamanno/)).toBeTruthy();
    expect(api.getVelocityGenes).not.toHaveBeenCalled();
    expect(useUnifiedViewStore.getState().velocityRun).toBeNull();
    expect(screen.queryByRole("button", { name: /Run velocity/ })).toBeNull();
  });

  it("shows what matched, the estimate and the run's table, warning and note, and arms the arrows", async () => {
    api.getVelocity.mockResolvedValue({
      ...base,
      attached: true,
      source: { file: "v.loom", cells_matched: 90, cells_in_dataset: 100, genes_in_file: 500, genes_by_id: 400, genes_by_name: 50, cells_without_now: 10 },
      run,
      estimate: { peak_bytes: 2 ** 30, seconds: 10, seconds_range: [5, 20], note: "about 10 s", n_cells: 100 },
    });
    api.getVelocityGenes.mockResolvedValue({ genes: [{ gene: "Ins2", gamma: 0.5, r2: 0.9, velocity_gene: true }], n_genes: 1, n_velocity_genes: 1, sort: "" });
    api.getVelocityPhase.mockResolvedValue({ gene: "Ins2", gamma: 0.5, r2: 0.9, velocity_gene: true, spliced: [1], unspliced: [1], n_points: 1, n_cells: 1, capped: false, max_points: 20000, color: null, method: "stochastic" });
    render(<VelocitySubtab />);
    await screen.findByText(/90 of 100 cells matched/);
    expect(screen.getByText(/400 by id, 50 by name/)).toBeTruthy();
    expect(screen.getByText(/10 of this dataset's cells have no velocity counts/)).toBeTruthy();
    expect(screen.getByText(/about 10 s, about 1.0 GB/)).toBeTruthy();
    expect(screen.getByText("Beta")).toBeTruthy();
    expect(screen.getByText(/Only 2% of the counts/)).toBeTruthy();
    expect(screen.getByText(/not a fate claim/)).toBeTruthy();
    expect(screen.getByText(/dynamical model and latent time are not offered/)).toBeTruthy();
    expect(useUnifiedViewStore.getState().velocityRun).toEqual({ datasetId: "d1", id: "r1" });
    await waitFor(() => expect(api.getVelocityPhase).toHaveBeenCalledWith("d1", "Ins2", "clusters"));
    await screen.findByTestId("phase-plot");

    fireEvent.click(screen.getByLabelText("Arrows on the plot"));
    expect(useUnifiedViewStore.getState().velocityArrows.on).toBe(false);
    fireEvent.click(screen.getByText("Colour by confidence"));
    expect(useSettingsStore.getState().colorBy).toBe("velocity_confidence");
  });

  it("says when the cells changed and draws nothing, and on 3D says why no arrows", async () => {
    api.getVelocity.mockResolvedValue({ ...base, attached: true, source: { file: "x" }, run, stale: "The cells changed: run velocity again.", estimate: null });
    render(<VelocitySubtab />);
    await screen.findByText(/run velocity again/);
    expect(useUnifiedViewStore.getState().velocityRun).toBeNull();
    expect(api.getVelocityGenes).not.toHaveBeenCalled();
  });
});

describe("session file", () => {
  it("is written only for a dataset with a run", () => {
    const d = { id: "d1", name: "x", n_cells: 1, n_genes: 1, obs_columns: [] } as unknown as DatasetInfo;
    useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: d });
    useUnifiedViewStore.setState({ velocityRun: null });
    expect(captureSession().view?.velocity_arrows).toBeUndefined();
    useUnifiedViewStore.setState({ velocityRun: { datasetId: "d1", id: "r" }, velocityArrows: { on: false, scale: 2, density: 1 } });
    expect(captureSession().view?.velocity_arrows).toEqual({ on: false, scale: 2, density: 1 });
    useUnifiedViewStore.setState({ velocityRun: { datasetId: "other", id: "r" } });
    expect(captureSession().view?.velocity_arrows).toBeUndefined();
  });

  const session = (arrows: unknown) => ({ format: "scview-session", version: 1, view: { velocity_arrows: arrows } });

  it("keeps the arrows' switch and scale", () => {
    const s = parseSession(session({ on: false, scale: 2, density: 1.2 }));
    expect(s.view?.velocity_arrows).toEqual({ on: false, scale: 2, density: 1.2 });
    expect(parseSession(session(null)).view?.velocity_arrows).toBeNull();
  });

  it("refuses a scale outside its range", () => {
    expect(() => parseSession(session({ on: true, scale: 99, density: 1 }))).toThrow();
  });
});
