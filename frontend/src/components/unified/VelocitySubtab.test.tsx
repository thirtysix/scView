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
import { hoverTip, unhover } from "@/test/hoverTip";

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

    // Tips open on the labels themselves; the labels still name their controls.
    expect(document.querySelector("[data-infotip]")).toBeNull();
    const tip = async (el: Element) => {
      const t = await hoverTip(el);
      unhover(el);
      return t;
    };
    expect(await tip(screen.getByText("Model"))).toMatch(/Steady state fits each gene's ratio/);
    expect(await tip(screen.getByText("Neighbours"))).toMatch(/neighbouring cells each cell's counts are averaged/);
    expect(await tip(screen.getByText("Arrows on the plot"))).toMatch(/Scale sets their length/);
    expect(await tip(screen.getByText("Phase portrait"))).toMatch(/The red line is the fitted steady-state ratio/);
    expect(await tip(screen.getByText("STARsolo files"))).toMatch(/spliced and unspliced reads/);
    expect(screen.getByLabelText("Neighbours")).toBeTruthy();
    expect(screen.getByLabelText("Top genes")).toBeTruthy();
  });

  it("says which file sample is which dataset sample, where the file's other cells went, and takes several files", async () => {
    // A loom of day0 alone attached to a 3-sample dataset (Pollyea, Zenodo 6587903).
    api.getVelocity.mockResolvedValue({
      ...base,
      attached: true,
      run: null,
      estimate: null,
      source: {
        file: "day0.loom", cells_in_file: 9099, cells_matched: 3054, cells_in_dataset: 4679, cells_in_file_unmatched: 6045,
        cells_other_sample: 0, sample_key: "batch",
        samples: [{ file_sample: "cellsorted day0 UIB5X", dataset_sample: "day0", cells_in_file: 9099, cells_matched: 3054 }],
        samples_without: [{ dataset_sample: "day2", cells: 836 }, { dataset_sample: "day4", cells: 789 }],
      },
    });
    render(<VelocitySubtab />);
    await screen.findByText(/3,054 of 4,679 cells matched/);
    expect(screen.getByText(/6,045 of the file's 9,099 cells are not in this dataset \(filtered out, or another sample's\)/)).toBeTruthy();
    expect(screen.getByText("Samples matched by column batch: cellsorted day0 UIB5X = day0 (3,054 of 9,099 cells).")).toBeTruthy();
    const missing = screen.getByText(/No counts for day2 \(836 cells\), day4 \(789 cells\): attach every sample's file together/);
    expect(missing.className).toContain("text-amber-700");
    cleanup();
    api.getVelocity.mockResolvedValue({
      ...base, attached: true, run: null, estimate: null,
      source: { file: "day2.loom", cells_in_file: 1631, cells_matched: 836, cells_in_dataset: 4679, cells_in_file_unmatched: 795, cells_other_sample: 16 },
    });
    render(<VelocitySubtab />);
    await screen.findByText(/795 of the file's 1,631 cells are not in this dataset \(filtered out; 16 of them share a barcode with another sample's cell there, and were not paired with it\)/);
    const input = screen.getByLabelText("Velocity .loom or .h5ad files") as HTMLInputElement;
    expect(input.multiple).toBe(true);
    api.attachVelocity.mockResolvedValue({ attached: true, source: {} });
    const files = [new File(["a"], "day0.loom"), new File(["b"], "day2.loom")];
    fireEvent.change(input, { target: { files } });
    await waitFor(() => expect(api.attachVelocity).toHaveBeenCalledWith("d1", files));
  });

  it("shows the direction check and runs again with the same order, on the plot on screen", async () => {
    const direction = {
      checked: true, kind: "order", label: "the order of clusters given", spec: { group_by: "clusters", order: ["Pre", "Beta"] },
      group_column: "clusters", embedding: "X_umap", k: 30, n_backward: 1, note: "Direction check.",
      transitions: [{ from: "Pre", to: "Beta", n_cells: 86, mean_cos: -0.4, low: -0.5, high: -0.3, share_forward: 0.2, status: "backward" }],
    };
    api.getVelocity.mockResolvedValue({ ...base, attached: true, source: { file: "x" }, run: { ...run, direction }, estimate: null });
    api.getVelocityGenes.mockResolvedValue({ genes: [], n_genes: 0, n_velocity_genes: 0, sort: "" });
    api.runVelocity.mockResolvedValue({ ...run, direction });
    render(<VelocitySubtab />);
    await screen.findByText(/Backward at Pre -> Beta \(-0.40, n = 86\)/);
    expect((screen.getByLabelText("Group order") as HTMLInputElement).value).toBe("Pre > Beta");
    fireEvent.click(screen.getByText("Run again"));
    await waitFor(() => expect(api.runVelocity).toHaveBeenCalled());
    expect(api.runVelocity.mock.calls[0]![1]).toMatchObject({
      group_by: "clusters", embedding: "X_umap", direction: { group_by: "clusters", order: ["Pre", "Beta"] },
    });
  });

  it("leaves out typed genes and the MURK list, and says what was left out", async () => {
    const murk = { id: "murk_barile2021", name: "MURK genes (Barile 2021)", n_genes: 89, species: "mouse", source: "", url: "", note: "" };
    const excluded = { sets: ["murk_barile2021"], n_given: 91, n_matched: 90, n_unmatched: 1, unmatched: ["Nope"], n_passing_filter: 78 };
    const withRun = { ...run, params: { ...run.params, exclude_sets: ["murk_barile2021"], exclude_given: ["Gata1", "Nope"] }, excluded };
    api.getVelocity.mockResolvedValue({ ...base, attached: true, source: { file: "x" }, run: withRun, estimate: null, gene_sets: [murk] });
    api.getVelocityGenes.mockResolvedValue({ genes: [], n_genes: 0, n_velocity_genes: 0, sort: "" });
    api.runVelocity.mockResolvedValue(withRun);
    render(<VelocitySubtab />);
    await screen.findByText(/Left out: 90 genes \(with MURK genes \(Barile 2021\)\), 78 of them with enough counts/);
    expect(screen.getByText(/Not in the counts: Nope \(1\)/)).toBeTruthy();
    const box = screen.getByLabelText("Genes to leave out") as HTMLTextAreaElement;
    expect(box.value).toBe("Gata1 Nope");
    const murkBox = screen.getByText(/MURK genes \(Barile 2021\), 89 mouse/).closest("label")!.querySelector("input")!;
    expect(murkBox.checked).toBe(true);
    fireEvent.change(box, { target: { value: "Gata1, Klf1\nGata1" } });
    fireEvent.click(screen.getByText("Run again"));
    await waitFor(() => expect(api.runVelocity).toHaveBeenCalled());
    expect(api.runVelocity.mock.calls[0]![1]).toMatchObject({ exclude_genes: ["Gata1", "Klf1"], exclude_sets: ["murk_barile2021"] });
    expect(await hoverTip(screen.getByText(/MURK genes \(Barile 2021\), 89 mouse/))).toMatch(/Barile et al. 2021/);
  });

  it("leaves the flagged genes out and runs again", async () => {
    const direction = {
      checked: true, kind: "order", label: "x", spec: { group_by: "clusters", order: ["Pre", "Beta"] }, group_column: "clusters",
      embedding: "X_umap", k: 30, note: "", transitions: [],
      gene_flags: { checked: true, rule: "rule", n_velocity_genes: 100, n_flagged: 1, top_n: 50, n_in_top: 1,
        genes: [{ gene: "Smim1", r2: 0.5, rank: 3, steps: [] }] },
    };
    const withRun = { ...run, params: { ...run.params, exclude_given: ["Gata1"] }, direction };
    api.getVelocity.mockResolvedValue({ ...base, attached: true, source: { file: "x" }, run: withRun, estimate: null });
    api.getVelocityGenes.mockResolvedValue({ genes: [], n_genes: 0, n_velocity_genes: 0, sort: "" });
    api.runVelocity.mockResolvedValue(withRun);
    render(<VelocitySubtab />);
    fireEvent.click(await screen.findByText("Leave these out and run again"));
    await waitFor(() => expect(api.runVelocity).toHaveBeenCalled());
    expect(api.runVelocity.mock.calls[0]![1].exclude_genes).toEqual(["Gata1", "Smim1"]);
    expect((screen.getByLabelText("Genes to leave out") as HTMLTextAreaElement).value).toBe("Gata1 Smim1");
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
