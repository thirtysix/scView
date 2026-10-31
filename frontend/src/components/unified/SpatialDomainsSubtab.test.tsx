import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { DomainsInfo, DomainsResult } from "@/api/spatialDomains";
import type { DatasetInfo, SpatialInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getSpatialDomains: vi.fn(),
  runDomains: vi.fn(),
  runEnrichment: vi.fn(),
  runNiches: vi.fn(),
}));
vi.mock("@/api/spatialDomains", async (orig) => ({
  ...(await orig<typeof import("@/api/spatialDomains")>()),
  ...api,
}));
const refresh = vi.hoisted(() => vi.fn(() => Promise.resolve({})));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));
vi.mock("@/components/plots/Plot", () => ({
  default: (props: { data: { type: string; z?: unknown; colorscale?: string; zmin?: number; zmax?: number }[]; layout: { xaxis?: { type?: string } } }) => (
    <div data-testid="plot" data-trace={props.data[0]?.type} data-axis={props.layout.xaxis?.type ?? ""}
         data-range={`${props.data[0]?.zmin ?? ""},${props.data[0]?.zmax ?? ""}`} />
  ),
}));

import { SpatialDomainsSubtab } from "./SpatialDomainsSubtab";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSpatialDomainsRunStore } from "@/stores/spatialDomainsRunStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const lib = (id: string, n: number) => ({
  id, n_obs: n, images: {}, spot_diameter_fullres: 89, bounds: [0, 0, 100, 100] as [number, number, number, number],
});
const VISIUM: SpatialInfo = { technology: "visium", unit: "spot", library_column: "library_id", libraries: [lib("A", 1000)] };
const HD: SpatialInfo = { technology: "visium_hd", unit: "bin", library_column: "library_id", libraries: [lib("A", 1000)] };

function setDataset(spatial: SpatialInfo | null) {
  const ds: DatasetInfo = {
    id: "d1", name: "d1", filename: "", n_cells: 1000, n_genes: 100, status: "ready", available_embeddings: ["X_umap"],
    obs_columns: [
      { name: "leiden", dtype: "category", n_unique: 5 },
      { name: "total_counts", dtype: "float32", n_unique: 900 },
    ] as DatasetInfo["obs_columns"],
    created_at: "", spatial,
  };
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: ds });
}

const PLAN = {
  n_obs: 1000, technology: "visium", unit: "spot", libraries: ["A"], rep: "X_pca", n_pcs: 10, n_pcs_available: 50, n_hops: 3,
  dims: 40, k: 7, k_min: 2, k_max: 15, repeats: 5, seed: 0, n_fit: 1000, graph: "the capture grid (6 neighbours)", notes: [],
  estimate_seconds: 20, estimate_scan_seconds: 180, estimate_bytes: 1000,
};
const INFO: DomainsInfo = {
  runnable: { ok: true, reason: null, plan: PLAN },
  defaults: { k: 7, n_hops: 3, n_pcs: 10, k_min: 2, k_max: 15, repeats: 5, max_fit_obs: 50000, n_perms: 1000, k_window: 20, n_niches: 8, domain_column: "spatial_domain", niche_column: "spatial_niche" },
  n_obs: 1000,
  free_column: "spatial_domain",
  free_suggested_column: "spatial_domain_suggested",
  free_niche_column: "spatial_niche",
  niches_allowed: { ok: false, reason: "Composition niches need one label per cell, and a Visium spot mixes several cells.", note: null },
  estimates: { enrichment_seconds: 30, enrichment_bytes: 1000 },
  domains: { available: false, columns: [] },
  niches: { available: false, columns: [] },
  enrichment: { available: false, columns: [] },
  running: false,
};
const DONE: DomainsResult = {
  available: true, columns: ["spatial_domain"], column: "spatial_domain", k: 3, suggested: false, unit: "spot", rep: "X_pca", n_pcs: 10,
  n_hops: 3, n_obs: 1000, n_fit: 1000, graph: "the capture grid (6 neighbours)", n_links: 2900, n_isolated: 0, libraries: ["A"], notes: [],
  sizes: [500, 300, 200], per_library: { A: [500, 300, 200] }, coherence: { within: 0.8, shuffled: 0.35 }, seconds: 2,
  method: "CellCharter-style spatial domains", stale: false, stale_reason: null, scan: null,
};

const groupProps = { groupByColumn: "leiden", setGroupByColumn: vi.fn() };

beforeEach(() => {
  for (const f of Object.values(api)) f.mockReset();
  refresh.mockClear();
  groupProps.setGroupByColumn.mockReset();
  api.getSpatialDomains.mockResolvedValue(INFO);
  useSpatialDomainsRunStore.setState({
    datasetId: null, kind: null, running: false, stage: null, progress: null, curve: null, warnings: [], error: null, finished: null,
    version: 0,
  });
  useSettingsStore.setState({ colorBy: "leiden", embedding: "X_umap" });
  useUnifiedViewStore.setState({ activeSubtab: "domains" });
  setDataset(VISIUM);
});

describe("SpatialDomainsSubtab", () => {
  it("says what a domain is and what a run will need before it starts", async () => {
    render(<SpatialDomainsSubtab {...groupProps} />);
    expect(screen.getByText(/stretch of tissue whose spots look alike/)).toBeInTheDocument();
    const plan = await screen.findByTestId("domains-plan");
    expect(plan).toHaveTextContent("First 10 components of the PCA, 3 hops");
    expect(plan).toHaveTextContent("Suggest k tries k = 2 to 15, 5 fits each; about 3 min");
    expect(api.getSpatialDomains).toHaveBeenCalledWith("d1", { k: 7, n_hops: 3, n_pcs: 10, enrichment_column: undefined });
  });

  it("calls bins bins", async () => {
    setDataset(HD);
    render(<SpatialDomainsSubtab {...groupProps} />);
    expect(screen.getByText(/stretch of tissue whose bins look alike/)).toBeInTheDocument();
    await screen.findByTestId("domains-plan");
  });

  it("runs at the chosen k and colours the tissue by the result when it finishes", async () => {
    api.runDomains.mockResolvedValue(DONE);
    render(<SpatialDomainsSubtab {...groupProps} />);
    await screen.findByTestId("domains-plan");
    fireEvent.change(screen.getByLabelText("Domains (k)"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Find domains" }));
    await waitFor(() => expect(api.runDomains).toHaveBeenCalled());
    expect(api.runDomains.mock.calls[0]![1]).toEqual({ k: 3, n_hops: 3, n_pcs: 10, column: "spatial_domain", overwrite: false });
    await waitFor(() => expect(refresh).toHaveBeenCalledWith("d1", "spatial_domain"));
    await waitFor(() => expect(useSettingsStore.getState().embedding).toBe("spatial"));
    expect(useSettingsStore.getState().colorBy).toBe("spatial_domain");
  });

  it("Suggest k asks for no k, and the curve says what it suggests", async () => {
    const curve = { ks: [2, 3, 4], mean: [0.5, 0.8, 0.4], sd: [0.1, 0.1, 0.1], peaks: [3], suggested: 3, has_peak: true, repeats: 5, n_fit: 1000 };
    api.runDomains.mockResolvedValue({ ...DONE, suggested: true, scan: curve });
    api.getSpatialDomains.mockResolvedValueOnce(INFO).mockResolvedValue({ ...INFO, domains: { ...DONE, suggested: true, scan: curve } });
    render(<SpatialDomainsSubtab {...groupProps} />);
    await screen.findByTestId("domains-plan");
    fireEvent.click(screen.getByRole("button", { name: "Suggest k" }));
    // The click runs the preflight before the call: a slow CI runner took over the 1 s default.
    await waitFor(() => expect(api.runDomains).toHaveBeenCalled(), { timeout: 5000 });
    expect(api.runDomains.mock.calls[0]![1].k).toBeNull();
    expect(api.runDomains.mock.calls[0]![1].column).toBe("spatial_domain_suggested");
    const figure = await screen.findByTestId("stability-curve");
    expect(figure).toHaveTextContent("Suggested k = 3, the highest local peak");
  });

  it("will not replace an earlier result until the user says so", async () => {
    api.getSpatialDomains.mockResolvedValue({ ...INFO, domains: DONE, free_column: "spatial_domain_k7" });
    render(<SpatialDomainsSubtab {...groupProps} />);
    await screen.findByText(/3 domains in/);
    const name = screen.getByLabelText("Save as") as HTMLInputElement;
    await waitFor(() => expect(name.value).toBe("spatial_domain_k7"));
    expect(screen.getByRole("button", { name: "Find domains" })).toBeEnabled();
    fireEvent.change(name, { target: { value: "spatial_domain" } });
    expect(screen.getByRole("button", { name: "Find domains" })).toBeDisabled();
    fireEvent.click(screen.getByLabelText(/Replace the earlier result/));
    expect(screen.getByRole("button", { name: "Find domains" })).toBeEnabled();
    api.runDomains.mockResolvedValue(DONE);
    fireEvent.click(screen.getByRole("button", { name: "Find domains" }));
    await waitFor(() => expect(api.runDomains).toHaveBeenCalled(), { timeout: 5000 });
    expect(api.runDomains.mock.calls[0]![1]).toMatchObject({ column: "spatial_domain", overwrite: true });
  });

  it("shows sizes with n, the coherence, a stale reason, and Find markers opens Markers on the column", async () => {
    api.getSpatialDomains.mockResolvedValue({
      ...INFO, domains: { ...DONE, stale: true, stale_reason: "The PCs changed since this ran (a pipeline run): run it again." },
    });
    render(<SpatialDomainsSubtab {...groupProps} />);
    await screen.findByText(/n = 1,000 spots/);
    expect(screen.getByText(/80% of neighbouring spots share a domain \(same labels shuffled: 35%\)/)).toBeInTheDocument();
    expect(screen.getByTestId("domains-stale")).toHaveTextContent("PCs changed");
    fireEvent.click(screen.getAllByRole("button", { name: "Find markers" })[0]!);
    expect(groupProps.setGroupByColumn).toHaveBeenCalledWith("spatial_domain");
    expect(useUnifiedViewStore.getState().activeSubtab).toBe("markers");
  });

  it("draws enrichment on category axes, symmetric about 0, and names the strongest pairs", async () => {
    api.getSpatialDomains.mockResolvedValue({
      ...INFO,
      enrichment: {
        available: true, columns: ["leiden"], column: "leiden", labels: ["0", "1", "2"], sizes: [1, 2, 3],
        zscore: [[10, -8, 3], [-8, 6, null], [3, null, 4]], count: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], n_perms: 1000, n_labelled: 1000,
        unit: "spot", notes: [], stale: false, stale_reason: null,
      },
    });
    render(<SpatialDomainsSubtab {...groupProps} />);
    const figure = await screen.findByTestId("enrichment-heatmap");
    const plot = figure.querySelector("[data-testid=plot]")!;
    expect(plot.getAttribute("data-axis")).toBe("category");
    expect(plot.getAttribute("data-range")).toBe("-10,10");
    expect(figure).toHaveTextContent("Most often side by side: 0 and 2 (z 3.0)");
    expect(figure).toHaveTextContent("Most apart: 0 and 1 (z -8.0)");
    expect(figure).toHaveTextContent("n = 1,000 spots; 3 labels");
  });

  it("refuses niches on spots with the backend's sentence and allows them on cells", async () => {
    render(<SpatialDomainsSubtab {...groupProps} />);
    expect(await screen.findByTestId("niches-refused")).toHaveTextContent("a Visium spot mixes several cells");
    expect(screen.queryByRole("button", { name: "Find niches" })).toBeNull();
  });

  it("runs niches where they are allowed and draws the niche table", async () => {
    const allowed = { ok: true, reason: null, note: null };
    const niches = {
      available: true, columns: ["spatial_niche"], column: "spatial_niche", label_column: "leiden", labels: ["a", "b"], niches: ["0", "1"],
      fractions: [[0.9, 0.1], [0.2, 0.8]], sizes: [600, 400], k_window: 20, n_niches: 2, n_labelled: 1000, unit: "cell", notes: [],
      stale: false, stale_reason: null,
    };
    api.getSpatialDomains.mockResolvedValue({ ...INFO, niches_allowed: allowed });
    api.runNiches.mockResolvedValue(niches);
    render(<SpatialDomainsSubtab {...groupProps} />);
    const button = await screen.findByRole("button", { name: "Find niches" });
    api.getSpatialDomains.mockResolvedValue({ ...INFO, niches_allowed: allowed, niches });
    fireEvent.click(button);
    await waitFor(() => expect(api.runNiches).toHaveBeenCalled());
    expect(api.runNiches.mock.calls[0]![1]).toEqual({ label_column: "leiden", k_window: 20, n_niches: 8, column: "spatial_niche" });
    const figure = await screen.findByTestId("niche-heatmap");
    expect(figure).toHaveTextContent('2 niches in "spatial_niche"');
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe("spatial_niche"));
  });

  it("shows a run's progress and Cancel while it streams", async () => {
    useSpatialDomainsRunStore.setState({
      datasetId: "d1", kind: "domains", running: true,
      stage: { stage: "scan", label: "Stability of each k", index: 3, total: 5, elapsed_s: 20 },
      progress: { done: 12, total: 75, elapsed_s: 30 },
    });
    render(<SpatialDomainsSubtab {...groupProps} />);
    const box = await screen.findByTestId("domains-progress");
    expect(box).toHaveTextContent("Step 3 of 5: Stability of each k, 12 of 75 (30 s so far)");
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Find domains" })).toBeDisabled();
  });

  it("says so without positions", () => {
    setDataset(null);
    render(<SpatialDomainsSubtab {...groupProps} />);
    expect(screen.getByText("This dataset has no spot or cell positions.")).toBeInTheDocument();
    expect(api.getSpatialDomains).not.toHaveBeenCalled();
  });
});
