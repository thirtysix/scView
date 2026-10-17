import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const api = vi.hoisted(() => ({
  getHvgSensitivityInfo: vi.fn(),
  runHvgSensitivity: vi.fn(),
  keepHvgSetting: vi.fn(),
  getDataset: vi.fn(),
  fetchEmbeddingBinary: vi.fn(),
  decodeArrowBuffer: vi.fn(),
}));
vi.mock("@/api/clustering", () => ({
  getHvgSensitivityInfo: api.getHvgSensitivityInfo,
  runHvgSensitivity: api.runHvgSensitivity,
  keepHvgSetting: api.keepHvgSetting,
}));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/embeddings", () => ({ fetchEmbeddingBinary: api.fetchEmbeddingBinary }));
vi.mock("@/lib/arrow", () => ({ decodeArrowBuffer: api.decodeArrowBuffer }));

import type { HvgClusterMatch, HvgRow, HvgSensitivityInfo, HvgSetting } from "@/api/clustering";
import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { HvgSensitivityPanel } from "./HvgSensitivity";

const REF = "scview_leiden_r0.5";

const setting = (flavor: HvgSetting["flavor"], n: number, label: string): HvgSetting => ({
  flavor,
  n_top_genes: n,
  key: `${flavor}_${n}`,
  label: `${label}, ${n.toLocaleString("en-US")} genes`,
});

const defaults = [setting("seurat", 2000, "Seurat"), setting("seurat", 1000, "Seurat"), setting("cell_ranger", 2000, "Cell Ranger")];

const info: HvgSensitivityInfo = {
  available: true,
  reason: null,
  n_cells: 400,
  n_genes: 3000,
  max_settings: 8,
  min_genes: 200,
  max_genes: 10000,
  max_cells: 100000,
  first_run_seconds: 30,
  recorded: {
    reference: REF,
    method: "leiden",
    resolution: 0.5,
    flavor: "seurat",
    n_top_genes: 2000,
    hvg_batch_key: null,
    target_sum: 10000,
    n_comps: 50,
    n_neighbors: 15,
    batch_key: null,
    notes: [],
    reference_from_file: false,
    pcs_made_by: "0.16.1",
    pcs_made_on: "2026-10-17",
    pcs_older: false,
  },
  current: defaults[0],
  default_settings: defaults,
  flavors: [
    { flavor: "seurat", label: "Seurat", unavailable: null },
    { flavor: "cell_ranger", label: "Cell Ranger", unavailable: null },
    { flavor: "seurat_v3", label: "Seurat v3 (counts)", unavailable: "Needs scikit-misc, which is not installed." },
    { flavor: "pearson_residuals", label: "Pearson residuals", unavailable: null },
  ],
  estimate_seconds: 36,
  seconds_per_setting: 12,
};

const m = (cluster: string, jaccard: number, status: HvgClusterMatch["status"] = "stable"): HvgClusterMatch => ({
  cluster, n_cells: 200, jaccard, match: "0", contained: jaccard, purity: 1, parts: 1, status,
});

const result = (s: HvgSetting, clusters: HvgClusterMatch[], extra: Partial<HvgRow> = {}): HvgRow => ({
  ...s,
  is_current: s.key === "seurat_2000",
  n_hvgs: s.n_top_genes,
  shared_hvgs: Math.min(s.n_top_genes, 2000),
  ari: 0.8,
  ami: 0.85,
  n_clusters: 2,
  sizes: [200, 200],
  clusters,
  seconds: 10,
  column: `${REF}_hvg_${s.key}`,
  exists: false,
  ...extra,
});

function renderPanel() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <HvgSensitivityPanel datasetId="ds" onClose={() => {}} />
    </QueryClientProvider>,
  );
}

function runWithRows() {
  api.runHvgSensitivity.mockImplementation(async (_id, body, h) => {
    h.onStart({ n_cells: 400, reference: REF, recorded: info.recorded, current: defaults[0], settings: defaults,
                n_current_hvgs: 2000, estimate_seconds: 36 });
    h.onResult(result(defaults[0]!, [m("A", 1), m("B", 1)], { ari: 1, ami: 1 }));
    h.onResult(result(defaults[1]!, [m("A", 0.95), m("B", 0.4, "splits")]));
    h.onResult(result(defaults[2]!, [m("A", 0.9), m("B", 0.8)]));
    return body;
  });
}

describe("HvgSensitivityPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useDatasetStore.setState({
      currentDatasetId: "ds",
      currentDataset: { id: "ds", obs_columns: [{ name: REF, dtype: "category", values: ["A", "B"] }] } as unknown as DatasetInfo,
    });
    useSettingsStore.setState({ colorBy: REF, embedding: "X_umap" });
    useUnifiedViewStore.getState().clearOverlayState();
    useSelectionStore.setState({ highlightedGroup: null });
  });

  it("explains why the check can't run", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue({ ...info, available: false, reason: "This dataset has no clustering to compare with." });
    renderPanel();
    expect(await screen.findByRole("alert")).toHaveTextContent("no clustering");
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
  });

  it("runs the default settings and shows each one and each cluster's stability", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue(info);
    runWithRows();
    renderPanel();
    expect(await screen.findByText(/About 35 s for 3 settings/)).toBeInTheDocument();
    expect(screen.getByText(/Leiden at 0.5, 50 PCs, 15 neighbours/)).toBeInTheDocument();
    // An unavailable flavour says why.
    expect(screen.getByText(/Seurat v3 \(counts\): Needs scikit-misc/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await waitFor(() => expect(screen.getByText("(current, re-run)")).toBeInTheDocument());
    expect(api.runHvgSensitivity.mock.calls[0]![1]).toEqual({
      settings: [
        { flavor: "seurat", n_top_genes: 2000 },
        { flavor: "seurat", n_top_genes: 1000 },
        { flavor: "cell_ranger", n_top_genes: 2000 },
      ],
    });
    // The control has no "Keep"; the others do.
    expect(screen.getAllByRole("button", { name: /Keep this one/ })).toHaveLength(2);
    expect(screen.getByText(/1 robust, 1 depend on the genes, 0 unstable/)).toBeInTheDocument();
    // B, the less stable, comes first; clicking it highlights it on the plot.
    const clusterRows = screen.getAllByTitle("Highlight this cluster on the plot");
    expect(within(clusterRows[0]!).getByText("B")).toBeInTheDocument();
    expect(within(clusterRows[0]!).getByText("0.60")).toBeInTheDocument(); // mean of 0.4 and 0.8
    expect(within(clusterRows[0]!).getByText("stable in 1 of 2")).toBeInTheDocument();
    expect(within(clusterRows[1]!).getByText("robust")).toBeInTheDocument();
    // The highlight shows once the plot is coloured by the compared clustering.
    useSettingsStore.setState({ colorBy: "cell_type" });
    fireEvent.click(clusterRows[0]!);
    expect(useSelectionStore.getState().highlightedGroup).toEqual({ column: REF, value: "B" });
    expect(useSettingsStore.getState().colorBy).toBe(REF);
  });

  it("keeps a setting and colours the plot by it", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue(info);
    runWithRows();
    api.keepHvgSetting.mockResolvedValue({ column: `${REF}_hvg_seurat_1000`, created: true, n_clusters: 3 });
    api.getDataset.mockResolvedValue({ id: "ds", obs_columns: [] });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    const keep = await screen.findAllByRole("button", { name: /Keep this one/ });
    fireEvent.click(keep[0]!);
    await waitFor(() => expect(useSettingsStore.getState().colorBy).toBe(`${REF}_hvg_seurat_1000`));
    expect(api.keepHvgSetting).toHaveBeenCalledWith("ds", expect.objectContaining({ flavor: "seurat", n_top_genes: 1000 }));
    expect(api.getDataset).toHaveBeenCalledWith("ds");
    expect(screen.getAllByRole("button", { name: "Show" })).toHaveLength(1);
  });

  it("colours each cell by its cluster's mean Jaccard", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue(info);
    runWithRows();
    api.fetchEmbeddingBinary.mockResolvedValue(new ArrayBuffer(0));
    api.decodeArrowBuffer.mockReturnValue({ [REF]: new Int32Array([0, 1, 1, -1]) });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    fireEvent.click(await screen.findByRole("button", { name: /Colour plot by stability/ }));
    await waitFor(() => expect(useUnifiedViewStore.getState().overlayValues).not.toBeNull());
    const s = useUnifiedViewStore.getState();
    expect(api.fetchEmbeddingBinary).toHaveBeenCalledWith("ds", "X_umap", REF);
    expect(Array.from(s.overlayValues!.slice(0, 3)).map((v) => Number(v.toFixed(3)))).toEqual([0.925, 0.6, 0.6]);
    expect(Number.isNaN(s.overlayValues![3]!)).toBe(true);
    expect(s.scatterOverlay).toEqual({ type: "score", name: "Cluster stability (mean Jaccard)" });
    expect(s.overlayDatasetId).toBe("ds");
  });

  it("adds a setting and refuses a gene count it can't run", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue(info);
    renderPanel();
    await screen.findByText(/About 35 s for 3 settings/);
    fireEvent.change(screen.getByLabelText("Flavour to add"), { target: { value: "pearson_residuals" } });
    fireEvent.change(screen.getByLabelText("Genes to add"), { target: { value: "50000" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByText(/between 200 and 3,000/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Genes to add"), { target: { value: "3000" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    expect(screen.getByLabelText("Pearson residuals, 3,000 genes")).toBeChecked();
    expect(screen.getByText(/About 50 s for 4 settings/)).toBeInTheDocument();
    // Unticking every setting leaves nothing to run.
    for (const box of screen.getAllByRole("checkbox")) fireEvent.click(box);
    expect(screen.getByRole("button", { name: "Run" })).toBeDisabled();
  });

  it("warns when the control does not reproduce the stored clusters", async () => {
    api.getHvgSensitivityInfo.mockResolvedValue(info);
    api.runHvgSensitivity.mockImplementation(async (_id, _body, h) => {
      h.onResult(result(defaults[0]!, [m("A", 0.9), m("B", 0.9)], { ari: 0.93 }));
    });
    renderPanel();
    fireEvent.click(await screen.findByRole("button", { name: "Run" }));
    expect(await screen.findByText(/Re-running the current setting gives ARI 0.93, not 1/)).toBeInTheDocument();
  });
});
