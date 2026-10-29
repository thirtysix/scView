import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import type { SpotCheck, SpotInfo, SpotReferenceCheck, SpotSummary } from "@/api/spotDeconvolution";
import type { DatasetInfo } from "@/api/types";

const api = vi.hoisted(() => ({
  getSpotDeconvolution: vi.fn(),
  checkSpotReference: vi.fn(),
  getSpotShare: vi.fn(),
  runSpotDeconvolution: vi.fn(),
  getSpotCheck: vi.fn(),
  runSpotCheck: vi.fn(),
}));
vi.mock("@/api/spotDeconvolution", async (orig) => ({
  ...(await orig<typeof import("@/api/spotDeconvolution")>()),
  ...api,
}));
const refresh = vi.hoisted(() => vi.fn(async () => ({})));
vi.mock("@/hooks/useDatasetRefresh", () => ({ useDatasetRefresh: () => refresh }));

import { SpotCompositionSubtab } from "./SpotCompositionSubtab";
import { fmtShare, heatmapLayout, shareFill, shareLabel, toOverlayValues, typeRows } from "@/lib/spotComposition";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

const SPATIAL = {
  technology: "visium" as const, unit: "spot" as const, library_column: "library_id",
  libraries: [{ id: "lib1", n_obs: 4, images: {}, spot_diameter_fullres: 89, bounds: [0, 0, 1, 1] as [number, number, number, number] }],
};

function ds(id: string, spatial: DatasetInfo["spatial"] = null): DatasetInfo {
  return {
    id, name: id === "d1" ? "Lymph node" : "Atlas", filename: "", n_cells: 4, n_genes: 100, status: "ready",
    available_embeddings: [], created_at: "", spatial, obs_columns: [],
  };
}

function setDataset() {
  const spots = ds("d1", SPATIAL);
  useDatasetStore.setState({ currentDatasetId: "d1", currentDataset: spots, availableDatasets: [spots, ds("ref")] });
}

const INFO: SpotInfo = {
  methods: ["rctd", "nnls"], method: "rctd", method_names: { rctd: "RCTD-style", nnls: "NNLS" }, min_type_cells: 25,
  max_types: 50, min_spot_umi: 100, min_shared_genes: 500, available: true, reason: null, saved: null, n_spots: 4,
  running: false, note: "",
};
const CHECK: SpotReferenceCheck = {
  ok: true, reason: null, n_cells: 1000, n_genes: 2000, label_columns: [{ column: "Subset", n_labels: 3 }],
  celltype_column: "Subset",
  types: [
    { type: "B_GC", n_cells: 400, small: false }, { type: "T_naive", n_cells: 590, small: false },
    { type: "Mast", n_cells: 10, small: true },
  ],
  n_small: 1, species: "human", dataset_species: "human", shared_genes_by_name: 1900, min_shared_genes: 500,
  estimate: { peak_bytes: 2 * 1024 ** 3, seconds: 60, seconds_range: [30, 120], note: "about 60 s" },
};
const RESULT: SpotSummary = {
  result_id: "r1", method: "rctd", method_name: "RCTD-style Poisson-lognormal fit", types: ["B_GC", "T_naive"],
  type_stats: [
    { type: "B_GC", n_reference_cells: 400, cell_size: 5000, rna_mean: 0.3, rna_max: 0.9, cell_mean: 0.25, cell_max: 0.88, n_top: 1 },
    { type: "T_naive", n_reference_cells: 590, cell_size: 3000, rna_mean: 0.7, rna_max: 1, cell_mean: 0.75, cell_max: 1, n_top: 3 },
  ],
  small_types: ["Mast"], reference: { id: "ref", name: "Atlas", celltype_column: "Subset", n_reference_cells: 990 },
  celltype_column: "Subset", sources: { reference: "X", spots: "X", renamed: null }, s_e: 0.4,
  libraries: [{ id: "lib1", n_spots: 4, n_fitted: 4, s_g: 0.62, bulk: [0.3, 0.7] }],
  n_shared_genes: 1900, n_genes: 1200, n_platform_genes: 3000, n_spots: 4, n_fitted: 4, n_skipped: 0, n_unconverged: 0,
  n_clipped: 0, cluster_column: "leiden",
  clusters: { rna: { "0": [0.8, 0.2], "1": [0.1, 0.9] }, cells: { "0": [0.7, 0.3], "1": [0.05, 0.95] } },
  columns: { rna: "spot_deconv", cells: "spot_deconv_cells", top: "spot_deconv_top", top_share: "spot_deconv_top_share" },
  seconds: 12, notes: ["Left out of the reference (fewer than 25 cells): Mast."], note: "",
};
const SELF_CHECK: SpotCheck = {
  id: "c1", types: ["B_GC", "T_naive"], n_spots: 1000, cells_per_spot: [2, 10], depths: [20000, 5000],
  reference: { id: "ref", name: "Atlas", celltype_column: "Subset" }, seconds: 30, note: "Synthetic spots.",
  runs: [
    { method: "rctd", depth: 20000, s_e: 0.2, median_umi: 19800, n_genes: 1200, rmse: 0.031, jsd: 0.012, aupr: 0.97, n_spots: 1000,
      per_type: [{ type: "B_GC", r: 0.98, bias: 0, rmse: 0.03 }, { type: "T_naive", r: 0.97, bias: 0, rmse: 0.03 }] },
    { method: "nnls", depth: 20000, s_e: null, median_umi: 19800, n_genes: 1200, rmse: 0.071, jsd: 0.04, aupr: 0.9, n_spots: 1000,
      per_type: [{ type: "B_GC", r: 0.9, bias: 0, rmse: 0.07 }, { type: "T_naive", r: 0.9, bias: 0, rmse: 0.07 }] },
  ],
};

beforeEach(() => {
  vi.clearAllMocks();
  setDataset();
  useUnifiedViewStore.getState().clearOverlayState();
  api.getSpotDeconvolution.mockResolvedValue(INFO);
  api.checkSpotReference.mockResolvedValue(CHECK);
  api.getSpotCheck.mockRejectedValue(new Error("API error 404: none"));
  api.getSpotShare.mockResolvedValue({ values: [0.1, null, 0.5, 1] });
});

async function pickReference() {
  fireEvent.change(await screen.findByLabelText("Reference dataset"), { target: { value: "ref" } });
  await screen.findByTestId("spot-reference-check");
}

describe("SpotCompositionSubtab", () => {
  it("shows the reference check first, then fits with progress and lists the types", async () => {
    let finish: (r: SpotSummary) => void = () => {};
    api.runSpotDeconvolution.mockImplementation((_id, _body, h) => {
      h.onProgress?.("Fitting the spots", 2, 4);
      return new Promise((res) => (finish = res));
    });
    render(<SpotCompositionSubtab />);
    await pickReference();
    const chk = screen.getByTestId("spot-reference-check");
    expect(chk.textContent).toContain("1,000 reference cells, 2 types with 25+ cells");
    expect(chk.textContent).toContain("reference human, this dataset human");
    expect(chk.textContent).toContain("Left out (fewer than 25 cells): Mast (10)");
    expect(chk.textContent).toContain("about 60 s");
    expect(api.checkSpotReference).toHaveBeenCalledWith("d1", "ref", null, "rctd");
    expect((screen.getByLabelText("Reference cell-type column") as HTMLSelectElement).value).toBe("Subset");

    fireEvent.click(screen.getByRole("button", { name: /Fit spots/ }));
    expect(await screen.findByTestId("spot-progress")).toHaveTextContent("Fitting the spots (2/4)");
    expect(api.runSpotDeconvolution.mock.calls[0]![1]).toEqual({ reference_id: "ref", celltype_column: "Subset", method: "rctd" });
    finish(RESULT);
    const res = await screen.findByTestId("spot-composition-result");
    expect(refresh).toHaveBeenCalledWith("d1", "spot_deconv_top");
    const rows = within(res).getAllByRole("row");
    expect(rows[1]).toHaveTextContent("T_naive70%100%3"); // largest mean share first
    expect(res.textContent).toContain("spot noise 0.4; platform effect 0.62");
    expect(screen.getByTestId("spot-caveats").textContent).toContain("missing from the reference");
  });

  it("colours the tissue by a type's share on a fixed scale and follows the RNA / cell toggle", async () => {
    api.getSpotDeconvolution.mockResolvedValue({ ...INFO, saved: RESULT });
    render(<SpotCompositionSubtab />);
    fireEvent.click(await screen.findByRole("button", { name: "B_GC" }));
    await waitFor(() => expect(useUnifiedViewStore.getState().scatterOverlay).not.toBeNull());
    let s = useUnifiedViewStore.getState();
    expect(s.scatterOverlay).toEqual({ type: "score", name: "B_GC RNA share", spotShare: { type: "B_GC", kind: "rna" } });
    expect(Array.from(s.overlayValues!).map((v) => (Number.isNaN(v) ? "nan" : v))).toEqual([
      expect.closeTo(0.1), "nan", 0.5, 1,
    ]);
    expect(s.overlayDatasetId).toBe("d1");
    fireEvent.click(screen.getByLabelText("cells (estimated)"));
    await waitFor(() => expect(api.getSpotShare).toHaveBeenLastCalledWith("d1", "B_GC", "cell"));
    await waitFor(() => expect(useUnifiedViewStore.getState().overlayLabel).toBe("B_GC cell share (est.)"));
    // The heatmap now shows cell shares: cluster 1's T_naive 95%.
    expect(screen.getByTestId("spot-heatmap").textContent).toContain("95");
    // Dominant type: an obs colouring, the overlay cleared.
    fireEvent.click(screen.getByRole("button", { name: "Colour by dominant type" }));
    s = useUnifiedViewStore.getState();
    expect(s.scatterOverlay).toBeNull();
    expect(useSettingsStore.getState().colorBy).toBe("spot_deconv_top");
  });

  it("explains a dataset it cannot fit and a refused reference", async () => {
    api.getSpotDeconvolution.mockResolvedValueOnce({ ...INFO, available: false, reason: "Spot composition needs Visium spots." });
    const { unmount } = render(<SpotCompositionSubtab />);
    expect(await screen.findByText("Spot composition needs Visium spots.")).toBeInTheDocument();
    expect(screen.queryByLabelText("Reference dataset")).toBeNull();
    unmount();
    api.checkSpotReference.mockResolvedValue({ ok: false, reason: "The reference looks mouse and this dataset human." });
    render(<SpotCompositionSubtab />);
    fireEvent.change(await screen.findByLabelText("Reference dataset"), { target: { value: "ref" } });
    expect(await screen.findByText(/looks mouse/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Fit spots/ })).toBeDisabled();
  });

  it("shows a run error in words", async () => {
    api.runSpotDeconvolution.mockRejectedValue(new Error('API error 422: {"detail": "No raw counts."}'));
    render(<SpotCompositionSubtab />);
    await pickReference();
    fireEvent.click(screen.getByRole("button", { name: /Fit spots/ }));
    expect(await screen.findByText("No raw counts.")).toBeInTheDocument();
  });

  it("runs the self-check and shows both methods", async () => {
    api.runSpotCheck.mockResolvedValue(SELF_CHECK);
    render(<SpotCompositionSubtab />);
    await pickReference();
    fireEvent.click(screen.getByRole("button", { name: /Run self-check/ }));
    const table = await screen.findByTestId("spot-check-table");
    expect(api.runSpotCheck.mock.calls[0]![1]).toEqual({ reference_id: "ref", celltype_column: "Subset" });
    expect(table.textContent).toContain("0.031");
    expect(table.textContent).toContain("NNLS (baseline)");
    expect(screen.getByTestId("spot-check").textContent).toContain("B_GC 0.98");
  });
});

describe("spot composition helpers", () => {
  it("formats, orders and lays out the shares", () => {
    expect(fmtShare(0.1234)).toBe("12%");
    expect(fmtShare(0.0412)).toBe("4.1%");
    expect(fmtShare(0.0004)).toBe("<0.1%");
    expect(fmtShare(null)).toBe("-");
    expect(typeRows(RESULT, "cell").map((r) => [r.type, r.mean])).toEqual([["T_naive", 0.75], ["B_GC", 0.25]]);
    expect(shareLabel("B", "rna")).toBe("B RNA share");
    expect(Array.from(toOverlayValues([0.5, null])).map(String)).toEqual(["0.5", "NaN"]);
    const h = heatmapLayout(RESULT, "rna", 1);
    expect(h.clusters).toEqual(["0", "1"]);
    expect(h.types).toEqual([{ type: "T_naive", index: 1 }]);
    expect(shareFill(0)).toBe("rgba(37, 99, 235, 0.080)");
    expect(shareFill(1)).toBe("rgba(37, 99, 235, 1.000)");
  });
});
