import { describe, it, expect, vi, beforeEach } from "vitest";

const api = vi.hoisted(() => ({ getDataset: vi.fn(), cellsMatching: vi.fn(), apiFetch: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));
vi.mock("@/api/cells", () => ({ cellsMatching: api.cellsMatching }));
vi.mock("@/api/client", () => ({ apiFetch: api.apiFetch }));

import type { DatasetInfo } from "@/api/types";
import type { CellFilter } from "@/lib/cellFilter";
import { parseSession, sessionJson, type SessionFile } from "@/lib/session";
import { captureSession, restoreSession } from "@/lib/sessionRestore";
import { DE_DEFAULTS, useDESettingsStore } from "@/stores/deSettingsStore";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useSplitViewStore } from "@/stores/splitViewStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";
import { useViewStore } from "@/stores/viewStore";

const kang = (extra: Partial<DatasetInfo> = {}): DatasetInfo => ({
  id: "kang",
  name: "Kang",
  filename: "kang.h5ad",
  n_cells: 100,
  n_genes: 50,
  status: "ready",
  available_embeddings: ["X_umap", "X_pca"],
  obs_columns: [
    { name: "cell_type", dtype: "category", n_unique: 3, values: ["B", "CD14 Mono", "NK"] },
    { name: "stim", dtype: "category", n_unique: 2, values: ["CTRL", "STIM"] },
    { name: "ind", dtype: "category", n_unique: 4, values: ["d1", "d2", "d3", "d4"] },
    { name: "n_counts", dtype: "float64", n_unique: 90 },
  ],
  expression_layers: [
    { key: "raw", label: "Log-normalized (raw)", n_genes: 50 },
    { key: "X", label: "Main", n_genes: 50 },
  ],
  default_expression_layer: "raw",
  cells_fingerprint: "fp-kang",
  created_at: "",
  ...extra,
});

const filter: CellFilter = {
  op: "and",
  layer: "raw",
  items: [
    { kind: "gene", gene: "CD14", op: ">", value: 1 },
    { kind: "category", column: "stim", values: ["STIM"] },
  ],
};

function resetStores() {
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useViewStore.setState({ activePanel: "load" });
  useSettingsStore.setState({
    embedding: "X_umap",
    colorBy: "",
    colorByChosen: false,
    expressionLayer: "",
    pointSize: 2,
    opacity: 0.8,
    plotBackground: "white",
    categoricalPalette: "cvd",
    continuousColormap: "viridis",
    maxRenderedCells: 100_000,
  });
  useSplitViewStore.setState({ splitBy: null, compareColumns: null, showContext: true });
  useSelectionStore.setState({ selectedCellIndices: null, selectionFilter: null, highlightedGroup: null });
  useUnifiedViewStore.setState({
    activeSubtab: "markers",
    groupByColumn: "",
    centroidLabels: null,
    camera: null,
    pendingCamera: null,
    pendingOverlay: null,
    scatterOverlay: null,
    overlayDatasetId: null,
  });
  useDESettingsStore.setState({ ...DE_DEFAULTS, scopedHighlight: null });
}

/** A workspace on Kang with everything set, as the user would leave it. */
function buildWorkspace(selection: "filter" | "lasso") {
  useDatasetStore.getState().setCurrentDataset(kang());
  useViewStore.getState().setPanel("unified");
  const s = useSettingsStore.getState();
  s.setEmbedding("X_pca");
  s.setColorBy("stim");
  s.setExpressionLayer("X");
  s.setCategoricalPalette("tableau20");
  s.setContinuousColormap("magma");
  s.setPointSize(4);
  s.setOpacity(0.5);
  s.setPlotBackground("dark");
  const u = useUnifiedViewStore.getState();
  u.setActiveSubtab("de");
  u.setGroupByColumn("cell_type");
  u.setCentroidLabels({ key: "stim", show: false });
  u.setCamera({ datasetId: "kang", embedding: "X_pca", camera: { target: [3, 4, 0], zoom: 2.5 } });
  u.setScatterOverlay({ type: "expression", gene: "CD14" });
  u.setOverlayDatasetId("kang");
  useSplitViewStore.getState().setCompareColumns(["cell_type", "stim"]);
  useSelectionStore.getState().setHighlight({ column: "cell_type", value: "CD14 Mono" });
  if (selection === "filter") useSelectionStore.getState().setSelection(new Set([1, 3, 9]), filter);
  else useSelectionStore.getState().setSelection(new Set([2, 4, 99]));
  useDESettingsStore.getState().update({
    mode: "groups",
    splitBy: "stim",
    groupA: "STIM",
    groupB: "CTRL",
    withinCol: "cell_type",
    withinVal: "NK",
    test: "pseudobulk",
    replicateKey: "ind",
    fcThreshold: 2,
    minPct: 0.25,
  });
}

/** Save, start again from scratch, read the file back: what "Open session" gets. */
function saveAndReset(): SessionFile {
  const file = sessionJson(captureSession(new Date("2026-10-16T20:00:00Z")));
  resetStores();
  return parseSession(JSON.parse(file));
}

beforeEach(() => {
  vi.clearAllMocks();
  resetStores();
  api.getDataset.mockResolvedValue(kang());
  api.cellsMatching.mockResolvedValue(new Set([1, 3, 9]));
  api.apiFetch.mockResolvedValue({ results: ["CD14"] });
});

describe("save then open on the same dataset", () => {
  it("saves a TF or pathway activity colouring and computes it again on open", async () => {
    buildWorkspace("lasso");
    const u = useUnifiedViewStore.getState();
    u.setScatterOverlay({ type: "score", name: "STAT1 (TF activity)", activity: { network: "collectri", source: "STAT1" } });
    const session = saveAndReset();
    expect(session.view?.overlay).toEqual({ type: "activity", network: "collectri", source: "STAT1" });
    await restoreSession(session);
    expect(useUnifiedViewStore.getState().pendingOverlay).toEqual({
      datasetId: "kang",
      overlay: { type: "activity", network: "collectri", source: "STAT1" },
    });
  });

  it("saves a protein colouring (CITE-seq) and restores it only where the protein exists", async () => {
    const protein = { names: ["CD19"], controls: [], layer: "protein", counts_layer: "protein_counts", normalisation: "CLR" };
    api.getDataset.mockResolvedValue({ ...kang(), protein });
    buildWorkspace("lasso");
    useUnifiedViewStore.getState().setScatterOverlay({ type: "expression", gene: "CD19", layer: "protein" });
    const file = sessionJson(captureSession(new Date("2026-10-16T20:00:00Z")));
    resetStores();
    const session = parseSession(JSON.parse(file));
    expect(session.view?.overlay).toEqual({ type: "expression", gene: "CD19", layer: "protein" });
    await restoreSession(session);
    expect(useUnifiedViewStore.getState().pendingOverlay).toEqual({
      datasetId: "kang",
      overlay: { type: "expression", gene: "CD19", layer: "protein" },
    });
    resetStores();
    api.getDataset.mockResolvedValue(kang()); // no proteins
    const report = await restoreSession(session);
    expect(report.notes).toContain("Protein overlay CD19: the protein is not in this dataset.");
    expect(useUnifiedViewStore.getState().pendingOverlay).toBeNull();
  });

  it("restores the whole workspace", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    expect(session.dataset).toEqual({ id: "kang", name: "Kang", n_cells: 100, cells_fingerprint: "fp-kang" });
    expect(session.selection).toEqual({ n_cells: 100, indices: [2, 4, 99] });

    const report = await restoreSession(session);
    expect(report).toEqual({ headline: "Opened “Kang”.", notes: [], level: "ok" });
    expect(useDatasetStore.getState().currentDatasetId).toBe("kang");
    expect(useViewStore.getState().activePanel).toBe("unified");
    const s = useSettingsStore.getState();
    expect([s.embedding, s.colorBy, s.colorByChosen, s.expressionLayer]).toEqual(["X_pca", "stim", true, "X"]);
    expect([s.categoricalPalette, s.continuousColormap, s.pointSize, s.opacity, s.plotBackground]).toEqual([
      "tableau20", "magma", 4, 0.5, "dark",
    ]);
    const u = useUnifiedViewStore.getState();
    expect([u.activeSubtab, u.groupByColumn]).toEqual(["de", "cell_type"]);
    expect(u.centroidLabels).toEqual({ key: "stim", show: false });
    expect(u.pendingCamera).toEqual({ datasetId: "kang", embedding: "X_pca", camera: { target: [3, 4, 0], zoom: 2.5 } });
    expect(u.pendingOverlay).toEqual({ datasetId: "kang", overlay: { type: "expression", gene: "CD14" } });
    expect(useSplitViewStore.getState().compareColumns).toEqual(["cell_type", "stim"]);
    const sel = useSelectionStore.getState();
    expect(sel.highlightedGroup).toEqual({ column: "cell_type", value: "CD14 Mono" });
    expect([...sel.selectedCellIndices!].sort((a, b) => a - b)).toEqual([2, 4, 99]);
    expect(sel.selectionFilter).toBeNull();
    const de = useDESettingsStore.getState();
    expect([de.mode, de.splitBy, de.groupA, de.groupB, de.withinCol, de.withinVal, de.test, de.replicateKey, de.fcThreshold, de.minPct])
      .toEqual(["groups", "stim", "STIM", "CTRL", "cell_type", "NK", "pseudobulk", "ind", 2, 0.25]);
    // the restored highlight must not overwrite the restored scope
    expect(de.scopedHighlight).toEqual({ column: "cell_type", value: "CD14 Mono" });
  });

  it("saves a filter's selection as the filter and evaluates it again", async () => {
    buildWorkspace("filter");
    const session = saveAndReset();
    expect(session.selection).toEqual({ filter });
    await restoreSession(session);
    expect(api.cellsMatching).toHaveBeenCalledWith("kang", filter);
    expect(useSelectionStore.getState().selectionFilter).toEqual(filter);
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([1, 3, 9]);
  });

  it("restores on the open dataset too, starting from its defaults", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    useDatasetStore.getState().setCurrentDataset(kang());
    useSelectionStore.getState().setSelection(new Set([50])); // something to be replaced
    useSplitViewStore.getState().setSplitBy("stim");
    await restoreSession({ ...session, selection: null, view: { ...session.view, split_by: null } });
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
    expect(useSplitViewStore.getState().splitBy).toBeNull();
  });
});

describe("when the dataset differs or is missing", () => {
  it("says the cells changed and skips positions and the camera, keeping what goes by name", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    api.getDataset.mockResolvedValue(kang({ n_cells: 90, cells_fingerprint: "fp-other" }));
    const report = await restoreSession(session);
    expect(report.level).toBe("warning");
    expect(report.headline).toBe("Opened “Kang”, but its cells have changed since the session was saved (100 cells then, 90 now).");
    expect(report.notes).toEqual([
      "Camera: not restored, the cells differ.",
      "Selection of 3 cells: not restored, it lists cells by position and the cells differ.",
    ]);
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
    expect(useUnifiedViewStore.getState().pendingCamera).toBeNull();
    expect(useSettingsStore.getState().colorBy).toBe("stim");
  });

  it("restores a missing dataset's session on the open dataset with the same cells", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    api.getDataset.mockRejectedValue(new Error("API error 404: not found"));
    useDatasetStore.getState().setCurrentDataset(kang({ id: "kang-copy", name: "Kang copy" }));
    const report = await restoreSession(session);
    expect(report.headline).toMatch(/not in this library; the open dataset “Kang copy” has the same cells/);
    expect(report.notes).toEqual([]);
    expect(useDatasetStore.getState().currentDatasetId).toBe("kang-copy");
    expect([...useSelectionStore.getState().selectedCellIndices!].sort((a, b) => a - b)).toEqual([2, 4, 99]);
  });

  it("applies what fits to another open dataset and lists the rest", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    api.getDataset.mockRejectedValue(new Error("API error 404: not found"));
    api.apiFetch.mockResolvedValue({ results: ["CD14A"] }); // CD14 itself is not there
    useDatasetStore.getState().setCurrentDataset(
      kang({
        id: "pbmc",
        name: "PBMC 1k",
        cells_fingerprint: "fp-pbmc",
        available_embeddings: ["X_umap"],
        obs_columns: [{ name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] }],
      }),
    );
    const report = await restoreSession(session);
    expect(report.level).toBe("warning");
    expect(report.headline).toBe("The session's dataset “Kang” is not in this library. What fits the open dataset “PBMC 1k” was restored.");
    expect(report.notes).toEqual([
      "Embedding PCA: not in this dataset.",
      "Colour by “stim”: this dataset has no such column.",
      "Compare colourings: “stim” not in this dataset, so the comparison is off.",
      "Highlighted group “CD14 Mono” of “cell_type”: not in this dataset.",
      "Gene overlay CD14: the gene is not in this dataset.",
      "Camera: not restored, the cells differ.",
      "DE between groups: “stim” is not in this dataset.",
      "DE scope “cell_type = NK”: not in this dataset.",
      "Selection of 3 cells: not restored, it lists cells by position and the cells differ.",
    ]);
    expect(useDatasetStore.getState().currentDatasetId).toBe("pbmc");
    expect(useUnifiedViewStore.getState().groupByColumn).toBe("cell_type"); // fits
    expect(useSettingsStore.getState().categoricalPalette).toBe("tableau20");
  });

  it("restores only the colours with no dataset to restore onto", async () => {
    buildWorkspace("lasso");
    const session = saveAndReset();
    api.getDataset.mockRejectedValue(new Error("API error 404: not found"));
    const report = await restoreSession(session);
    expect(report.headline).toBe("The session's dataset “Kang” is not in this library, so only its colour settings were restored.");
    expect(useDatasetStore.getState().currentDataset).toBeNull();
    expect(useViewStore.getState().activePanel).toBe("load");
    expect(useSettingsStore.getState().plotBackground).toBe("dark");
  });

  it("stops on a server error instead of calling the dataset missing", async () => {
    api.getDataset.mockRejectedValue(new Error('API error 503: {"detail":"busy"}'));
    useDatasetStore.getState().setCurrentDataset(kang({ id: "other", name: "Other" }));
    await expect(
      restoreSession(parseSession({ format: "scview-session", version: 1, dataset: { id: "kang", name: "Kang" } })),
    ).rejects.toThrow(/dataset “Kang” could not be opened \(503/);
    expect(useDatasetStore.getState().currentDatasetId).toBe("other");
  });

  it("never uses cell positions beyond the dataset", async () => {
    const report = await restoreSession(
      parseSession({
        format: "scview-session",
        version: 1,
        dataset: { id: "kang", name: "Kang", n_cells: 100, cells_fingerprint: "fp-kang" },
        selection: { n_cells: 100, indices: [5, 100, 4000000] },
      }),
    );
    expect(report.notes).toEqual(["Selection: 2 cell positions beyond the dataset's cells were ignored."]);
    expect([...useSelectionStore.getState().selectedCellIndices!]).toEqual([5]);
  });

  it("reports a filter the dataset cannot answer", async () => {
    api.cellsMatching.mockRejectedValue(new Error("Gene 'CD14' is not in layer 'raw'."));
    const report = await restoreSession(
      parseSession({ format: "scview-session", version: 1, dataset: { id: "kang" }, selection: { filter } }),
    );
    expect(report.notes).toEqual(["Selection “CD14 > 1 AND stim = STIM”: Gene 'CD14' is not in layer 'raw'."]);
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
  });
});
