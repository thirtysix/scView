import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";

const api = vi.hoisted(() => ({ getDataset: vi.fn() }));
vi.mock("@/api/datasets", () => ({ getDataset: api.getDataset }));

import type { DatasetInfo } from "@/api/types";
import { useDatasetStore } from "@/stores/datasetStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useViewStore } from "@/stores/viewStore";
import { useSessionRestore } from "./useSessionRestore";

const KEY = "scview.session.v1";
const ds: DatasetInfo = {
  id: "ds1",
  name: "Kang",
  filename: "k.h5ad",
  n_cells: 10,
  n_genes: 5,
  status: "ready",
  available_embeddings: ["X_umap", "X_tsne"],
  obs_columns: [
    { name: "leiden", dtype: "category", n_unique: 2, values: ["0", "1"] },
    { name: "cell_type", dtype: "category", n_unique: 2, values: ["B", "T"] },
  ],
  created_at: "",
};

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
  useViewStore.setState({ activePanel: "load" });
  useSettingsStore.setState({ colorBy: "", colorByChosen: false, embedding: "X_umap", categoricalPalette: "cvd" });
  api.getDataset.mockResolvedValue(ds);
});

describe("useSessionRestore", () => {
  it("still restores the older flat snapshot", async () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ datasetId: "ds1", activePanel: "unified", colorBy: "leiden", embedding: "X_tsne", categoricalPalette: "classic" }),
    );
    renderHook(() => useSessionRestore());
    await waitFor(() => expect(useViewStore.getState().activePanel).toBe("unified"));
    expect(api.getDataset).toHaveBeenCalledWith("ds1");
    expect(useDatasetStore.getState().currentDatasetId).toBe("ds1");
    const s = useSettingsStore.getState();
    expect([s.colorBy, s.colorByChosen, s.embedding, s.categoricalPalette]).toEqual(["leiden", true, "X_tsne", "classic"]);
  });

  it("stores the slice in the session format once restored, and reads it back", async () => {
    renderHook(() => useSessionRestore());
    await act(async () => {
      useDatasetStore.getState().setCurrentDataset(ds); // opens on cell_type, its default
      useViewStore.getState().setPanel("markers");
    });
    const stored = JSON.parse(localStorage.getItem(KEY)!);
    expect(stored).toEqual({
      format: "scview-session",
      version: 1,
      dataset: { id: "ds1" },
      view: { panel: "markers", color_by: "cell_type", color_by_chosen: false, embedding: "X_umap" },
      colors: {
        categorical_palette: "cvd",
        continuous_colormap: "viridis",
        diverging_colormap: "RdBu",
        signed_colormap_mode: "auto",
        clip: { preset: "none" },
      },
    });

    // a reload
    useDatasetStore.setState({ currentDataset: null, currentDatasetId: null });
    useViewStore.setState({ activePanel: "load" });
    renderHook(() => useSessionRestore());
    await waitFor(() => expect(useViewStore.getState().activePanel).toBe("markers"));
    expect(useSettingsStore.getState().colorByChosen).toBe(false);
  });

  it("does not overwrite the snapshot with the start-up defaults before restoring it", async () => {
    let finish: (d: DatasetInfo) => void = () => {};
    api.getDataset.mockReturnValue(new Promise<DatasetInfo>((r) => (finish = r)));
    const saved = { format: "scview-session", version: 1, dataset: { id: "ds1" }, view: { panel: "unified" } };
    localStorage.setItem(KEY, JSON.stringify(saved));
    renderHook(() => useSessionRestore());
    expect(JSON.parse(localStorage.getItem(KEY)!)).toEqual(saved); // still loading the dataset
    await act(async () => finish(ds));
    await waitFor(() => expect(useViewStore.getState().activePanel).toBe("unified"));
  });

  it("stands aside for a share link: the link's view opens, not the last workspace", async () => {
    localStorage.setItem(KEY, JSON.stringify({ format: "scview-session", version: 1, dataset: { id: "ds1" }, view: { panel: "unified" } }));
    window.history.replaceState(null, "", "/#view=v1.z.abc");
    try {
      renderHook(() => useSessionRestore());
      expect(api.getDataset).not.toHaveBeenCalled();
      expect(useViewStore.getState().activePanel).toBe("load");
      // ...and keeps the slice up to date from then on.
      await act(async () => useViewStore.getState().setPanel("markers"));
      expect(JSON.parse(localStorage.getItem(KEY)!).view.panel).toBe("markers");
    } finally {
      window.history.replaceState(null, "", "/");
    }
  });

  it("ignores a broken snapshot", () => {
    localStorage.setItem(KEY, "{not json");
    renderHook(() => useSessionRestore());
    expect(api.getDataset).not.toHaveBeenCalled();
    expect(useViewStore.getState().activePanel).toBe("load");
  });
});
