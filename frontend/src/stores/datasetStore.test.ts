import { describe, it, expect, beforeEach } from "vitest";
import { useDatasetStore } from "./datasetStore";
import { useSelectionStore } from "./selectionStore";
import { useSettingsStore } from "./settingsStore";
import { useUnifiedViewStore } from "./unifiedViewStore";
import type { DatasetInfo } from "@/api/types";

const ds = (id: string, cols: string[], active?: string) =>
  ({ id, obs_columns: cols.map((name) => ({ name })), active_clustering: active }) as unknown as DatasetInfo;

describe("switching datasets", () => {
  beforeEach(() => {
    useDatasetStore.getState().setCurrentDataset(ds("kang", ["cluster", "cell_type", "stim"], "cluster"));
    useSelectionStore.getState().setSelection(new Set([1, 2, 3]));
    useSelectionStore.getState().setHighlight({ column: "cluster", value: "NK" });
    useSettingsStore.getState().setColorBy("cell_type");
    useUnifiedViewStore.getState().setGroupByColumn("stim");
  });

  it("clears the selection and drops columns the new dataset lacks", () => {
    useDatasetStore.getState().setCurrentDataset(ds("pbmc", ["leiden", "phase"], "leiden"));
    expect(useSelectionStore.getState().selectedCellIndices).toBeNull();
    expect(useSelectionStore.getState().highlightedGroup).toBeNull();
    expect(useSettingsStore.getState().colorBy).toBe("leiden"); // falls back to the clustering
    expect(useUnifiedViewStore.getState().groupByColumn).toBe("");
  });

  it("keeps columns that exist in the new dataset", () => {
    useDatasetStore.getState().setCurrentDataset(ds("kang2", ["cell_type", "stim"]));
    expect(useSettingsStore.getState().colorBy).toBe("cell_type");
    expect(useUnifiedViewStore.getState().groupByColumn).toBe("stim");
  });

  it("opens a dataset coloured by its best grouping when nothing was chosen", () => {
    const typed = (id: string) =>
      ({
        id,
        n_cells: 1000,
        active_clustering: "leiden_r1",
        obs_columns: [
          { name: "leiden_r1", dtype: "category", n_unique: 12, values: [] },
          { name: "cell_type", dtype: "category", n_unique: 9, values: [] },
        ],
      }) as unknown as DatasetInfo;
    useSettingsStore.getState().setDefaultColorBy("");
    useDatasetStore.getState().setCurrentDataset(typed("a"));
    expect(useSettingsStore.getState().colorBy).toBe("cell_type");
    expect(useSettingsStore.getState().colorByChosen).toBe(false);

    // A colouring the user picks is kept for the next dataset that has it...
    useSettingsStore.getState().setColorBy("leiden_r1");
    useDatasetStore.getState().setCurrentDataset(typed("b"));
    expect(useSettingsStore.getState().colorBy).toBe("leiden_r1");

    // ...but a default carried over from the previous dataset is re-picked.
    useSettingsStore.getState().setDefaultColorBy("leiden_r1");
    useDatasetStore.getState().setCurrentDataset(typed("c"));
    expect(useSettingsStore.getState().colorBy).toBe("cell_type");
  });

  it("opens a dataset on its hand labels, whatever the column is called", () => {
    const labelled = (id: string, users?: string[]) =>
      ({
        id,
        n_cells: 1000,
        obs_columns: [
          { name: "cell_type", dtype: "category", n_unique: 9, values: [] },
          { name: "T cell states", dtype: "category", n_unique: 4, values: [] },
        ],
        user_annotation_columns: users,
      }) as unknown as DatasetInfo;
    useSettingsStore.getState().setDefaultColorBy("");
    useDatasetStore.getState().setCurrentDataset(labelled("a", ["T cell states"]));
    expect(useSettingsStore.getState().colorBy).toBe("T cell states");
    // Not listed as hand labels (an older backend, or an imported column): cell type first.
    useDatasetStore.getState().setCurrentDataset(labelled("b"));
    expect(useSettingsStore.getState().colorBy).toBe("cell_type");
  });

  it("leaves the view alone when the same dataset is refreshed", () => {
    useDatasetStore.getState().setCurrentDataset(ds("kang", ["cluster", "cell_type", "stim"], "cluster"));
    expect(useSelectionStore.getState().selectedCellIndices?.size).toBe(3);
  });
});
