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

  it("leaves the view alone when the same dataset is refreshed", () => {
    useDatasetStore.getState().setCurrentDataset(ds("kang", ["cluster", "cell_type", "stim"], "cluster"));
    expect(useSelectionStore.getState().selectedCellIndices?.size).toBe(3);
  });
});
