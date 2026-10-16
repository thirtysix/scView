import { create } from "zustand";
import type { DatasetInfo } from "@/api/types";
import { useSelectionStore } from "@/stores/selectionStore";
import { useSettingsStore } from "@/stores/settingsStore";
import { useUnifiedViewStore } from "@/stores/unifiedViewStore";

/** Per-dataset view state lives in other stores; a selection of cell indices or a
 *  column name from the previous dataset means nothing (or the wrong thing) in
 *  the next one, so it is reset when the dataset changes. */
function resetViewForDataset(dataset: DatasetInfo | null) {
  const columns = new Set((dataset?.obs_columns ?? []).map((c) => c.name));
  const keep = (col: string) => (col && columns.has(col) ? col : "");
  useSelectionStore.getState().clearSelection();
  useSelectionStore.getState().setHighlight(null);
  const settings = useSettingsStore.getState();
  settings.setColorBy(keep(settings.colorBy) || keep(dataset?.active_clustering ?? ""));
  const unified = useUnifiedViewStore.getState();
  unified.setGroupByColumn(keep(unified.groupByColumn));
  unified.clearOverlayState();
}

interface DatasetState {
  currentDatasetId: string | null;
  currentDataset: DatasetInfo | null;
  availableDatasets: DatasetInfo[];
  isUploading: boolean;
  uploadProgress: number;
  setCurrentDataset: (dataset: DatasetInfo | null) => void;
  setCurrentDatasetId: (id: string | null) => void;
  setAvailableDatasets: (datasets: DatasetInfo[]) => void;
  setUploading: (uploading: boolean) => void;
  setUploadProgress: (progress: number) => void;
}

export const useDatasetStore = create<DatasetState>((set, get) => ({
  currentDatasetId: null,
  currentDataset: null,
  availableDatasets: [],
  isUploading: false,
  uploadProgress: 0,
  setCurrentDataset: (dataset) => {
    const changed = (dataset?.id ?? null) !== get().currentDatasetId;
    set({ currentDataset: dataset, currentDatasetId: dataset?.id ?? null });
    if (changed) resetViewForDataset(dataset);
  },
  setCurrentDatasetId: (id) => set({ currentDatasetId: id }),
  setAvailableDatasets: (datasets) => set({ availableDatasets: datasets }),
  setUploading: (uploading) => set({ isUploading: uploading }),
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
}));
