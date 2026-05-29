import { create } from "zustand";
import type { DatasetInfo } from "@/api/types";

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

export const useDatasetStore = create<DatasetState>((set) => ({
  currentDatasetId: null,
  currentDataset: null,
  availableDatasets: [],
  isUploading: false,
  uploadProgress: 0,
  setCurrentDataset: (dataset) =>
    set({ currentDataset: dataset, currentDatasetId: dataset?.id ?? null }),
  setCurrentDatasetId: (id) => set({ currentDatasetId: id }),
  setAvailableDatasets: (datasets) => set({ availableDatasets: datasets }),
  setUploading: (uploading) => set({ isUploading: uploading }),
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
}));
