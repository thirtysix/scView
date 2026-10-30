import { create } from "zustand";

/**
 * The tissue plot's own settings (Unified View, Embedding "Tissue"): the
 * section shown (one library at a time), and the H&E image under the spots.
 */
interface TissueViewState {
  /** Library id shown; null = the dataset's first */
  library: string | null;
  showImage: boolean;
  /** 0..1 */
  imageOpacity: number;
  /** The dataset last opened in Unified View: a spatial one opens on its tissue plot, once */
  openedOn: string | null;
  setOpenedOn: (datasetId: string | null) => void;
  setLibrary: (library: string | null) => void;
  setShowImage: (show: boolean) => void;
  setImageOpacity: (opacity: number) => void;
}

export const useTissueViewStore = create<TissueViewState>((set) => ({
  library: null,
  showImage: true,
  imageOpacity: 1,
  openedOn: null,
  setOpenedOn: (openedOn) => set({ openedOn }),
  setLibrary: (library) => set({ library }),
  setShowImage: (showImage) => set({ showImage }),
  setImageOpacity: (imageOpacity) => set({ imageOpacity: Math.min(1, Math.max(0, imageOpacity)) }),
}));
